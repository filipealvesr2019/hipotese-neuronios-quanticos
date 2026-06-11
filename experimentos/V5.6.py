import numpy as np
import os
import json

# =========================
# V5.6 — Specialized Architecture Experts + Diversity Loss
# =========================
class V56_MoE:
    def __init__(self, input_dim, hidden_sizes, output_dim,
                 top_k=2, lr=0.01, ema=0.9, diversity_weight=0.01, flop_weight=1e-6):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.n_experts = len(hidden_sizes)
        self.top_k = top_k
        self.lr = lr
        self.ema = ema
        self.diversity_weight = diversity_weight
        self.flop_weight = flop_weight

        rng = np.random.RandomState(42)

        # Experts weights
        self.W1 = [rng.randn(input_dim, h) * 0.1 for h in hidden_sizes]
        self.b1 = [np.zeros(h) for h in hidden_sizes]
        self.W2 = [rng.randn(h, output_dim) * 0.1 for h in hidden_sizes]
        self.b2 = [np.zeros(output_dim) for _ in range(self.n_experts)]

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, self.n_experts) * 0.1
        self.gb2 = np.zeros(self.n_experts)

        # Expert performance tracking
        self.expert_perf = np.zeros(self.n_experts)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        B = X.shape[0]
        h_list, out_list = [], []
        for i in range(self.n_experts):
            h = np.maximum(0, X @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_list.append(h)
            out_list.append(out)

        # Gate forward
        g = np.maximum(0, X @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2
        gate_probs = self.softmax(logits)

        # Top-k routing
        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            for idx in topk_idx[b]:
                gate[b, idx] = 1.0 / self.top_k

        # Expert output weighted by gate
        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list

    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    def train_step(self, X, y):
        B = X.shape[0]
        logits, gate, out_list, h_list = self.forward(X)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        expert_acc = np.zeros(self.n_experts)

        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            # Add weight diversity gradient
            dW2_div = np.zeros_like(self.W2[i])
            for j in range(self.n_experts):
                if i != j:
                    min_h = min(self.W2[i].shape[0], self.W2[j].shape[0])
                    # gradient of np.mean(W_i * W_j) w.r.t W_i is W_j / size
                    dW2_div[:min_h] += self.W2[j][:min_h] / self.W2[j][:min_h].size

            self.W2[i] -= self.lr * (dW2 + self.diversity_weight * dW2_div)
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

            pred = np.argmax(out_list[i], axis=1)
            expert_acc[i] = np.mean(pred == y)

        # EMA tracking
        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        # Reward: increase routing to better experts
        reward = np.dot(gate.T, (np.argmax(logits, axis=1) == y).astype(float)) / B
        gate += self.lr * reward[np.newaxis, :]

        # Routing Diversity regularization
        p_mean = np.mean(gate, axis=0)
        routing_diversity_loss = -self.diversity_weight * np.sum(p_mean * np.log(p_mean + 1e-9))

        # Weight Diversity Loss between experts
        weight_diversity_loss = 0
        for i in range(self.n_experts):
            for j in range(i+1, self.n_experts):
                min_h = min(self.W2[i].shape[0], self.W2[j].shape[0])
                # Normalize weights to make similarity scale-invariant
                wi_norm = self.W2[i][:min_h] / (np.linalg.norm(self.W2[i][:min_h]) + 1e-9)
                wj_norm = self.W2[j][:min_h] / (np.linalg.norm(self.W2[j][:min_h]) + 1e-9)
                sim = np.sum(wi_norm * wj_norm)
                weight_diversity_loss += sim

        # FLOPs penalty
        flop_est = sum(X.shape[0] * X.shape[1] * h for h in self.hidden_sizes)
        flop_loss = self.flop_weight * flop_est

        total_loss = self.loss(logits, y) + routing_diversity_loss + self.diversity_weight * weight_diversity_loss + flop_loss
        return total_loss, expert_acc

# =========================
# Dataset helpers
# =========================
def make_xor(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    return X, y

def make_gaussian(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n//2, 2)
    X1 = rng.randn(n//2, 2) + 2
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def make_spiral(n=500, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n_classes = 2
    X = []
    y = []
    for j in range(n_classes):
        r = np.linspace(0.0, 1, n//n_classes)
        t = np.linspace(j*3.14, (j+1)*3.14, n//n_classes) + rng.randn(n//n_classes)*noise
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X.append(np.vstack([x1, x2]).T)
        y.append([j]*(n//n_classes))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def make_mnist_like(n=500, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y

# =========================
# Métricas
# =========================
def accuracy(model, X, y):
    logits, _, _, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def entropy(gate):
    p = np.mean(gate, axis=0) + 1e-9
    return -np.sum(p * np.log(p))

def collapse(gate):
    return np.max(np.mean(gate, axis=0))

def mutual_info(gate):
    p = np.mean(gate, axis=0) + 1e-9
    H = -np.sum(p * np.log(p))
    return H / np.log(len(p))

# =========================
# Experiment runner
# =========================
def run():
    datasets = {
        "xor": make_xor,
        "gaussian": make_gaussian,
        "spiral": make_spiral,
        "mnist_like": make_mnist_like
    }

    results = {}
    hidden_sizes = [8, 16, 32, 64, 128]

    for name, fn in datasets.items():
        print(f"\n===== DATASET: {name} =====")
        X, y = fn()
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        model = V56_MoE(input_dim=X.shape[1],
                        hidden_sizes=hidden_sizes,
                        output_dim=len(np.unique(y)),
                        top_k=2,
                        lr=0.01,
                        diversity_weight=0.01,
                        flop_weight=1e-6)

        # Training loop
        for epoch in range(3):
            idx = np.random.permutation(len(X))
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                if len(batch) == 0:
                    continue
                loss_val, expert_acc = model.train_step(X[batch], y[batch])

        # Avaliação
        acc = accuracy(model, X, y)
        _, gate, _, _ = model.forward(X)
        ent = entropy(gate)
        coll = collapse(gate)
        mi = mutual_info(gate)
        usage = np.mean(gate, axis=0)
        flop = sum(X.shape[0] * X.shape[1] * h for h in hidden_sizes)
        score = acc / (1 + flop/1e6)

        print(f"V5.6 MOE ACC: {acc:.4f}")
        print(f"Entropy: {ent:.4f}")
        print(f"Collapse: {coll:.4f}")
        print(f"MI: {mi:.4f}")
        print(f"Usage: {np.round(usage,3)}")
        print(f"FLOPs (est.): {flop}")
        print(f"Score: {score:.4f}")
        print(f"Expert Perf: {model.expert_perf}")

        results[name] = {
            "acc": float(acc),
            "entropy": float(ent),
            "collapse": float(coll),
            "MI": float(mi),
            "usage": usage.tolist(),
            "FLOPs": int(flop),
            "score": float(score),
            "expert_perf": model.expert_perf.tolist()
        }

    # Salvar resultados
    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_6_architectural_specialization.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_6_architectural_specialization.json")

if __name__ == "__main__":
    run()
