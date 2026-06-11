import numpy as np
import os
import json

# =========================
# V5.7 — Gate Alignment + Expert Memory + Load Balancing
# =========================
class V57_MoE:
    def __init__(self, input_dim, hidden_sizes, output_dim,
                 top_k=2, lr=0.01, ema=0.9, diversity_weight=0.01, balance_weight=0.1, flop_weight=1e-6):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.n_experts = len(hidden_sizes)
        self.top_k = top_k
        self.lr = lr
        self.ema = ema
        self.diversity_weight = diversity_weight
        self.balance_weight = balance_weight
        self.flop_weight = flop_weight

        rng = np.random.RandomState(42)

        # Experts weights
        self.W1 = [rng.randn(input_dim, h) * 0.1 for h in hidden_sizes]
        self.b1 = [np.zeros(h) for h in hidden_sizes]
        self.W2 = [rng.randn(h, output_dim) * 0.1 for h in hidden_sizes]
        self.b2 = [np.zeros(output_dim) for _ in range(self.n_experts)]

        # Gate weights (Agora vamos treinar de verdade!)
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, self.n_experts) * 0.1
        self.gb2 = np.zeros(self.n_experts)

        # Expert performance tracking
        self.expert_perf = np.zeros(self.n_experts)
        
        # Expert routing memory (Prior: starts with 1 to avoid div zero)
        self.expert_memory = np.ones((self.n_experts, output_dim))

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
        return out, gate, out_list, h_list, g, gate_probs

    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    def train_step(self, X, y):
        B = X.shape[0]
        logits, gate, out_list, h_list, g, gate_probs = self.forward(X)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        expert_acc = np.zeros(self.n_experts)
        expert_correct = np.zeros((B, self.n_experts))

        # 1. Backprop Experts
        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            # Weight diversity gradient (to keep experts different)
            dW2_div = np.zeros_like(self.W2[i])
            for j in range(self.n_experts):
                if i != j:
                    min_h = min(self.W2[i].shape[0], self.W2[j].shape[0])
                    dW2_div[:min_h] += self.W2[j][:min_h] / self.W2[j][:min_h].size

            self.W2[i] -= self.lr * (dW2 + self.diversity_weight * dW2_div)
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

            pred = np.argmax(out_list[i], axis=1)
            expert_correct[:, i] = (pred == y).astype(float)
            expert_acc[i] = np.mean(expert_correct[:, i])
            
            # Update Expert Routing Memory
            for c in range(self.output_dim):
                self.expert_memory[i, c] += np.sum(expert_correct[:, i] * (y == c))

        # EMA tracking
        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        # 2. Gate Alignment Backprop
        # target_probs = blend of current batch success and historical success prior
        sum_correct = np.sum(expert_correct, axis=1, keepdims=True)
        local_target = np.where(sum_correct > 0, expert_correct / sum_correct, 1.0 / self.n_experts)
        
        memory_prior = np.zeros((B, self.n_experts))
        for b in range(B):
            memory_prior[b] = self.expert_memory[:, y[b]]
        memory_prior /= np.sum(memory_prior, axis=1, keepdims=True)
        
        target_probs = 0.5 * local_target + 0.5 * memory_prior
        
        # Load balancing penalty
        usage = np.mean(gate_probs, axis=0)
        load_balance_grad = (usage - 1.0 / self.n_experts)
        
        # Gate Gradients
        d_logits = (gate_probs - target_probs) / B + self.balance_weight * load_balance_grad
        
        d_gW2 = g.T @ d_logits
        d_gb2 = np.sum(d_logits, axis=0)
        d_g = d_logits @ self.gW2.T
        d_g[g <= 0] = 0
        d_gW1 = X.T @ d_g
        d_gb1 = np.sum(d_g, axis=0)
        
        self.gW2 -= self.lr * d_gW2
        self.gb2 -= self.lr * d_gb2
        self.gW1 -= self.lr * d_gW1
        self.gb1 -= self.lr * d_gb1

        # FLOPs penalty
        flop_est = sum(X.shape[0] * X.shape[1] * h for h in self.hidden_sizes)
        flop_loss = self.flop_weight * flop_est

        total_loss = self.loss(logits, y) + flop_loss
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
    logits, _, _, _, _, _ = model.forward(X)
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

        model = V57_MoE(input_dim=X.shape[1],
                        hidden_sizes=hidden_sizes,
                        output_dim=len(np.unique(y)),
                        top_k=2,
                        lr=0.01,
                        diversity_weight=0.01,
                        balance_weight=0.1,
                        flop_weight=1e-6)

        # Training loop
        for epoch in range(10): # Let's give it 10 epochs to learn alignment properly
            idx = np.random.permutation(len(X))
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                if len(batch) == 0:
                    continue
                loss_val, expert_acc = model.train_step(X[batch], y[batch])

        # Avaliação
        acc = accuracy(model, X, y)
        _, gate, _, _, _, gate_probs = model.forward(X)
        ent = entropy(gate_probs) # Using raw probs to see gate entropy better
        coll = collapse(gate)
        mi = mutual_info(gate)
        usage = np.mean(gate, axis=0)
        flop = sum(X.shape[0] * X.shape[1] * h for h in hidden_sizes)
        score = acc / (1 + flop/1e6)

        print(f"V5.7 MOE ACC: {acc:.4f}")
        print(f"Entropy: {ent:.4f}")
        print(f"Collapse: {coll:.4f}")
        print(f"MI: {mi:.4f}")
        print(f"Usage: {np.round(usage,3)}")
        print(f"FLOPs (est.): {flop}")
        print(f"Score: {score:.4f}")
        print(f"Expert Perf: {np.round(model.expert_perf, 4)}")

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
    with open("resultados_finais/v5_7_gate_alignment.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_7_gate_alignment.json")

if __name__ == "__main__":
    run()
