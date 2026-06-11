import numpy as np
import os
import json

# =========================
# V5.3H — Reward + FLOP Efficient MoE
# =========================
class V53H_MoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, top_k=2, lr=0.01, ema=0.9):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.lr = lr
        self.ema = ema

        rng = np.random.RandomState(42)

        # Experts weights
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

        # Expert performance tracking
        self.expert_perf = np.zeros(n_experts)

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

        g = np.maximum(0, X @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2
        gate_probs = self.softmax(logits)

        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            for idx in topk_idx[b]:
                gate[b, idx] = 1.0 / self.top_k

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

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

            pred = np.argmax(out_list[i], axis=1)
            expert_acc[i] = np.mean(pred == y)

        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        return self.loss(logits, y), expert_acc

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

def accuracy(model, X, y):
    logits, _, _, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def entropy(gate):
    p = np.mean(gate, axis=0) + 1e-9
    return -np.sum(p * np.log(p))

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

    for name, fn in datasets.items():
        print(f"\n===== DATASET: {name} =====")
        X, y = fn()
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        model = V53H_MoE(input_dim=X.shape[1],
                         hidden=32,
                         output_dim=len(np.unique(y)),
                         n_experts=5,
                         top_k=2,
                         lr=0.01)

        for epoch in range(3):
            idx = np.random.permutation(len(X))
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                if len(batch) == 0:
                    continue
                loss_val, expert_acc = model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)
        _, gate, _, _ = model.forward(X)
        ent = entropy(gate)
        usage = np.mean(gate, axis=0)
        flop = X.shape[0] * X.shape[1] * model.hidden * model.n_experts  # estimativa simples
        score = acc / (1 + flop/1e6)  # balance FLOPs x acurácia

        print(f"V5.3H MOE ACC: {acc:.4f}")
        print(f"Entropy: {ent:.4f}")
        print(f"Usage: {np.round(usage,3)}")
        print(f"FLOPs (est.): {flop}")
        print(f"Score: {score:.4f}")
        print(f"Expert Perf: {model.expert_perf}")

        results[name] = {
            "acc": float(acc),
            "entropy": float(ent),
            "usage": usage.tolist(),
            "FLOPs": int(flop),
            "score": float(score),
            "expert_perf": model.expert_perf.tolist()
        }

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_3H_reward_flop_moe.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_3H_reward_flop_moe.json")

if __name__ == "__main__":
    run()