import numpy as np
import json
import os

# =========================================================
# V5.1 — ERROR-AWARE ROUTING MOE
# =========================================================

class V51ErrorAwareMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=3, lr=0.01, ema=0.9, seed=42):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.lr = lr
        self.ema = ema

        rng = np.random.RandomState(seed)

        # Experts
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # Gate (input + expert score)
        self.gW1 = rng.randn(input_dim, 32) * 0.1
        self.gb1 = np.zeros(32)

        self.gW2 = rng.randn(32, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

        # EMA performance tracker
        self.expert_perf = np.zeros(n_experts)

    # =========================
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0, x)

    # =========================
    def forward_experts(self, x):
        hs, outs = [], []

        for i in range(self.n_experts):
            h = self.relu(x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            hs.append(h)
            outs.append(out)

        return hs, outs

    # =========================
    def gate(self, x):
        g = self.relu(x @ self.gW1 + self.gb1)

        logits = g @ self.gW2 + self.gb2

        # inject expert performance bias
        perf_bias = self.expert_perf.reshape(1, -1)

        logits = logits + 0.5 * perf_bias

        return self.softmax(logits)

    # =========================
    def forward(self, x):
        hs, outs = self.forward_experts(x)
        probs = self.gate(x)

        final = sum(probs[:, i:i+1] * outs[i] for i in range(self.n_experts))
        return final, probs, hs, outs

    # =========================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    # =========================
    def train_step(self, x, y):
        B = x.shape[0]

        hs, outs = self.forward_experts(x)
        probs = self.gate(x)

        logits = sum(probs[:, i:i+1] * outs[i] for i in range(self.n_experts))

        # softmax grad
        p = self.softmax(logits)
        p[np.arange(B), y] -= 1
        p /= B

        expert_acc = np.zeros(self.n_experts)

        # update experts
        for i in range(self.n_experts):

            gi = probs[:, i:i+1]

            grad = p * gi

            dh = grad @ self.W2[i].T
            dh[hs[i] <= 0] = 0

            dW2 = hs[i].T @ grad
            db2 = np.sum(grad, axis=0)

            dW1 = x.T @ dh
            db1 = np.sum(dh, axis=0)

            self.W1[i] -= self.lr * dW1
            self.W2[i] -= self.lr * dW2
            self.b1[i] -= self.lr * db1
            self.b2[i] -= self.lr * db2

            # accuracy proxy per expert
            pred = np.argmax(outs[i], axis=1)
            expert_acc[i] = np.mean(pred == y)

        # EMA update
        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        return self.loss(logits, y)

# =========================================================
# DATASETS
# =========================================================

def xor(n=1000):
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

def gaussian(n=1000):
    X = np.random.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def spiral(n=1000):
    X = np.zeros((n, 2))
    y = np.zeros(n)

    for i in range(n):
        r = i / n
        t = 4 * np.pi * r

        X[i, 0] = r * np.cos(t)
        X[i, 1] = r * np.sin(t)

        y[i] = int(r > 0.5)

    return X, y.astype(int)

# =========================================================
# METRICS
# =========================================================

def accuracy(model, X, y):
    logits, _, _, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p)) / len(p)

# =========================================================
# RUN
# =========================================================

def run():
    results = []

    datasets = {
        "xor": xor,
        "gaussian": gaussian,
        "spiral": spiral
    }

    for name, fn in datasets.items():
        print(f"\n===== {name.upper()} =====")

        X, y = fn()

        mlp_acc = None

        model = V51ErrorAwareMoE(
            input_dim=X.shape[1],
            hidden=32,
            output_dim=2,
            n_experts=3,
            lr=0.01
        )

        for epoch in range(5):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        _, gate_probs, _, _ = model.forward(X)

        ent = entropy(np.mean(gate_probs, axis=0))

        print(f"V5.1 MOE ACC: {acc:.4f}")
        print(f"Entropy: {ent:.4f}")
        print(f"Expert Perf: {model.expert_perf}")

        results.append({
            "dataset": name,
            "acc": float(acc),
            "entropy": float(ent),
            "expert_perf": model.expert_perf.tolist()
        })

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v5_1_error_aware.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_1_error_aware.json")


if __name__ == "__main__":
    run()