import numpy as np
import json
import os

# =========================================================
# V5.2 — DYNAMIC PRUNING + GROWTH MOE (FIXED)
# =========================================================

class V52DynamicMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 max_experts=5, min_experts=2,
                 lr=0.01, ema=0.9, prune_threshold=0.45,
                 seed=42):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim

        self.max_experts = max_experts
        self.min_experts = min_experts
        self.lr = lr
        self.ema = ema
        self.prune_threshold = prune_threshold

        self.rng = np.random.RandomState(seed)

        # Experts
        self.W1, self.b1 = [], []
        self.W2, self.b2 = [], []

        for _ in range(min_experts):
            self.add_expert()

        # Gate
        self.gW1 = self.rng.randn(input_dim, 32) * 0.1
        self.gb1 = np.zeros(32)
        self.gW2 = self.rng.randn(32, max_experts) * 0.1
        self.gb2 = np.zeros(max_experts)

        self.expert_perf = np.zeros(max_experts)
        self.active_mask = np.zeros(max_experts)

        self.active_mask[:min_experts] = 1

    # =========================
    def add_expert(self):
        self.W1.append(self.rng.randn(self.input_dim, self.hidden) * 0.1)
        self.b1.append(np.zeros(self.hidden))
        self.W2.append(self.rng.randn(self.hidden, self.output_dim) * 0.1)
        self.b2.append(np.zeros(self.output_dim))

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

        for i in range(len(self.W1)):
            if self.active_mask[i] == 0:
                hs.append(np.zeros((x.shape[0], self.hidden)))
                outs.append(np.zeros((x.shape[0], self.output_dim)))
                continue

            h = self.relu(x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]

            hs.append(h)
            outs.append(out)

        return hs, outs

    # =========================
    def gate(self, x):
        g = self.relu(x @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2

        logits = logits + 0.5 * self.expert_perf.reshape(1, -1)

        probs = self.softmax(logits)

        probs = probs * self.active_mask

        probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-9)

        return probs

    # =========================
    def forward(self, x):
        hs, outs = self.forward_experts(x)
        probs = self.gate(x)

        final = np.zeros((x.shape[0], self.output_dim))

        for i in range(len(self.W1)):
            final += probs[:, i:i+1] * outs[i]

        return final, probs, hs, outs

    # =========================
    def train_step(self, x, y):
        B = x.shape[0]

        hs, outs = self.forward_experts(x)
        probs = self.gate(x)

        logits = np.zeros((B, self.output_dim))

        for i in range(len(self.W1)):
            logits += probs[:, i:i+1] * outs[i]

        # softmax grad
        p = self.softmax(logits)
        p[np.arange(B), y] -= 1
        p /= B

        expert_acc = np.zeros(self.max_experts)

        # =========================
        # UPDATE EXPERTS
        # =========================
        for i in range(len(self.W1)):

            if self.active_mask[i] == 0:
                continue

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

            # =========================
            # FIX: correct expert accuracy
            # =========================
            pred_i = outs[i]
            pred_labels = np.argmax(pred_i, axis=1)

            expert_acc[i] = np.mean(pred_labels == y)

        # =========================
        # FIX: EMA SAFE UPDATE
        # =========================
        full_acc = np.zeros(self.max_experts)

        for i in range(self.max_experts):
            if self.active_mask[i] == 1:
                full_acc[i] = expert_acc[i]

        self.expert_perf = (
            self.ema * self.expert_perf +
            (1 - self.ema) * full_acc
        )

        self.prune_and_grow()

        return -np.mean(np.log(p[np.arange(B), y] + 1e-9))

    # =========================
    def prune_and_grow(self):

        active_count = np.sum(self.active_mask)

        for i in range(len(self.W1)):

            if self.active_mask[i] == 0:
                continue

            if self.expert_perf[i] < self.prune_threshold:

                if active_count > self.min_experts:
                    self.active_mask[i] = 0
                    active_count -= 1

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

        model = V52DynamicMoE(
            input_dim=X.shape[1],
            hidden=32,
            output_dim=2,
            max_experts=5,
            min_experts=2
        )

        for epoch in range(5):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        _, gate_probs, _, _ = model.forward(X)
        ent = entropy(np.mean(gate_probs, axis=0))

        print(f"V5.2 ACC: {acc:.4f}")
        print(f"Entropy: {ent:.4f}")
        print(f"Expert Perf: {model.expert_perf}")

        results.append({
            "dataset": name,
            "acc": float(acc),
            "entropy": float(ent),
            "expert_perf": model.expert_perf.tolist()
        })

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v5_2_dynamic_pruning.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_2_dynamic_pruning.json")


if __name__ == "__main__":
    run()