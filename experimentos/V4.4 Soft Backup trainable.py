import numpy as np
import json
import os

# =========================
# V4.4 Soft Backup — TRAINABLE VERSION
# =========================
class V44SoftBackupTrainable:
    def __init__(self, input_dim, hidden, output_dim,
                 n_states=2, seed=42, lr=0.01,
                 temperature=1.5, top1_ratio=0.9):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_states = n_states
        self.lr = lr
        self.temperature = temperature
        self.top1_ratio = top1_ratio

        rng = np.random.RandomState(seed)

        # States (experts)
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_states)]
        self.b1 = [np.zeros(hidden) for _ in range(n_states)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_states)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_states) * 0.1
        self.gb2 = np.zeros(n_states)

    # =========================
    # Softmax
    # =========================
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # =========================
    # Forward
    # =========================
    def forward(self, x):
        B = x.shape[0]

        # states
        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        # gate
        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2
        probs = self.softmax(logits / self.temperature)

        # soft backup routing
        top1 = np.argmax(probs, axis=1)

        gate = np.zeros_like(probs)

        for i in range(B):
            gate[i, top1[i]] = self.top1_ratio
            for j in range(self.n_states):
                if j != top1[i]:
                    gate[i, j] += (1 - self.top1_ratio) / (self.n_states - 1)

        gate = gate / gate.sum(axis=1, keepdims=True)

        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))
        return out, probs

    # =========================
    # Loss
    # =========================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    # =========================
    # Backward (simplificado mas funcional)
    # =========================
    def train_step(self, x, y):
        B = x.shape[0]

        # forward
        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits_g = g @ self.gW2 + self.gb2
        probs = self.softmax(logits_g / self.temperature)

        top1 = np.argmax(probs, axis=1)

        gate = np.zeros_like(probs)
        for i in range(B):
            gate[i, top1[i]] = self.top1_ratio
            for j in range(self.n_states):
                if j != top1[i]:
                    gate[i, j] += (1 - self.top1_ratio) / (self.n_states - 1)

        gate = gate / gate.sum(axis=1, keepdims=True)

        logits = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))

        # gradient output
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        # update experts
        for i in range(self.n_states):

            grad_out = probs_out * gate[:, i:i+1]

            dh = grad_out @ self.W2[i].T
            dh[h_states[i] <= 0] = 0

            dW2 = h_states[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)

            dW1 = x.T @ dh
            db1 = np.sum(dh, axis=0)

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

        return self.loss(logits, y)

# =========================
# Dataset simples (MNIST-like sintético)
# =========================
def make_data(n=2000, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y

def accuracy(model, X, y):
    logits, _ = model.forward(X)
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))

# =========================
# Experiment
# =========================
def run(n_seeds=10):
    results = []

    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")

        X, y = make_data(seed=seed)

        model = V44SoftBackupTrainable(
            input_dim=784,
            hidden=96,
            output_dim=10,
            n_states=2,
            seed=seed
        )

        # training
        for epoch in range(3):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        # entropy do gate
        _, gate_probs = model.forward(X)
        ent = entropy(np.mean(gate_probs, axis=0))

        print(f"Acc={acc:.4f} | Entropy={ent:.4f}")

        results.append({
            "seed": seed,
            "acc": float(acc),
            "entropy": float(ent)
        })

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v4_4_soft_backup_trainable.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v4_4_soft_backup_trainable.json")


if __name__ == "__main__":
    run()