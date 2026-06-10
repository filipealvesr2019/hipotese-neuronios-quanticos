import numpy as np
import json

# =====================================================
# Utils
# =====================================================

def set_seed(seed):
    np.random.seed(seed)

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std

def train_test_split(X, y, test_size=0.25):
    n = len(X)
    idx = np.random.permutation(n)

    split = int(n * (1 - test_size))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx]
    )

def make_circles(n_samples=2000, noise=0.1, seed=42):
    rng = np.random.RandomState(seed)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.rand(n_outer) * 2 * np.pi
    theta_inner = rng.rand(n_inner) * 2 * np.pi

    outer_x = np.c_[np.cos(theta_outer), np.sin(theta_outer)]
    inner_x = 0.5 * np.c_[np.cos(theta_inner), np.sin(theta_inner)]

    X = np.vstack([outer_x, inner_x])
    y = np.hstack([
        np.zeros(n_outer, dtype=np.int64),
        np.ones(n_inner, dtype=np.int64)
    ])

    X += rng.randn(*X.shape) * noise

    return X, y

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

# =====================================================
# MLP Baseline
# =====================================================

class MLPTradicional:

    def __init__(self, input_dim, hidden, output_dim, seed=42):

        rng = np.random.RandomState(seed)

        self.W1 = rng.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)

        self.W2 = rng.randn(hidden, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def forward(self, X):

        self.X = X

        self.h = np.maximum(
            0,
            X @ self.W1 + self.b1
        )

        self.logits = self.h @ self.W2 + self.b2

        return self.logits

    def backward(self, grad, lr=0.01):

        grad_W2 = self.h.T @ grad
        grad_b2 = grad.sum(axis=0)

        grad_h = grad @ self.W2.T
        grad_h[self.h <= 0] = 0

        grad_W1 = self.X.T @ grad_h
        grad_b1 = grad_h.sum(axis=0)

        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# =====================================================
# V4.2 Trainable
# =====================================================

class MultiStateLayerV42T:

    def __init__(
        self,
        input_dim,
        output_dim,
        n_states=4,
        seed=42,
        temperature=1.5,
        gate_hidden=8,
        skip_scale=0.2
    ):

        self.n_states = n_states
        self.temperature = temperature
        self.skip_scale = skip_scale

        rng = np.random.RandomState(seed)

        self.W = [
            rng.randn(input_dim, output_dim) * 0.1
            for _ in range(n_states)
        ]

        self.b = [
            np.zeros(output_dim)
            for _ in range(n_states)
        ]

        self.gate_W1 = rng.randn(input_dim, gate_hidden) * 0.1
        self.gate_b1 = np.zeros(gate_hidden)

        self.gate_W2 = rng.randn(gate_hidden, n_states) * 0.1
        self.gate_b2 = np.zeros(n_states)

    def gate(self, X):

        h = np.maximum(
            0,
            X @ self.gate_W1 + self.gate_b1
        )

        logits = h @ self.gate_W2 + self.gate_b2

        logits /= self.temperature

        probs = softmax(logits)

        return probs

    def forward(self, X):

        probs = self.gate(X)

        states = []

        for w, b in zip(self.W, self.b):
            states.append(X @ w + b)

        out = np.zeros_like(states[0])

        for i in range(self.n_states):
            out += probs[:, i:i+1] * states[i]

        self.last_probs = probs
        self.last_X = X

        return out

# =====================================================
# Experimento
# =====================================================

def run_v42_trainable():

    results = []

    for seed in range(10):

        print(f"\n=== Seed {seed} ===")

        set_seed(seed)

        X, y = make_circles(
            n_samples=2000,
            noise=0.1,
            seed=seed
        )

        X = normalize(X)

        Xtr, Xva, ytr, yva = train_test_split(X, y)

        model = MultiStateLayerV42T(
            input_dim=2,
            output_dim=2,
            n_states=4,
            seed=seed
        )

        lr = 0.01

        for epoch in range(50):

            idx = np.random.permutation(len(Xtr))

            for start in range(0, len(Xtr), 64):

                ids = idx[start:start+64]

                Xb = Xtr[ids]
                yb = ytr[ids]

                logits = model.forward(Xb)

                probs = softmax(logits)

                grad = probs.copy()
                grad[np.arange(len(yb)), yb] -= 1
                grad /= len(yb)

                for i in range(model.n_states):

                    g = model.last_probs[:, i:i+1]

                    grad_W = Xb.T @ (grad * g)
                    grad_b = (grad * g).sum(axis=0)

                    model.W[i] -= lr * grad_W
                    model.b[i] -= lr * grad_b

        preds = np.argmax(
            model.forward(Xva),
            axis=1
        )

        acc = np.mean(preds == yva)

        entropy = (
            -np.sum(
                np.mean(model.last_probs, axis=0)
                *
                np.log2(
                    np.mean(model.last_probs, axis=0)
                    + 1e-10
                )
            )
        )

        print(
            f"Acc={acc:.4f} | Entropy={entropy:.4f}"
        )

        results.append({
            "seed": int(seed),
            "acc": float(acc),
            "entropy": float(entropy)
        })

    with open(
        "resultados_finais_v42_trainable.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\nResultados salvos!")

# =====================================================

if __name__ == "__main__":
    run_v42_trainable()