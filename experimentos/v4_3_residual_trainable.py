import numpy as np
import json
import os

# =====================================================
# Utils
# =====================================================

def set_seed(seed):
    np.random.seed(seed)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p))

# =====================================================
# Dataset
# =====================================================

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def make_synthetic_mnist(seed=0):

    X, y = make_classification(
        n_samples=5000,
        n_features=784,
        n_informative=50,
        n_redundant=0,
        n_classes=10,
        random_state=seed
    )

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )

# =====================================================
# V4.3 Residual Trainable
# =====================================================

class V43Residual:

    def __init__(
        self,
        input_dim,
        hidden=96,
        output_dim=10,
        states=2,
        seed=0
    ):

        rng = np.random.RandomState(seed)

        self.states = states
        self.hidden = hidden

        self.W1 = [
            rng.randn(input_dim, hidden) * 0.02
            for _ in range(states)
        ]

        self.b1 = [
            np.zeros(hidden)
            for _ in range(states)
        ]

        self.W2 = [
            rng.randn(hidden, output_dim) * 0.02
            for _ in range(states)
        ]

        self.b2 = [
            np.zeros(output_dim)
            for _ in range(states)
        ]

        # gate

        self.G1 = rng.randn(input_dim, 8) * 0.02
        self.gb1 = np.zeros(8)

        self.G2 = rng.randn(8, states) * 0.02
        self.gb2 = np.zeros(states)

        # residual

        self.Wskip = rng.randn(input_dim, output_dim) * 0.02
        self.bskip = np.zeros(output_dim)

    def gate(self, X):

        h = np.maximum(
            0,
            X @ self.G1 + self.gb1
        )

        logits = h @ self.G2 + self.gb2

        probs = softmax(logits)

        return probs

    def forward(self, X):

        gate_probs = self.gate(X)

        chosen = np.argmax(
            gate_probs,
            axis=1
        )

        B = X.shape[0]

        logits = np.zeros((B, 10))

        usage = np.zeros(self.states)

        hidden_cache = []

        for s in range(self.states):

            mask = chosen == s

            usage[s] = np.sum(mask)

            if np.sum(mask) == 0:
                hidden_cache.append(None)
                continue

            xs = X[mask]

            h = np.maximum(
                0,
                xs @ self.W1[s] + self.b1[s]
            )

            out = h @ self.W2[s] + self.b2[s]

            logits[mask] = out

            hidden_cache.append(h)

        logits += X @ self.Wskip + self.bskip

        return logits, chosen, usage, hidden_cache

    def train_epoch(
        self,
        X,
        y,
        lr=0.01,
        batch_size=128
    ):

        idx = np.random.permutation(len(X))

        for start in range(
            0,
            len(X),
            batch_size
        ):

            ids = idx[start:start+batch_size]

            xb = X[ids]
            yb = y[ids]

            logits, chosen, usage, hidden_cache = self.forward(xb)

            probs = softmax(logits)

            grad = probs.copy()

            grad[
                np.arange(len(yb)),
                yb
            ] -= 1

            grad /= len(yb)

            # residual

            gWskip = xb.T @ grad
            gbskip = np.sum(grad, axis=0)

            self.Wskip -= lr * gWskip
            self.bskip -= lr * gbskip

            # experts

            for s in range(self.states):

                mask = chosen == s

                if np.sum(mask) == 0:
                    continue

                xs = xb[mask]
                gs = grad[mask]

                h = hidden_cache[s]

                gW2 = h.T @ gs
                gb2 = np.sum(gs, axis=0)

                dh = gs @ self.W2[s].T
                dh[h <= 0] = 0

                gW1 = xs.T @ dh
                gb1 = np.sum(dh, axis=0)

                self.W2[s] -= lr * gW2
                self.b2[s] -= lr * gb2

                self.W1[s] -= lr * gW1
                self.b1[s] -= lr * gb1

# =====================================================
# FLOPs
# =====================================================

def estimate_flops():

    return 250000

# =====================================================
# Run
# =====================================================

def run_experiment():

    results = []

    for seed in range(10):

        print(f"\n=== Seed {seed} ===")

        set_seed(seed)

        Xtr, Xte, ytr, yte = make_synthetic_mnist(seed)

        model = V43Residual(
            input_dim=784,
            hidden=96,
            output_dim=10,
            states=2,
            seed=seed
        )

        for epoch in range(10):

            model.train_epoch(
                Xtr,
                ytr,
                lr=0.01
            )

        logits, chosen, usage, _ = model.forward(Xte)

        pred = np.argmax(
            logits,
            axis=1
        )

        acc = np.mean(pred == yte)

        usage = usage / np.sum(usage)

        ent = entropy(usage)

        flops = estimate_flops()

        acc_per_mflop = acc / (flops / 1_000_000)

        print(
            f"Acc={acc:.4f} | "
            f"Entropy={ent:.4f} | "
            f"Acc/MFLOP={acc_per_mflop:.4f}"
        )

        results.append({
            "seed": seed,
            "acc": float(acc),
            "entropy": float(ent),
            "acc_per_mflop": float(acc_per_mflop)
        })

    os.makedirs(
        "resultados_finais",
        exist_ok=True
    )

    with open(
        "resultados_finais/v4_3_residual_trainable.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\nResultados salvos!")

if __name__ == "__main__":

    print(
        "=== V4.3 Residual REAL Trainable ==="
    )

    run_experiment()