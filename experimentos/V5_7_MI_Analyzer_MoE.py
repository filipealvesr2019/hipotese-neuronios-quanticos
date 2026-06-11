import numpy as np
import json
import os

# =====================================================
# DATASETS
# =====================================================

def make_xor(n=2000, seed=0):
    rng = np.random.RandomState(seed)

    X = rng.randn(n, 2)

    y = (
        (X[:, 0] > 0).astype(int) ^
        (X[:, 1] > 0).astype(int)
    )

    return X, y


def make_gaussian(n=2000, seed=0):

    rng = np.random.RandomState(seed)

    X0 = rng.randn(n // 2, 2) + np.array([-2, -2])
    X1 = rng.randn(n // 2, 2) + np.array([2, 2])

    X = np.vstack([X0, X1])

    y = np.concatenate([
        np.zeros(n // 2),
        np.ones(n // 2)
    ]).astype(int)

    return X, y


def make_spiral(n=2000, seed=0):

    rng = np.random.RandomState(seed)

    n2 = n // 2

    theta = np.sqrt(rng.rand(n2)) * 4 * np.pi

    r = 2 * theta + np.pi

    X0 = np.c_[
        np.cos(theta) * r,
        np.sin(theta) * r
    ]

    X1 = np.c_[
        np.cos(theta + np.pi) * r,
        np.sin(theta + np.pi) * r
    ]

    X = np.vstack([X0, X1])

    y = np.concatenate([
        np.zeros(n2),
        np.ones(n2)
    ]).astype(int)

    X += rng.randn(*X.shape) * 0.2

    return X, y


def make_mnist_like(n=4000, seed=0):

    rng = np.random.RandomState(seed)

    X = rng.randn(n, 784)

    y = rng.randint(0, 10, n)

    return X, y


# =====================================================
# MODEL
# =====================================================

class V57MoE:

    def __init__(
        self,
        input_dim,
        hidden,
        output_dim,
        n_experts=5,
        lr=0.01,
        seed=0
    ):

        self.n_experts = n_experts
        self.lr = lr

        rng = np.random.RandomState(seed)

        self.W1 = [
            rng.randn(input_dim, hidden) * 0.05
            for _ in range(n_experts)
        ]

        self.b1 = [
            np.zeros(hidden)
            for _ in range(n_experts)
        ]

        self.W2 = [
            rng.randn(hidden, output_dim) * 0.05
            for _ in range(n_experts)
        ]

        self.b2 = [
            np.zeros(output_dim)
            for _ in range(n_experts)
        ]

        self.gW = rng.randn(
            input_dim,
            n_experts
        ) * 0.05

        self.gb = np.zeros(n_experts)

    # =====================================

    def softmax(self, x):

        x = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        e = np.exp(x)

        return e / np.sum(
            e,
            axis=1,
            keepdims=True
        )

    # =====================================

    def forward(self, X):

        gate_logits = X @ self.gW + self.gb

        gate_probs = self.softmax(
            gate_logits
        )

        expert_outputs = []

        for i in range(self.n_experts):

            h = np.maximum(
                0,
                X @ self.W1[i] + self.b1[i]
            )

            out = (
                h @ self.W2[i]
                + self.b2[i]
            )

            expert_outputs.append(out)

        expert_outputs = np.stack(
            expert_outputs,
            axis=1
        )

        mixture = np.sum(
            gate_probs[:, :, None]
            * expert_outputs,
            axis=1
        )

        return (
            mixture,
            gate_probs,
            expert_outputs
        )

    # =====================================

    def train_step(self, X, y):

        logits, gate, _ = self.forward(X)

        p = self.softmax(logits)

        B = len(X)

        grad = p.copy()

        grad[np.arange(B), y] -= 1

        grad /= B

        for e in range(self.n_experts):

            h = np.maximum(
                0,
                X @ self.W1[e] + self.b1[e]
            )

            grad_out = (
                grad
                * gate[:, e:e+1]
            )

            dW2 = h.T @ grad_out

            db2 = np.sum(
                grad_out,
                axis=0
            )

            dh = (
                grad_out
                @ self.W2[e].T
            )

            dh[h <= 0] = 0

            dW1 = X.T @ dh

            db1 = np.sum(
                dh,
                axis=0
            )

            self.W2[e] -= self.lr * dW2
            self.b2[e] -= self.lr * db2

            self.W1[e] -= self.lr * dW1
            self.b1[e] -= self.lr * db1

        return np.mean(
            np.argmax(p, axis=1) == y
        )


# =====================================================
# METRICS
# =====================================================

def entropy(v):

    v = v + 1e-12

    return -np.sum(
        v * np.log(v)
    )


def mutual_information(y, experts):

    classes = np.unique(y)

    mi = 0.0

    N = len(y)

    for c in classes:

        for e in range(
            np.max(experts) + 1
        ):

            pce = np.mean(
                (y == c)
                & (experts == e)
            )

            if pce <= 0:
                continue

            pc = np.mean(y == c)

            pe = np.mean(experts == e)

            mi += pce * np.log(
                pce / (pc * pe)
            )

    return float(mi)


# =====================================================
# BENCH
# =====================================================

def evaluate_dataset(
    name,
    X,
    y,
    output_dim
):

    print(
        f"\n===== DATASET: {name} ====="
    )

    model = V57MoE(
        input_dim=X.shape[1],
        hidden=32,
        output_dim=output_dim,
        n_experts=5
    )

    for epoch in range(10):

        idx = np.random.permutation(
            len(X)
        )

        for i in range(
            0,
            len(X),
            64
        ):

            batch = idx[i:i+64]

            model.train_step(
                X[batch],
                y[batch]
            )

    logits, gate, _ = model.forward(X)

    pred = np.argmax(
        logits,
        axis=1
    )

    acc = np.mean(
        pred == y
    )

    chosen = np.argmax(
        gate,
        axis=1
    )

    usage = np.bincount(
        chosen,
        minlength=5
    )

    usage = usage / usage.sum()

    ent = entropy(usage)

    collapse = 1.0 - (
        ent / np.log(5)
    )

    mi = mutual_information(
        y,
        chosen
    )

    print(
        f"ACC: {acc:.4f}"
    )

    print(
        f"Entropy: {ent:.4f}"
    )

    print(
        f"Collapse: {collapse:.4f}"
    )

    print(
        f"MI: {mi:.4f}"
    )

    print(
        f"Usage: {usage}"
    )

    return {
        "acc": float(acc),
        "entropy": float(ent),
        "collapse": float(collapse),
        "mi": float(mi),
        "usage": usage.tolist()
    }


# =====================================================
# RUN
# =====================================================

def run():

    results = {}

    X, y = make_xor()
    results["xor"] = evaluate_dataset(
        "xor",
        X,
        y,
        2
    )

    X, y = make_gaussian()
    results["gaussian"] = evaluate_dataset(
        "gaussian",
        X,
        y,
        2
    )

    X, y = make_spiral()
    results["spiral"] = evaluate_dataset(
        "spiral",
        X,
        y,
        2
    )

    X, y = make_mnist_like()
    results["mnist_like"] = evaluate_dataset(
        "mnist_like",
        X,
        y,
        10
    )

    os.makedirs(
        "resultados_finais",
        exist_ok=True
    )

    with open(
        "resultados_finais/v5_7_mi_analyzer.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        "\nSaved -> resultados_finais/v5_7_mi_analyzer.json"
    )


if __name__ == "__main__":
    run()