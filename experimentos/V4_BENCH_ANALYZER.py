import numpy as np
import json
import os

# =========================================================
# DATASETS (mesmos do bench)
# =========================================================

def xor_data(n=2000):
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y


def gaussian_clusters(n=2000, centers=4, d=2):
    X = []
    y = []
    for i in range(centers):
        c = np.random.randn(d) * 3
        xi = np.random.randn(n // centers, d) + c
        yi = np.full(n // centers, i % 2)
        X.append(xi)
        y.append(yi)
    return np.vstack(X), np.hstack(y)


def spirals(n=2000):
    n = n // 2
    theta = np.sqrt(np.random.rand(n)) * 3 * np.pi
    r = theta

    X1 = np.c_[r * np.cos(theta), r * np.sin(theta)]
    X2 = np.c_[r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)]

    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n)
    return X, y


# =========================================================
# MLP BASELINE
# =========================================================

class MLP:
    def __init__(self, d, h, c, lr=0.01):
        self.W1 = np.random.randn(d, h) * 0.1
        self.W2 = np.random.randn(h, c) * 0.1
        self.b1 = np.zeros(h)
        self.b2 = np.zeros(c)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        h = self.relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out

    def train_step(self, x, y):
        h = self.relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        probs = self.softmax(logits)
        B = len(y)

        probs[np.arange(B), y] -= 1
        probs /= B

        dW2 = h.T @ probs
        db2 = np.sum(probs, axis=0)

        dh = probs @ self.W2.T
        dh[h <= 0] = 0

        dW1 = x.T @ dh
        db1 = np.sum(dh, axis=0)

        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)


# =========================================================
# V4 MOE
# =========================================================

class V4MoE:
    def __init__(self, d, h, c, n_experts=2, lr=0.01):
        self.n = n_experts
        self.lr = lr

        self.W1 = [np.random.randn(d, h) * 0.1 for _ in range(self.n)]
        self.W2 = [np.random.randn(h, c) * 0.1 for _ in range(self.n)]
        self.b1 = [np.zeros(h) for _ in range(self.n)]
        self.b2 = [np.zeros(c) for _ in range(self.n)]

        self.gW = np.random.randn(d, self.n) * 0.1
        self.gb = np.zeros(self.n)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        gate_logits = x @ self.gW + self.gb
        gate = self.softmax(gate_logits)

        top = np.argmax(gate, axis=1)

        out = np.zeros((len(x), self.W2[0].shape[1]))

        for i in range(len(x)):
            j = top[i]
            h = self.relu(x[i] @ self.W1[j] + self.b1[j])
            out[i] = h @ self.W2[j] + self.b2[j]

        return out, gate, top

    def train_step(self, x, y):
        logits, gate, top = self.forward(x)

        probs = self.softmax(logits)
        B = len(y)

        probs[np.arange(B), y] -= 1
        probs /= B

        for i in range(B):
            j = top[i]

            grad = probs[i]

            h = self.relu(x[i] @ self.W1[j] + self.b1[j])

            self.W2[j] -= self.lr * np.outer(h, grad)
            self.b2[j] -= self.lr * grad

            dh = self.W2[j] @ grad
            dh[h <= 0] = 0

            self.W1[j] -= self.lr * np.outer(x[i], dh)
            self.b1[j] -= self.lr * dh

        # gate update
        ggrad = np.zeros_like(gate)
        ggrad[np.arange(B), top] = 1

        self.gW -= self.lr * x.T @ ggrad
        self.gb -= self.lr * np.sum(ggrad, axis=0)

    def predict(self, x):
        out, _, _ = self.forward(x)
        return np.argmax(out, axis=1)


# =========================================================
# METRICS
# =========================================================

def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)


def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))


def mutual_info_like(top, n_experts):
    counts = np.bincount(top, minlength=n_experts)
    p = counts / np.sum(counts)
    return entropy(p)


def collapse_score(gate):
    mean = np.mean(gate, axis=0)
    return entropy(mean)


# =========================================================
# RUN
# =========================================================

def run():
    datasets = {
        "xor": xor_data,
        "gaussian": gaussian_clusters,
        "spiral": spirals
    }

    results = {}

    for name, fn in datasets.items():
        print(f"\n===== DATASET: {name} =====")

        X, y = fn()

        mlp = MLP(X.shape[1], 32, len(np.unique(y)))
        moe = V4MoE(X.shape[1], 32, len(np.unique(y)))

        for _ in range(10):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                b = idx[i:i+64]
                mlp.train_step(X[b], y[b])
                moe.train_step(X[b], y[b])

        mlp_acc = accuracy(mlp, X, y)
        moe_acc = accuracy(moe, X, y)

        _, gate, top = moe.forward(X)

        mi = mutual_info_like(top, moe.n)
        collapse = collapse_score(gate)

        print("MLP:", mlp_acc)
        print("MOE:", moe_acc)
        print("MI:", mi)
        print("Collapse:", collapse)

        results[name] = {
            "mlp_acc": float(mlp_acc),
            "moe_acc": float(moe_acc),
            "mi": float(mi),
            "collapse": float(collapse)
        }

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v4_bench_analyzer.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v4_bench_analyzer.json")


if __name__ == "__main__":
    run()