import numpy as np
import json
import os

# =========================================================
# DATASETS CONTROLADOS
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


def mnist_like(n=3000, d=20, classes=10):
    X = np.random.randn(n, d)
    W = np.random.randn(d, classes)
    logits = X @ W
    y = np.argmax(logits, axis=1)
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
        return out, h

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
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)


# =========================================================
# V4 MOE TOP-2
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

        top2 = np.argsort(gate, axis=1)[:, -2:]

        B = len(x)
        final = np.zeros((B, self.W2[0].shape[1]))

        for i in range(B):
            for j in top2[i]:
                h = self.relu(x[i] @ self.W1[j] + self.b1[j])
                out = h @ self.W2[j] + self.b2[j]
                final[i] += gate[i, j] * out

        return final, gate

    def train_step(self, x, y):
        gate_logits = x @ self.gW + self.gb
        gate = self.softmax(gate_logits)

        top2 = np.argsort(gate, axis=1)[:, -2:]

        B = len(x)
        logits = np.zeros((B, self.W2[0].shape[1]))

        caches = []

        for i in range(B):
            sample_outs = []
            for j in top2[i]:
                h = self.relu(x[i] @ self.W1[j] + self.b1[j])
                out = h @ self.W2[j] + self.b2[j]
                sample_outs.append((j, h, out))
                logits[i] += gate[i, j] * out
            caches.append(sample_outs)

        probs = self.softmax(logits)

        probs[np.arange(B), y] -= 1
        probs /= B

        # update experts
        for i in range(B):
            for j, h, out in caches[i]:
                grad = probs[i] * gate[i, j]

                dW2 = np.outer(h, grad)
                db2 = grad

                dh = self.W2[j] @ grad
                dh[h <= 0] = 0

                dW1 = np.outer(x[i], dh)
                db1 = dh

                self.W1[j] -= self.lr * dW1
                self.W2[j] -= self.lr * dW2
                self.b1[j] -= self.lr * db1
                self.b2[j] -= self.lr * db2

        # gate update (simples)
        ggrad = np.zeros_like(gate)
        ggrad[np.arange(B), top2[:, 0]] += 1
        ggrad[np.arange(B), top2[:, 1]] += 1

        self.gW -= self.lr * x.T @ ggrad
        self.gb -= self.lr * np.sum(ggrad, axis=0)

    def predict(self, x):
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)


# =========================================================
# METRICS
# =========================================================

def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)


def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))


# =========================================================
# RUN BENCH
# =========================================================

def run():
    datasets = {
        "xor": xor_data,
        "gaussian": gaussian_clusters,
        "spiral": spirals,
        "mnist_like": mnist_like
    }

    results = {}

    for name, fn in datasets.items():
        print(f"\n===== DATASET: {name} =====")

        X, y = fn()

        mlp = MLP(X.shape[1], 32, len(np.unique(y)))
        moe = V4MoE(X.shape[1], 32, len(np.unique(y)))

        # train
        for _ in range(10):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                b = idx[i:i+64]
                mlp.train_step(X[b], y[b])
                moe.train_step(X[b], y[b])

        mlp_acc = accuracy(mlp, X, y)
        moe_acc = accuracy(moe, X, y)

        print("MLP ACC:", mlp_acc)
        print("MOE ACC:", moe_acc)

        results[name] = {
            "mlp_acc": float(mlp_acc),
            "moe_acc": float(moe_acc),
        }

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v4_bench.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v4_bench.json")


if __name__ == "__main__":
    run()