import numpy as np
import json
import os

# =========================================================
# DATASET (mantém simples pra ver efeito da arquitetura)
# =========================================================

def xor_data(n=2000):
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y


def spiral(n=2000):
    n = n // 2
    t = np.sqrt(np.random.rand(n)) * 3 * np.pi
    r = t

    X1 = np.c_[r*np.cos(t), r*np.sin(t)]
    X2 = np.c_[r*np.cos(t+np.pi), r*np.sin(t+np.pi)]

    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n)
    return X, y


# =========================================================
# V5 MOE — SPECIALIZATION FORCE
# =========================================================

class V5MoE:
    def __init__(self, d, h, c, n_experts=2, lr=0.01):
        self.n = n_experts
        self.lr = lr

        self.W1 = [np.random.randn(d, h)*0.1 for _ in range(self.n)]
        self.W2 = [np.random.randn(h, c)*0.1 for _ in range(self.n)]
        self.b1 = [np.zeros(h) for _ in range(self.n)]
        self.b2 = [np.zeros(c) for _ in range(self.n)]

        self.gW = np.random.randn(d, self.n)*0.1
        self.gb = np.zeros(self.n)

        self.temp = 2.0  # annealing control

    # -------------------------
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # -------------------------
    def forward(self, x):
        gate_logits = x @ self.gW + self.gb
        gate = self.softmax(gate_logits / self.temp)

        top = np.argmax(gate, axis=1)

        out = np.zeros((len(x), self.W2[0].shape[1]))

        for i in range(len(x)):
            j = top[i]
            h = self.relu(x[i] @ self.W1[j] + self.b1[j])
            out[i] = h @ self.W2[j] + self.b2[j]

        return out, gate, top

    # -------------------------
    def train_step(self, x, y):
        logits, gate, top = self.forward(x)

        probs = self.softmax(logits)
        B = len(y)

        probs[np.arange(B), y] -= 1
        probs /= B

        # update experts
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

        # -------------------------
        # 🔥 ROUTING UPDATE
        # -------------------------
        ggrad = np.zeros_like(gate)
        ggrad[np.arange(B), top] = 1

        self.gW -= self.lr * x.T @ ggrad
        self.gb -= self.lr * np.sum(ggrad, axis=0)

        # -------------------------
        # 🔥 SPECIALIZATION FORCE
        # -------------------------

        # 1. entropy collapse penalty
        p = np.mean(gate, axis=0)
        entropy = -np.sum(p * np.log(p + 1e-9))
        collapse_grad = (1 - entropy) * 0.01
        self.gW += collapse_grad * self.gW

        # 2. temperature annealing
        self.temp *= 0.999
        self.temp = max(self.temp, 0.7)

    # -------------------------
    def predict(self, x):
        out, _, _ = self.forward(x)
        return np.argmax(out, axis=1)


# =========================================================
# METRICS
# =========================================================

def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)


# =========================================================
# RUN EXPERIMENT
# =========================================================

def run():
    datasets = {
        "xor": xor_data,
        "spiral": spiral
    }

    results = {}

    for name, fn in datasets.items():
        print(f"\n===== {name.upper()} =====")

        X, y = fn()

        mlp = None  # opcional comparativo futuro
        moe = V5MoE(X.shape[1], 32, len(np.unique(y)))

        for _ in range(15):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                b = idx[i:i+64]
                moe.train_step(X[b], y[b])

        acc = accuracy(moe, X, y)

        print("V5 MOE ACC:", acc)

        results[name] = {"moe_acc": float(acc)}

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v5_moe.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_moe.json")


if __name__ == "__main__":
    run()