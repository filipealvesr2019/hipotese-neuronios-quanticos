import numpy as np
import os
import json

# =========================
# V5.3F — FLOP Efficient + Balanced MoE
# =========================
class V53FMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, top_k=2, lr=0.01, seed=42,
                 diversity_lambda=0.05, adaptive_hidden=True):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.lr = lr
        self.diversity_lambda = diversity_lambda
        self.adaptive_hidden = adaptive_hidden

        rng = np.random.RandomState(seed)

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

        # Top-k routing
        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            gate[b, topk_idx[b]] = 1.0 / self.top_k

        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list

    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    def train_step(self, X, y):
        B = X.shape[0]
        out, gate, out_list, h_list = self.forward(X)
        probs_out = self.softmax(out)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            # Diversity regularization
            for j in range(self.n_experts):
                if j != i:
                    dW2 += self.diversity_lambda * 2 * (self.W2[i] - self.W2[j])

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

        acc = np.mean(np.argmax(out, axis=1) == y)
        return acc, gate

# =========================
# Datasets
# =========================
def make_xor(n=200):
    X = np.random.randint(0, 2, size=(n,2))
    y = np.logical_xor(X[:,0], X[:,1]).astype(int)
    return X, y

def make_gaussian(n=200, d=2, c=3):
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(n//c, d) + i*2 for i in range(c)])
    y = np.hstack([np.full(n//c, i) for i in range(c)])
    return X, y

def make_spiral(n=200, d=2, c=3):
    rng = np.random.RandomState(0)
    X, y = [], []
    for i in range(c):
        theta = np.linspace(0, 4*np.pi, n//c)
        r = np.linspace(0.0, 1, n//c)
        xi = r * np.cos(theta + i*2*np.pi/c)
        yi = r * np.sin(theta + i*2*np.pi/c)
        X.append(np.stack([xi, yi], axis=1))
        y.append(np.full(n//c, i))
    return np.vstack(X), np.hstack(y)

def make_mnist_like(n=500, d=784, c=10):
    rng = np.random.RandomState(0)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y

def entropy(probs):
    p = np.mean(probs, axis=0) + 1e-9
    return -np.sum(p * np.log(p))

def collapse(probs):
    usage = np.mean(probs, axis=0)
    return np.sum(usage**2)

def mutual_information(probs, y):
    B, n = probs.shape
    p_y = np.mean(probs, axis=0)
    H_y = entropy(p_y)
    H_y_given_x = np.mean([entropy(probs[i]) for i in range(B)])
    return H_y - H_y_given_x

def compute_flops(model, X):
    B = X.shape[0]
    flops_per_expert = B * (model.input_dim * model.hidden + model.hidden * model.output_dim)
    active_experts = np.mean(np.sum(model.forward(X)[1] > 0, axis=1))
    return int(flops_per_expert * active_experts)

# =========================
# Runner
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
        model = V53FMoE(input_dim=X.shape[1], hidden=16, output_dim=len(np.unique(y)), n_experts=5)

        n_epochs = 3
        batch_size = 32
        for epoch in range(n_epochs):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                acc, gate = model.train_step(X[batch], y[batch])

        Ent = entropy(gate)
        Coll = collapse(gate)
        MI = mutual_information(gate, y)
        usage = np.mean(gate, axis=0)
        FLOPs = compute_flops(model, X)

        print(f"V5.3F MOE ACC: {acc:.4f}")
        print(f"Entropy: {Ent:.4f}")
        print(f"Collapse: {Coll:.4f}")
        print(f"MI: {MI:.4f}")
        print(f"Usage: {np.round(usage,3)}")
        print(f"FLOPs (est.): {FLOPs}")

        results[name] = {
            "ACC": float(acc),
            "Entropy": float(Ent),
            "Collapse": float(Coll),
            "MI": float(MI),
            "Usage": usage.tolist(),
            "FLOPs": FLOPs
        }

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_3F_flop_efficient.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> resultados_finais/v5_3F_flop_efficient.json")

if __name__ == "__main__":
    run()