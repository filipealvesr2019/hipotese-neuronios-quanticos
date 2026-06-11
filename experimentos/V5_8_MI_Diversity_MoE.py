import numpy as np
import os
import json

# =========================
# V5.8 MI + Diversity Regularized MoE
# =========================
class V58MoE:
    def __init__(self, input_dim, hidden, output_dim, n_experts=5, topk=2,
                 lr=0.01, seed=42, temperature=1.5, diversity_lambda=0.1):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.topk = topk
        self.lr = lr
        self.temperature = temperature
        self.diversity_lambda = diversity_lambda

        rng = np.random.RandomState(seed)

        # Experts weights
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # Gate weights
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        B = x.shape[0]
        h_states, out_states = [], []
        for i in range(self.n_experts):
            h = np.maximum(0, x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        # Gate
        g = np.maximum(0, x @ self.gW1 + self.gb1)
        logits = g @ self.gW2 + self.gb2
        probs = self.softmax(logits / self.temperature)

        # Top-k routing
        topk_idx = np.argsort(probs, axis=1)[:, -self.topk:]
        gate = np.zeros_like(probs)
        for i in range(B):
            gate[i, topk_idx[i]] = 1.0 / self.topk

        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_experts))
        return out, probs, gate, h_states, out_states

    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    def train_step(self, x, y):
        B = x.shape[0]
        out, gate_probs, gate, h_states, out_states = self.forward(x)

        # Gradient output
        probs_out = self.softmax(out)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        # Update experts
        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_states[i] <= 0] = 0

            dW2 = h_states[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = x.T @ dh
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
        return acc, gate_probs

# =========================
# Datasets controlados
# =========================
def make_xor(n=200):
    X = np.random.randint(0, 2, size=(n,2))
    y = np.logical_xor(X[:,0], X[:,1]).astype(int)
    return X, y

def make_gaussian(n=200, d=2, c=3):
    X = []
    y = []
    rng = np.random.RandomState(0)
    for i in range(c):
        X.append(rng.randn(n//c, d) + i*2)
        y.append(np.full(n//c, i))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def make_spiral(n=200, d=2, c=3):
    X = []
    y = []
    rng = np.random.RandomState(0)
    for i in range(c):
        theta = np.linspace(0, 4*np.pi, n//c)
        r = np.linspace(0.0, 1, n//c)
        xi = r * np.cos(theta + i*2*np.pi/c)
        yi = r * np.sin(theta + i*2*np.pi/c)
        X.append(np.stack([xi, yi], axis=1))
        y.append(np.full(n//c, i))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def make_mnist_like(n=500, d=784, c=10):
    rng = np.random.RandomState(0)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y

# =========================
# Métricas MI, Entropy, Collapse
# =========================
def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))

def mutual_information(probs, y):
    B, n = probs.shape
    p_y = np.zeros(probs.shape[1])
    for i in range(B):
        p_y += probs[i]
    p_y /= B
    H_y = entropy(p_y)
    H_y_given_x = 0
    for i in range(B):
        H_y_given_x += entropy(probs[i])
    H_y_given_x /= B
    MI = H_y - H_y_given_x
    return MI

def collapse(probs):
    usage = np.mean(probs, axis=0)
    return np.sum(usage**2)

# =========================
# Run Experiment
# =========================
def run():
    datasets = {
        "xor": make_xor(),
        "gaussian": make_gaussian(),
        "spiral": make_spiral(),
        "mnist_like": make_mnist_like()
    }

    results = {}

    for name, (X, y) in datasets.items():
        print(f"\n===== DATASET: {name} =====")
        model = V58MoE(input_dim=X.shape[1], hidden=16, output_dim=len(np.unique(y)), n_experts=5)
        n_epochs = 3
        for epoch in range(n_epochs):
            idx = np.random.permutation(len(X))
            batch_size = 32
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                acc, gate_probs = model.train_step(X[batch], y[batch])

        Ent = entropy(np.mean(gate_probs, axis=0))
        Coll = collapse(gate_probs)
        MI = mutual_information(gate_probs, y)
        usage = np.mean(gate_probs, axis=0)

        print(f"ACC: {acc:.4f}")
        print(f"Entropy: {Ent:.4f}")
        print(f"Collapse: {Coll:.4f}")
        print(f"MI: {MI:.4f}")
        print(f"Usage: {np.round(usage,3)}")

        results[name] = {
            "ACC": float(acc),
            "Entropy": float(Ent),
            "Collapse": float(Coll),
            "MI": float(MI),
            "Usage": usage.tolist()
        }

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_8_mi_diversity_moe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> resultados_finais/v5_8_mi_diversity_moe.json")

if __name__ == "__main__":
    run()