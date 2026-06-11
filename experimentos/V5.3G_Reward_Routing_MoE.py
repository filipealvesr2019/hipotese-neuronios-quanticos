import numpy as np
import os
import json

# =========================
# V5.3G — Reward Routing MoE
# =========================
class V53GRewardMoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_experts=5, top_k=2, lr=0.01, seed=42):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.lr = lr

        rng = np.random.RandomState(seed)

        # Experts
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_experts)]
        self.b1 = [np.zeros(hidden) for _ in range(n_experts)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_experts)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_experts)]

        # Gate
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_experts) * 0.1
        self.gb2 = np.zeros(n_experts)

    # Softmax
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    # Forward pass
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
        topk_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]

        gate = np.zeros_like(gate_probs)
        for b in range(B):
            for idx in topk_idx[b]:
                gate[b, idx] = 1.0 / self.top_k

        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list

    # Cross-entropy loss
    def loss(self, logits, y):
        probs = self.softmax(logits)
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    # Train step with reward routing
    def train_step(self, X, y):
        B = X.shape[0]
        logits, gate, out_list, h_list = self.forward(X)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        # Experts update
        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)
            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

        # Reward routing for gate (policy gradient)
        rewards = np.zeros_like(gate)
        for i in range(self.n_experts):
            pred = np.argmax(out_list[i], axis=1)
            rewards[:, i] = (pred == y).astype(float)

        # Gate gradient: simple REINFORCE
        gate_grad = gate * (rewards - np.mean(rewards, axis=1, keepdims=True))
        g_hidden = X @ self.gW1 + self.gb1
        g_hidden[g_hidden <= 0] = 0
        self.gW2 -= self.lr * g_hidden.T @ gate_grad
        self.gb2 -= self.lr * np.sum(gate_grad, axis=0)

        acc = np.mean(np.argmax(logits, axis=1) == y)
        return acc, gate

# =========================
# Datasets
# =========================
def make_xor(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, (n, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    return X, y

def make_gaussian(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n//2, 2) + np.array([0, 0])
    X1 = rng.randn(n//2, 2) + np.array([3, 3])
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def make_spiral(n=200, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n_classes = 2
    X = []
    y = []
    for j in range(n_classes):
        r = np.linspace(0.0, 1, n//n_classes)
        t = np.linspace(j*3.14, (j+1)*3.14, n//n_classes) + rng.randn(n//n_classes)*noise
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        X.append(np.vstack([x1, x2]).T)
        y.append([j]*(n//n_classes))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def make_mnist_like(n=200, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y

# =========================
# Metrics
# =========================
def entropy(probs):
    p = np.mean(probs, axis=0) + 1e-9
    return -np.sum(p * np.log(p))

def collapse(probs):
    usage = np.mean(probs, axis=0)
    return np.sum(usage**2)

def mutual_information(probs, y):
    B = probs.shape[0]
    p_y = np.mean(probs, axis=0)
    H_y = -np.sum(p_y * np.log(p_y + 1e-9))
    H_y_given_x = np.mean([-np.sum(probs[i]*np.log(probs[i]+1e-9)) for i in range(B)])
    MI = H_y - H_y_given_x
    return MI

def compute_flops(X, model):
    B = X.shape[0]
    hidden = model.hidden
    n_experts = model.n_experts
    input_dim = model.input_dim
    output_dim = model.output_dim
    flops_per_expert = B * (input_dim*hidden + hidden*output_dim)
    active_experts = np.mean(np.sum(model.forward(X)[1]>0, axis=1))
    return int(flops_per_expert * active_experts)

# =========================
# Run Experiment
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
        model = V53GRewardMoE(input_dim=X.shape[1], hidden=32, output_dim=len(np.unique(y)))
        n_epochs = 3

        for epoch in range(n_epochs):
            idx = np.random.permutation(len(X))
            batch_size = 32
            for i in range(0, len(X), batch_size):
                batch = idx[i:i+batch_size]
                acc, gate_probs = model.train_step(X[batch], y[batch])

        Ent = entropy(gate_probs)
        Coll = collapse(gate_probs)
        MI = mutual_information(gate_probs, y)
        FLOPs = compute_flops(X, model)
        Usage = np.mean(gate_probs, axis=0)


        print(f"V5.3G MOE ACC: {acc:.4f}")
        print(f"Entropy: {Ent:.4f}")
        print(f"Collapse: {Coll:.4f}")
        print(f"MI: {MI:.4f}")
        print(f"Usage: {np.round(Usage, 3)}")
        print(f"FLOPs (est.): {FLOPs}\n")

        results[name] = {
            "acc": float(acc),
            "entropy": float(Ent),
            "collapse": float(Coll),
            "MI": float(MI),
            "Usage": Usage.tolist(),
            "FLOPs": int(FLOPs)
        }

    os.makedirs("resultados_finais", exist_ok=True)
    with open("resultados_finais/v5_3G_reward_routing.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v5_3G_reward_routing.json")

if __name__ == "__main__":
    run()