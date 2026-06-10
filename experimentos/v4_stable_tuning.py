# v4_stable_full.py
import numpy as np
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ===============================
# Utilitários
# ===============================
def set_seed(seed):
    np.random.seed(seed)

def softmax_crossentropy(logits, y):
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = e_logits / np.sum(e_logits, axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), y] -= 1
    grad /= n
    return loss, grad

def normalize(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X = (X - X_mean) / X_std
    return X, X_mean, X_std

def compute_entropy(dist):
    dist = dist + 1e-10
    return -np.sum(dist * np.log2(dist))

# ===============================
# MultiStateLayerV4
# ===============================
class MultiStateLayerV4:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)
        rng3 = np.random.RandomState(seed + 200)
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]
        self.gate_W1 = rng2.randn(input_dim, gate_hidden) * 0.1
        self.gate_b1 = np.zeros(gate_hidden)
        self.gate_W2 = rng2.randn(gate_hidden, n_states) * 0.1
        self.gate_b2 = np.zeros(n_states)
        self.skip_W = rng3.randn(input_dim, output_dim) * 0.1
        self.skip_b = np.zeros(output_dim)

    def forward(self, x):
        B = x.shape[0]
        states = [x @ w + b for w, b in zip(self.W, self.b)]
        h_g = np.maximum(x @ self.gate_W1 + self.gate_b1, 0)
        gate_logits = (h_g @ self.gate_W2 + self.gate_b2) / self.temperature
        gate_logits -= gate_logits.max(axis=1, keepdims=True)
        probs = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=1, keepdims=True)
        hard_probs = np.zeros_like(probs)
        hard_probs[np.arange(B), np.argmax(probs, axis=1)] = 1.0
        out = sum(hard_probs[:, i:i+1] * states[i] for i in range(self.n_states))
        out += x @ self.skip_W + self.skip_b
        self.gate_weights_cached = hard_probs
        self.states_cached = states
        self.probs_cached = probs
        return out

    def analyze_gate(self, x):
        h_g = np.maximum(x @ self.gate_W1 + self.gate_b1, 0)
        gate_logits = (h_g @ self.gate_W2 + self.gate_b2) / self.temperature
        gate_logits -= gate_logits.max(axis=1, keepdims=True)
        probs = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=1, keepdims=True)
        hard_probs = np.zeros_like(probs)
        hard_probs[np.arange(probs.shape[0]), np.argmax(probs, axis=1)] = 1.0
        return hard_probs, probs

# ===============================
# MLPMultiEstadoV4
# ===============================
class MLPMultiEstadoV4:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16):
        self.l1 = MultiStateLayerV4(input_dim, h, n_states, seed, temperature, gate_hidden)
        self.l2 = MultiStateLayerV4(h, h, n_states, seed + 10, temperature, gate_hidden)
        self.l3_W = np.random.randn(h, output_dim) * 0.1
        self.l3_b = np.zeros(output_dim)

    def forward(self, x):
        self.h1 = np.maximum(self.l1.forward(x), 0)
        self.h2 = np.maximum(self.l2.forward(self.h1), 0)
        out = self.h2 @ self.l3_W + self.l3_b
        return out

    def analyze_gate(self, x):
        g1, _ = self.l1.analyze_gate(x)
        h1 = np.maximum(self.l1.forward(x), 0)
        g2, _ = self.l2.analyze_gate(h1)
        return g1, g2

# ===============================
# Dataset
# ===============================
def make_synthetic_mnist(n_samples=2000, n_features=784, n_classes=10, seed=0):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=50,
        n_classes=n_classes,
        random_state=seed
    )
    X, _, _ = normalize(X)
    return X, y

# ===============================
# Experimento de Entropy vs Accuracy
# ===============================
def run_entropy_accuracy(n_seeds=10):
    results = []
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        X, y = make_synthetic_mnist(seed=seed)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        hidden = 96
        model = MLPMultiEstadoV4(input_dim=n_features, h=hidden, output_dim=n_classes, n_states=2, seed=seed)
        logits = model.forward(X)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        g1, g2 = model.analyze_gate(X)
        l1_ent = float(compute_entropy(np.mean(g1, axis=0)))
        l2_ent = float(compute_entropy(np.mean(g2, axis=0)))
        print(f"Acc={acc:.4f} | L1={l1_ent:.4f} | L2={l2_ent:.4f}")
        results.append({"seed": seed, "acc": acc, "l1_entropy": l1_ent, "l2_entropy": l2_ent})

    output_path = "resultados_finais/v4_entropy_accuracy_full.json"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {output_path}")
    return results

# ===============================
# Run direto
# ===============================
if __name__ == "__main__":
    print("=== V4 Entropy vs Accuracy — Full Standalone ===")
    run_entropy_accuracy(n_seeds=10)