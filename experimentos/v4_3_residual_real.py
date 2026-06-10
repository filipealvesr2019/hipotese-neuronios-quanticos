import numpy as np
import os
import json

# ==============================
# Funções auxiliares
# ==============================
def set_seed(seed):
    np.random.seed(seed)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

def compute_entropy(dist):
    dist = dist + 1e-10
    return -np.sum(dist * np.log2(dist))

# ==============================
# Dataset sintético MNIST-like
# ==============================
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def make_synthetic_mnist(n_samples=2000, n_features=784, n_classes=10, seed=0):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=50,
        n_classes=n_classes, random_state=seed
    )
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return X, y

# ==============================
# V4 Sparse Layer + Residual
# ==============================
class V43Residual:
    def __init__(self, input_dim, output_dim, n_states=2, hidden_gate=8, temperature=1.0, skip_scale=0.2, seed=42):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_gate = hidden_gate
        self.temperature = temperature
        self.skip_scale = skip_scale

        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)
        rng3 = np.random.RandomState(seed + 200)

        # Estados
        self.W = [rng.randn(input_dim, output_dim)*0.05 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate MLP
        self.gate_W1 = rng2.randn(input_dim, hidden_gate)*0.05
        self.gate_b1 = np.zeros(hidden_gate)
        self.gate_W2 = rng2.randn(hidden_gate, n_states)*0.05
        self.gate_b2 = np.zeros(n_states)

        # Residual skip
        self.skip_W = rng3.randn(input_dim, output_dim)*0.05
        self.skip_b = np.zeros(output_dim)

    def forward(self, x):
        B = x.shape[0]
        # Saídas dos estados
        states_out = [x @ w + b for w,b in zip(self.W, self.b)]

        # Gate MLP
        h = np.maximum(0, x @ self.gate_W1 + self.gate_b1)
        logits = h @ self.gate_W2 + self.gate_b2
        logits /= self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)

        # Top-1 hard routing
        top1 = np.argmax(probs, axis=1)
        gw = np.zeros_like(probs)
        for i in range(B):
            gw[i, top1[i]] = 1.0

        # Combinação dos estados + residual
        out = sum(gw[:,i:i+1]*states_out[i] for i in range(self.n_states))
        out += self.skip_scale*(x @ self.skip_W + self.skip_b)
        return out, probs

# ==============================
# Teste múltiplas seeds
# ==============================
def run_v43_residual(n_seeds=10):
    results = []
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        X, y = make_synthetic_mnist(seed=seed)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed)

        model = V43Residual(input_dim=X.shape[1], output_dim=len(np.unique(y)), n_states=2, seed=seed)

        # Forward pass apenas
        logits, probs = model.forward(Xva)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == yva)
        l1_entropy = compute_entropy(np.mean(probs, axis=0))
        l2_entropy = compute_entropy(np.mean(probs, axis=1))
        print(f"V4.3 Residual acc (forward only): {acc:.4f} | L1={l1_entropy:.4f} | L2={l2_entropy:.4f}")

        results.append({
            "seed": seed,
            "acc": float(acc),
            "l1_entropy": float(l1_entropy),
            "l2_entropy": float(l2_entropy)
        })

    # Salvar resultados
    output_path = os.path.join("resultados_finais", "v4_3_residual_real_results.json")
    os.makedirs("resultados_finais", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados salvos em {output_path}")
    return results

if __name__ == "__main__":
    print("=== V4.3 Residual Test ===")
    run_v43_residual(n_seeds=10)