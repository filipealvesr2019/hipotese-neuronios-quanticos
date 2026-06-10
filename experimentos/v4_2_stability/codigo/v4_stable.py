# v4_stable.py
import sys
import os
import numpy as np
import json

# Adiciona raiz do projeto para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from scripts.bateria_completa import LinearLayer, MLPTradicional, make_circles, normalize, train_test_split, set_seed, softmax_crossentropy

# ===============================
# MultiStateLayer V4 (Stable + Load Balancing)
# ===============================
class MultiStateLayerV4Stable:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16, lambda_lb=0.001):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        self.lambda_lb = lambda_lb

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

        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        B = x.shape[0]
        states_out = [x @ w + b for w, b in zip(self.W, self.b)]
        self.cache['states'] = states_out

        h_g = np.maximum(x @ self.gate_W1 + self.gate_b1, 0)
        logits = (h_g @ self.gate_W2 + self.gate_b2) / self.temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # Top-1 hard routing
        hard = np.zeros_like(probs)
        idx = np.argmax(probs, axis=1)
        hard[np.arange(B), idx] = 1.0
        self.cache['probs'] = probs
        self.cache['hard'] = hard

        # Load balancing regularization (suave)
        lb_penalty = self.lambda_lb * np.var(np.mean(hard, axis=0))

        out = sum(hard[:, i:i+1] * states_out[i] for i in range(self.n_states))
        out += x @ self.skip_W + self.skip_b

        return out, lb_penalty

    def predict(self, x):
        out, _ = self.forward(x)
        return np.argmax(out, axis=1)

# ===============================
# MLPMultiEstado V4 Stable
# ===============================
class MLPMultiEstadoV4Stable:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16, lambda_lb=0.001):
        self.l1 = MultiStateLayerV4Stable(input_dim, h, n_states, seed, temperature, gate_hidden, lambda_lb)
        self.l2 = MultiStateLayerV4Stable(h, h, n_states, seed+10, temperature, gate_hidden, lambda_lb)
        self.l3 = LinearLayer(h, output_dim, seed+20)

    def forward(self, x):
        h1, _ = self.l1.forward(x)
        h1 = np.maximum(h1, 0)
        h2, _ = self.l2.forward(h1)
        h2 = np.maximum(h2, 0)
        out = self.l3.forward(h2)
        return out

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

# ===============================
# Função principal de teste de estabilidade
# ===============================
def run_v4_stable(num_seeds=10):
    print("=== V4 Stable Test — MNIST-like Synthetic ===")
    results = []

    X, y = make_circles(n_samples=2000, noise=0.1, seed=42)
    X, _, _ = normalize(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y)

    for seed in range(num_seeds):
        set_seed(seed)
        h = 64
        model = MLPMultiEstadoV4Stable(2, h, 2, n_states=4, seed=seed, lambda_lb=0.001)

        # Forward-only accuracy (não treina ainda, só valida estabilidade inicial)
        y_pred = model.predict(Xva)
        acc = np.mean(y_pred == yva)
        results.append({'seed': seed, 'accuracy': acc})

        print(f"Seed {seed}: Acc={acc:.4f}")

    # Salva resultados
    out_file = os.path.join(os.path.dirname(__file__), "../../resultados_finais/v4_stable_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados salvos em {out_file}")
    return results

# ===============================
# Rodar se script for principal
# ===============================
if __name__ == "__main__":
    run_v4_stable()