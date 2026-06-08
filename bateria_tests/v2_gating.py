"""
MultiStateLayer V2 com gating input-dependente.
O gate gera pesos por amostra (nao globais).
4 estados, selecao inteligente via softmax.
"""
import numpy as np
import random
import time
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# ============================================================
# MultiStateLayer V2 — com gating input-dependente
# ============================================================
class MultiStateLayerV2:
    """
    4 estados internos + gate que gera pesos por amostra.
    Gate: Linear(input_dim -> n_states) + Softmax
    """
    def __init__(self, input_dim, output_dim, n_states=4, seed=42):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)

        # Estados: cada um com W e b próprios
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate: input_dim -> n_states (peso por estado, por amostra)
        self.gate_W = rng2.randn(input_dim, n_states) * 0.1
        self.gate_b = np.zeros(n_states)

        # Caches para backward
        self.x_cached = None
        self.states_cached = None
        self.gate_weights_cached = None
        self.out_cached = None

    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=0.5):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)

        # Estados: cada um com W e b próprios
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate: input_dim -> n_states (peso por estado, por amostra)
        self.gate_W = rng2.randn(input_dim, n_states) * 0.1
        self.gate_b = np.zeros(n_states)

        # Caches para backward
        self.x_cached = None
        self.states_cached = None
        self.gate_weights_cached = None
        self.out_cached = None

    def forward(self, x):
        self.x_cached = x
        B = x.shape[0]

        # Estados: cada um produz (B, output_dim)
        self.states_cached = [x @ w + b for w, b in zip(self.W, self.b)]

        # Gate: (B, n_states) -> softmax com temperatura
        gate_logits = (x @ self.gate_W + self.gate_b) / self.temperature
        # Softmax estável
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        exp = np.exp(gate_logits)
        self.gate_weights_cached = exp / np.sum(exp, axis=1, keepdims=True)

        # Saída: (B, output_dim) = sum_i gate[:,i:i+1] * states[i]
        out = np.zeros((B, self.output_dim))
        for i in range(self.n_states):
            out += self.gate_weights_cached[:, i:i+1] * self.states_cached[i]
        self.out_cached = out
        return out

    def backward(self, grad, lr, l2=1e-4):
        B = grad.shape[0]
        x = self.x_cached
        gw = self.gate_weights_cached  # (B, n_states)

        # Gradientes para cada estado
        for i in range(self.n_states):
            g = grad * gw[:, i:i+1]  # (B, output_dim)
            dW = (x.T @ g) / B + l2 * self.W[i]
            db = np.mean(g, axis=0)
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

        # Gradiente para o gate
        # dL/d(logit_k) = sum_d( dL/d(out_d) * state_k_d ) * softmax_k * (1 - softmax_k) + cross terms
        # Versão simplificada: grad_gate_logits = sum_d(grad * states[i]) * gw * (1 - gw)
        # Mas vamos fazer a derivada completa do softmax + weighted sum

        # dL/d(gate_weight[i]) = sum_d(grad * states[i])  (para cada amostra)
        dL_dgw = np.zeros((B, self.n_states))
        for i in range(self.n_states):
            dL_dgw[:, i] = np.sum(grad * self.states_cached[i], axis=1)

        # Jacobiano softmax: d(gw_j)/d(logit_k) = gw_j * (delta_jk - gw_k)
        # dL/d(logit_k) = sum_j dL/d(gw_j) * gw_j * (delta_jk - gw_k)
        dL_dlogits = np.zeros((B, self.n_states))
        for k in range(self.n_states):
            for j in range(self.n_states):
                jacob = gw[:, j] * (1.0 if j == k else 0.0 - gw[:, k])
                dL_dlogits[:, k] += dL_dgw[:, j] * jacob

        # Ajuste pela temperatura: gate_logits = original_logits / T
        dL_dlogits_orig = dL_dlogits / self.temperature

        # Gradiente do gate linear
        dgate_W = (x.T @ dL_dlogits_orig) / B + l2 * self.gate_W
        dgate_b = np.mean(dL_dlogits_orig, axis=0)
        self.gate_W -= lr * dgate_W
        self.gate_b -= lr * dgate_b

        # Gradiente para entrada (através dos estados e gate)
        dx = np.zeros_like(x)
        for i in range(self.n_states):
            dx += (grad * gw[:, i:i+1]) @ self.W[i].T
        # Através do gate: dL/dx += dL/dlogits_orig @ gate_W.T
        dx += dL_dlogits_orig @ self.gate_W.T

        return dx

    def params(self):
        p = sum(w.size for w in self.W) + sum(b.size for b in self.b)
        p += self.gate_W.size + self.gate_b.size
        return p

    def get_gate_weights(self, x):
        """Retorna pesos do gate para uma entrada."""
        gate_logits = (x @ self.gate_W + self.gate_b) / self.temperature
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        exp = np.exp(gate_logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def get_states(self, x):
        return [x @ w + b for w, b in zip(self.W, self.b)]


# ============================================================
# MLP MultiEstado V2
# ============================================================
class MLPMultiEstadoV2:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42, temperature=0.5):
        self.l1 = MultiStateLayerV2(input_dim, h, n_states, seed, temperature)
        self.l2 = MultiStateLayerV2(h, h, n_states, seed + 10, temperature)
        self.l3 = LinearLayer(h, output_dim, seed + 20)

    def forward(self, x):
        self.h1 = np.maximum(self.l1.forward(x), 0)
        self.h2 = np.maximum(self.l2.forward(self.h1), 0)
        return self.l3.forward(self.h2)

    def backward(self, grad, lr, l2=1e-4):
        grad = self.l3.backward(grad, lr, l2)
        grad[self.h2 <= 0] = 0
        grad = self.l2.backward(grad, lr, l2)
        grad[self.h1 <= 0] = 0
        self.l1.backward(grad, lr, l2)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def params(self):
        return self.l1.params() + self.l2.params() + self.l3.params()

    def analyze_gate(self, x):
        """Retorna os pesos do gate para cada camada."""
        g1 = self.l1.get_gate_weights(x)
        h = np.maximum(self.l1.forward(x), 0)
        g2 = self.l2.get_gate_weights(h)
        return g1, g2


def get_ms_v2_hidden(target_params, input_dim, output_dim, n_states):
    """Encontra hidden dim para V2 que mais se aproxima de target_params."""
    best = 1
    best_diff = float('inf')
    for h in range(1, 512):
        # V2: cada layer tem gate_W + gate_b extras
        # l1: n_states * (input_dim * h + h) + (input_dim * n_states + n_states)
        # l2: n_states * (h * h + h) + (h * n_states + n_states)
        # l3: h * output_dim + output_dim
        p1 = n_states * (input_dim * h + h) + (input_dim * n_states + n_states)
        p2 = n_states * (h * h + h) + (h * n_states + n_states)
        p3 = h * output_dim + output_dim
        p = p1 + p2 + p3
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best = h
    return best


if __name__ == "__main__":
    # Teste rápido: V2 consegue aprender XOR?
    from common import make_xor, normalize, train_test_split, train, set_seed

    set_seed(42)
    X, y = make_xor(42)
    X, _, _ = normalize(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y)

    # Tradicional
    mt = MLPTradicional(2, 8, 2, 42)
    at, tt, _ = train(mt, Xtr, ytr, Xva, yva, epochs=200)
    print(f"Tradicional XOR: {at*100:.2f}%")

    # V2 com params equivalentes
    pt = mt.params()
    h_v2 = get_ms_v2_hidden(pt, 2, 2, 4)
    mv2 = MLPMultiEstadoV2(2, h_v2, 2, 4, 42)
    av2, tv2, _ = train(mv2, Xtr, ytr, Xva, yva, epochs=200)
    print(f"MultiEstado V2 XOR: {av2*100:.2f}%")
    print(f"Params Trad={pt}, V2={mv2.params()}, hidden={h_v2}")

    # Gate analysis
    g1, g2 = mv2.analyze_gate(Xva[:10])
    print("\nGate weights (10 amostras val):")
    print("Layer1 gate:")
    print(np.array2string(g1, precision=3, suppress_small=True))
    print("Layer2 gate:")
    print(np.array2string(g2, precision=3, suppress_small=True))
