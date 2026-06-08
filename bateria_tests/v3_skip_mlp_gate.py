import numpy as np
import random
import time
import json
import os
import sys

# Add parent to path for common imports if any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bateria_completa import LinearLayer, MLPTradicional, make_circles, normalize, train_test_split, train, set_seed

# ============================================================
# MultiStateLayer V3 — Gate MLP e Skip Connections
# ============================================================
class MultiStateLayerV3:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=0.5, gate_hidden=16):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)
        rng3 = np.random.RandomState(seed + 200)

        # Estados: cada um com W e b próprios
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate MLP: input_dim -> gate_hidden -> n_states
        self.gate_W1 = rng2.randn(input_dim, gate_hidden) * 0.1
        self.gate_b1 = np.zeros(gate_hidden)
        self.gate_W2 = rng2.randn(gate_hidden, n_states) * 0.1
        self.gate_b2 = np.zeros(n_states)

        # Skip Connection: input_dim -> output_dim
        self.skip_W = rng3.randn(input_dim, output_dim) * 0.1
        self.skip_b = np.zeros(output_dim)

        # Caches
        self.x_cached = None
        self.states_cached = None
        self.gate_h_cached = None
        self.gate_weights_cached = None
        self.out_cached = None

    def forward(self, x):
        self.x_cached = x
        B = x.shape[0]

        # Estados
        self.states_cached = [x @ w + b for w, b in zip(self.W, self.b)]

        # Gate MLP
        h_g_pre = x @ self.gate_W1 + self.gate_b1
        self.gate_h_cached = np.maximum(h_g_pre, 0) # ReLU
        gate_logits = (self.gate_h_cached @ self.gate_W2 + self.gate_b2) / self.temperature
        
        # Softmax
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        exp = np.exp(gate_logits)
        self.gate_weights_cached = exp / np.sum(exp, axis=1, keepdims=True)

        # Combination
        out = np.zeros((B, self.output_dim))
        for i in range(self.n_states):
            out += self.gate_weights_cached[:, i:i+1] * self.states_cached[i]
            
        # Skip Connection
        skip_out = x @ self.skip_W + self.skip_b
        out += skip_out
        
        self.out_cached = out
        return out

    def backward(self, grad, lr, l2=1e-4):
        B = grad.shape[0]
        x = self.x_cached
        gw = self.gate_weights_cached

        # Gradientes para skip connection
        dskip_W = (x.T @ grad) / B + l2 * self.skip_W
        dskip_b = np.mean(grad, axis=0)
        dx_skip = grad @ self.skip_W.T
        
        self.skip_W -= lr * dskip_W
        self.skip_b -= lr * dskip_b

        # Gradientes para os estados
        dx_states = np.zeros_like(x)
        for i in range(self.n_states):
            g = grad * gw[:, i:i+1]
            dW = (x.T @ g) / B + l2 * self.W[i]
            db = np.mean(g, axis=0)
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
            dx_states += g @ self.W[i].T

        # Gradiente para o gate
        dL_dgw = np.zeros((B, self.n_states))
        for i in range(self.n_states):
            dL_dgw[:, i] = np.sum(grad * self.states_cached[i], axis=1)

        dL_dlogits = np.zeros((B, self.n_states))
        for k in range(self.n_states):
            for j in range(self.n_states):
                jacob = gw[:, j] * (1.0 if j == k else 0.0 - gw[:, k])
                dL_dlogits[:, k] += dL_dgw[:, j] * jacob

        dL_dlogits_orig = dL_dlogits / self.temperature

        # MLP Gate backward
        dgate_W2 = (self.gate_h_cached.T @ dL_dlogits_orig) / B + l2 * self.gate_W2
        dgate_b2 = np.mean(dL_dlogits_orig, axis=0)
        
        dh_g = dL_dlogits_orig @ self.gate_W2.T
        dh_g[self.gate_h_cached <= 0] = 0 # ReLU derivative
        
        dgate_W1 = (x.T @ dh_g) / B + l2 * self.gate_W1
        dgate_b1 = np.mean(dh_g, axis=0)
        
        dx_gate = dh_g @ self.gate_W1.T
        
        # Update Gate
        self.gate_W2 -= lr * dgate_W2
        self.gate_b2 -= lr * dgate_b2
        self.gate_W1 -= lr * dgate_W1
        self.gate_b1 -= lr * dgate_b1

        # Gradiente total para x
        dx = dx_states + dx_gate + dx_skip
        return dx

    def params(self):
        p = sum(w.size for w in self.W) + sum(b.size for b in self.b)
        p += self.gate_W1.size + self.gate_b1.size
        p += self.gate_W2.size + self.gate_b2.size
        p += self.skip_W.size + self.skip_b.size
        return p

    def get_gate_weights(self, x):
        h_g_pre = x @ self.gate_W1 + self.gate_b1
        gate_h = np.maximum(h_g_pre, 0)
        gate_logits = (gate_h @ self.gate_W2 + self.gate_b2) / self.temperature
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        exp = np.exp(gate_logits)
        return exp / np.sum(exp, axis=1, keepdims=True)


class MLPMultiEstadoV3:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42, temperature=0.5, gate_hidden=16):
        self.l1 = MultiStateLayerV3(input_dim, h, n_states, seed, temperature, gate_hidden)
        self.l2 = MultiStateLayerV3(h, h, n_states, seed + 10, temperature, gate_hidden)
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
        g1 = self.l1.get_gate_weights(x)
        h1 = np.maximum(self.l1.forward(x), 0)
        g2 = self.l2.get_gate_weights(h1)
        return g1, g2
        
def get_ms_v3_hidden(target_params, input_dim, output_dim, n_states, gate_hidden=16):
    best = 1
    best_diff = float('inf')
    for h in range(1, 512):
        # L1
        p1 = n_states * (input_dim * h + h) 
        p1 += (input_dim * gate_hidden + gate_hidden) + (gate_hidden * n_states + n_states) # gate
        p1 += (input_dim * h + h) # skip
        
        # L2
        p2 = n_states * (h * h + h)
        p2 += (h * gate_hidden + gate_hidden) + (gate_hidden * n_states + n_states) # gate
        p2 += (h * h + h) # skip
        
        p3 = h * output_dim + output_dim
        
        p = p1 + p2 + p3
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best = h
    return best

if __name__ == "__main__":
    print("==================================================")
    print(" Testando V3: MLP Gate + Skip Connections")
    print("==================================================")
    
    set_seed(42)
    X, y = make_circles(n_samples=2000, noise=0.1, seed=42)
    X, _, _ = normalize(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y)

    mt = MLPTradicional(2, 64, 2, 42)
    at, tt, _ = train(mt, Xtr, ytr, Xva, yva, epochs=300, lr=0.01)
    print(f"Tradicional Acc: {at*100:.2f}% (Params: {mt.params()})")

    pt = mt.params()
    h_v3 = get_ms_v3_hidden(pt, 2, 2, 4, gate_hidden=16)
    mv3 = MLPMultiEstadoV3(2, h_v3, 2, 4, 42, temperature=0.3, gate_hidden=16)
    av3, tv3, _ = train(mv3, Xtr, ytr, Xva, yva, epochs=300, lr=0.01)
    print(f"MultiEstado V3 Acc: {av3*100:.2f}% (Params: {mv3.params()}, Hidden: {h_v3})")

    # TESTE A: Visualizar gate
    print("\n--- TESTE A: Visualizar Gate ---")
    g1, g2 = mv3.analyze_gate(Xva[:5])
    print("Amostra de 5 inputs - Layer 1 Gate Weights (qual estado domina):")
    print(np.round(g1, 3))
    print("Layer 2 Gate Weights:")
    print(np.round(g2, 3))

    # TESTE C: Distribuição de uso
    print("\n--- TESTE C: Distribuicao de uso ---")
    g1_all, g2_all = mv3.analyze_gate(Xva)
    dist1 = np.mean(g1_all, axis=0) * 100
    dist2 = np.mean(g2_all, axis=0) * 100
    print(f"Layer 1 Uso Médio dos Estados: E1={dist1[0]:.1f}% E2={dist1[1]:.1f}% E3={dist1[2]:.1f}% E4={dist1[3]:.1f}%")
    print(f"Layer 2 Uso Médio dos Estados: E1={dist2[0]:.1f}% E2={dist2[1]:.1f}% E3={dist2[2]:.1f}% E4={dist2[3]:.1f}%")

    # TESTE B: Ablação
    print("\n--- TESTE B: Ablacao (Zerar estado 1 e ver impacto) ---")
    acc_original = np.mean(mv3.predict(Xva) == yva) * 100
    print(f"Desempenho Original: {acc_original:.2f}%")
    
    for state_idx in range(4):
        # Backup weights
        w_backup = mv3.l1.W[state_idx].copy()
        b_backup = mv3.l1.b[state_idx].copy()
        
        # Zero state
        mv3.l1.W[state_idx].fill(0)
        mv3.l1.b[state_idx].fill(0)
        
        acc_ablated = np.mean(mv3.predict(Xva) == yva) * 100
        impact = acc_original - acc_ablated
        
        print(f"L1_E{state_idx+1} removido -> {acc_ablated:.2f}% (Impacto: {impact:+.2f}pp)")
        
        # Restore
        mv3.l1.W[state_idx] = w_backup
        mv3.l1.b[state_idx] = b_backup
