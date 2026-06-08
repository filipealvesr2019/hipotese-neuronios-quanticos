import numpy as np
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bateria_completa import LinearLayer, MLPTradicional, make_circles, normalize, train_test_split, set_seed, softmax_crossentropy

# ============================================================
# MultiStateLayer V4 — Sparse Routing (Top-1) via STE
# ============================================================
class MultiStateLayerV4:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)
        rng3 = np.random.RandomState(seed + 200)

        # Estados: W e b
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate MLP
        self.gate_W1 = rng2.randn(input_dim, gate_hidden) * 0.1
        self.gate_b1 = np.zeros(gate_hidden)
        self.gate_W2 = rng2.randn(gate_hidden, n_states) * 0.1
        self.gate_b2 = np.zeros(n_states)

        # Skip Connection
        self.skip_W = rng3.randn(input_dim, output_dim) * 0.1
        self.skip_b = np.zeros(output_dim)

        self.x_cached = None
        self.states_cached = None
        self.gate_h_cached = None
        self.probs_cached = None
        self.gate_weights_cached = None
        self.out_cached = None

    def forward(self, x):
        self.x_cached = x
        B = x.shape[0]

        # Estados
        self.states_cached = [x @ w + b for w, b in zip(self.W, self.b)]

        # Gate MLP
        h_g_pre = x @ self.gate_W1 + self.gate_b1
        self.gate_h_cached = np.maximum(h_g_pre, 0)
        gate_logits = (self.gate_h_cached @ self.gate_W2 + self.gate_b2) / self.temperature
        
        # Softmax
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        exp = np.exp(gate_logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.probs_cached = probs

        # Top-1 Hard Routing
        hard_probs = np.zeros_like(probs)
        indices = np.argmax(probs, axis=1)
        hard_probs[np.arange(B), indices] = 1.0
        self.gate_weights_cached = hard_probs

        # Saida
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

        # Skip connection
        dskip_W = (x.T @ grad) / B + l2 * self.skip_W
        dskip_b = np.mean(grad, axis=0)
        dx_skip = grad @ self.skip_W.T
        self.skip_W -= lr * dskip_W
        self.skip_b -= lr * dskip_b

        # Estados (Gradients only flow through the activated state due to gw)
        dx_states = np.zeros_like(x)
        for i in range(self.n_states):
            g = grad * gw[:, i:i+1] # gw is 1 or 0
            dW = (x.T @ g) / B + l2 * self.W[i]
            db = np.mean(g, axis=0)
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
            dx_states += g @ self.W[i].T

        # Gate (Straight-Through Estimator)
        dL_dgw = np.zeros((B, self.n_states))
        for i in range(self.n_states):
            dL_dgw[:, i] = np.sum(grad * self.states_cached[i], axis=1)

        # STE: treat dL_dgw as dL_dprobs
        probs = self.probs_cached
        dL_dlogits = np.zeros((B, self.n_states))
        for k in range(self.n_states):
            for j in range(self.n_states):
                jacob = probs[:, j] * (1.0 if j == k else 0.0 - probs[:, k])
                dL_dlogits[:, k] += dL_dgw[:, j] * jacob

        dL_dlogits_orig = dL_dlogits / self.temperature

        dgate_W2 = (self.gate_h_cached.T @ dL_dlogits_orig) / B + l2 * self.gate_W2
        dgate_b2 = np.mean(dL_dlogits_orig, axis=0)
        
        dh_g = dL_dlogits_orig @ self.gate_W2.T
        dh_g[self.gate_h_cached <= 0] = 0
        
        dgate_W1 = (x.T @ dh_g) / B + l2 * self.gate_W1
        dgate_b1 = np.mean(dh_g, axis=0)
        
        dx_gate = dh_g @ self.gate_W1.T
        
        self.gate_W2 -= lr * dgate_W2
        self.gate_b2 -= lr * dgate_b2
        self.gate_W1 -= lr * dgate_W1
        self.gate_b1 -= lr * dgate_b1

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
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        hard_probs = np.zeros_like(probs)
        indices = np.argmax(probs, axis=1)
        hard_probs[np.arange(probs.shape[0]), indices] = 1.0
        return hard_probs


class MLPMultiEstadoV4:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16):
        self.l1 = MultiStateLayerV4(input_dim, h, n_states, seed, temperature, gate_hidden)
        self.l2 = MultiStateLayerV4(h, h, n_states, seed + 10, temperature, gate_hidden)
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

def get_ms_v4_hidden(target_params, input_dim, output_dim, n_states, gate_hidden=16):
    best = 1
    best_diff = float('inf')
    for h in range(1, 512):
        p1 = n_states * (input_dim * h + h) 
        p1 += (input_dim * gate_hidden + gate_hidden) + (gate_hidden * n_states + n_states) 
        p1 += (input_dim * h + h) 
        p2 = n_states * (h * h + h)
        p2 += (h * gate_hidden + gate_hidden) + (gate_hidden * n_states + n_states) 
        p2 += (h * h + h) 
        p3 = h * output_dim + output_dim
        p = p1 + p2 + p3
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best = h
    return best

def compute_entropy(dist):
    dist = dist + 1e-10
    return -np.sum(dist * np.log2(dist))

def run_v4_experiment():
    print("==================================================")
    print(" V4 SPARSE ROUTING (Top-1) + Skip + MLP Gate")
    print("==================================================")
    
    set_seed(42)
    X, y = make_circles(n_samples=2000, noise=0.1, seed=42)
    X, _, _ = normalize(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y)

    mt = MLPTradicional(2, 64, 2, 42)
    
    # Treinando o tradicional como baseline
    n = Xtr.shape[0]
    for ep in range(300):
        lr_ = 0.01 * (0.99 ** ep)
        idx = np.random.permutation(n)
        for s in range(0, n, 64):
            e = min(s + 64, n)
            ids = idx[s:e]
            Xb, yb = Xtr[ids], ytr[ids]
            logits = mt.forward(Xb)
            _, grad = softmax_crossentropy(logits, yb)
            mt.backward(grad, lr_)
    
    at = np.mean(mt.predict(Xva) == yva)
    print(f"Tradicional Acc: {at*100:.2f}% (Params: {mt.params()})")

    pt = mt.params()
    h_v4 = get_ms_v4_hidden(pt, 2, 2, 4, gate_hidden=16)
    mv4 = MLPMultiEstadoV4(2, h_v4, 2, 4, seed=42, temperature=1.0, gate_hidden=16)
    print(f"MultiEstado V4 (Hidden: {h_v4}, Params: {mv4.params()})")

    history = []
    # Treinando V4
    for ep in range(300):
        lr_ = 0.01 * (0.99 ** ep)
        idx = np.random.permutation(n)
        for s in range(0, n, 64):
            e = min(s + 64, n)
            ids = idx[s:e]
            Xb, yb = Xtr[ids], ytr[ids]
            logits = mv4.forward(Xb)
            _, grad = softmax_crossentropy(logits, yb)
            mv4.backward(grad, lr_)
            
        if ep % 30 == 0 or ep == 299:
            g1, g2 = mv4.analyze_gate(Xva)
            dist1 = np.mean(g1, axis=0)
            dist2 = np.mean(g2, axis=0)
            history.append({
                'epoch': ep,
                'l1_dist': dist1.tolist(),
                'l2_dist': dist2.tolist(),
                'l1_ent': float(compute_entropy(dist1)),
                'l2_ent': float(compute_entropy(dist2))
            })

    av4 = np.mean(mv4.predict(Xva) == yva)
    print(f"V4 Sparse Acc: {av4*100:.2f}%\n")

    print("--- Dinâmica do Sparse Gate (Evolução da Entropia) ---")
    print(f"{'Epoca':>5} | {'L1 Dist':>25} | {'L2 Dist':>25} | {'L1 H':>6} | {'L2 H':>6}")
    for h in history:
        ep = h['epoch']
        d1 = h['l1_dist']
        d2 = h['l2_dist']
        s1 = f"[{d1[0]:.2f}, {d1[1]:.2f}, {d1[2]:.2f}, {d1[3]:.2f}]"
        s2 = f"[{d2[0]:.2f}, {d2[1]:.2f}, {d2[2]:.2f}, {d2[3]:.2f}]"
        print(f"{ep:>5} | {s1:>25} | {s2:>25} | {h['l1_ent']:>6.3f} | {h['l2_ent']:>6.3f}")

    # TESTE B: Ablação
    print("\n--- Ablação (Sparse State) ---")
    acc_original = np.mean(mv4.predict(Xva) == yva) * 100
    print(f"Desempenho Original: {acc_original:.2f}%")
    
    for state_idx in range(4):
        w_backup = mv4.l1.W[state_idx].copy()
        b_backup = mv4.l1.b[state_idx].copy()
        
        mv4.l1.W[state_idx].fill(0)
        mv4.l1.b[state_idx].fill(0)
        
        acc_ablated = np.mean(mv4.predict(Xva) == yva) * 100
        impact = acc_original - acc_ablated
        
        print(f"L1_E{state_idx+1} removido -> {acc_ablated:.2f}% (Impacto: {impact:+.2f}pp)")
        
        mv4.l1.W[state_idx] = w_backup
        mv4.l1.b[state_idx] = b_backup

if __name__ == "__main__":
    run_v4_experiment()
