import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bateria_completa import LinearLayer, MLPTradicional, make_circles, normalize, train_test_split, set_seed, softmax_crossentropy
from experimentos.v4_sparse_routing.codigo.v4_sparse_routing import compute_entropy, get_ms_v4_hidden

# ============================================================
# MultiStateLayer V4.1 — Sparse Routing + Load Balancing
# ============================================================
class MultiStateLayerV4_1:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16, alpha_bal=0.1):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        self.alpha_bal = alpha_bal
        
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

        self.x_cached = None
        self.states_cached = None
        self.gate_h_cached = None
        self.probs_cached = None
        self.gate_weights_cached = None
        self.out_cached = None

    def forward(self, x):
        self.x_cached = x
        B = x.shape[0]

        self.states_cached = [x @ w + b for w, b in zip(self.W, self.b)]

        h_g_pre = x @ self.gate_W1 + self.gate_b1
        self.gate_h_cached = np.maximum(h_g_pre, 0)
        gate_logits = (self.gate_h_cached @ self.gate_W2 + self.gate_b2) / self.temperature
        
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        exp = np.exp(gate_logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.probs_cached = probs

        hard_probs = np.zeros_like(probs)
        indices = np.argmax(probs, axis=1)
        hard_probs[np.arange(B), indices] = 1.0
        self.gate_weights_cached = hard_probs

        out = np.zeros((B, self.output_dim))
        for i in range(self.n_states):
            out += self.gate_weights_cached[:, i:i+1] * self.states_cached[i]
            
        skip_out = x @ self.skip_W + self.skip_b
        out += skip_out
        
        self.out_cached = out
        return out

    def backward(self, grad, lr, l2=1e-4):
        B = grad.shape[0]
        x = self.x_cached
        gw = self.gate_weights_cached
        probs = self.probs_cached

        dskip_W = (x.T @ grad) / B + l2 * self.skip_W
        dskip_b = np.mean(grad, axis=0)
        dx_skip = grad @ self.skip_W.T
        self.skip_W -= lr * dskip_W
        self.skip_b -= lr * dskip_b

        dx_states = np.zeros_like(x)
        for i in range(self.n_states):
            g = grad * gw[:, i:i+1]
            dW = (x.T @ g) / B + l2 * self.W[i]
            db = np.mean(g, axis=0)
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
            dx_states += g @ self.W[i].T

        dL_dgw = np.zeros((B, self.n_states))
        for i in range(self.n_states):
            dL_dgw[:, i] = np.sum(grad * self.states_cached[i], axis=1)

        dL_dprobs = dL_dgw.copy()
        
        # === LOAD BALANCING LOSS GRADIENT ===
        # Loss_aux = alpha * sum(dist^2) onde dist = mean(probs, axis=0)
        # d(Loss_aux)/d(probs[b,k]) = (2 * alpha / B) * dist[k]
        if self.alpha_bal > 0:
            dist = np.mean(probs, axis=0)
            dL_dprobs += (2.0 * self.alpha_bal / B) * dist

        dL_dlogits = np.zeros((B, self.n_states))
        for k in range(self.n_states):
            for j in range(self.n_states):
                jacob = probs[:, j] * (1.0 if j == k else 0.0 - probs[:, k])
                dL_dlogits[:, k] += dL_dprobs[:, j] * jacob

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

class MLPMultiEstadoV4_1:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42, temperature=1.0, gate_hidden=16, alpha_bal=0.1):
        self.l1 = MultiStateLayerV4_1(input_dim, h, n_states, seed, temperature, gate_hidden, alpha_bal)
        self.l2 = MultiStateLayerV4_1(h, h, n_states, seed + 10, temperature, gate_hidden, alpha_bal)
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

def run_v4_1_30_seeds():
    print("==================================================")
    print(" PASSO 2 e 3: V4.1 (LOAD BALANCING) vs TRADICIONAL")
    print(" 30 Seeds - Dataset: Círculos (2000 amostras, noise=0.1)")
    print("==================================================")
    
    n_seeds = 30
    results_trad = []
    results_v4_1 = []
    
    epochs = 300
    lr = 0.01
    
    # Parâmetro de balanceamento (força do gradiente extra para espalhar os estados)
    ALPHA_BAL = 5.0
    
    print(f"{'Seed':>5} | {'Trad Acc':>10} | {'V4.1 Acc':>10} | {'Diff':>8} | {'L2 Entropia':>11}")
    print("-" * 58)

    for seed in range(1, n_seeds + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        n = Xtr.shape[0]

        mt = MLPTradicional(2, 64, 2, seed)
        for ep in range(epochs):
            lr_ = lr * (0.99 ** ep)
            idx = np.random.permutation(n)
            for s in range(0, n, 64):
                e = min(s + 64, n)
                ids = idx[s:e]
                Xb, yb = Xtr[ids], ytr[ids]
                logits = mt.forward(Xb)
                _, grad = softmax_crossentropy(logits, yb)
                mt.backward(grad, lr_)
        at = np.mean(mt.predict(Xva) == yva) * 100
        results_trad.append(at)

        pt = mt.params()
        h_v4 = get_ms_v4_hidden(pt, 2, 2, 4, gate_hidden=16)
        mv4_1 = MLPMultiEstadoV4_1(2, h_v4, 2, 4, seed=seed, temperature=1.0, gate_hidden=16, alpha_bal=ALPHA_BAL)
        
        for ep in range(epochs):
            lr_ = lr * (0.99 ** ep)
            idx = np.random.permutation(n)
            for s in range(0, n, 64):
                e = min(s + 64, n)
                ids = idx[s:e]
                Xb, yb = Xtr[ids], ytr[ids]
                logits = mv4_1.forward(Xb)
                _, grad = softmax_crossentropy(logits, yb)
                mv4_1.backward(grad, lr_)
        
        av4_1 = np.mean(mv4_1.predict(Xva) == yva) * 100
        results_v4_1.append(av4_1)

        diff = av4_1 - at
        
        # Medindo Entropia da Layer 2 para garantir que balanceamento funcionou
        _, g2 = mv4_1.analyze_gate(Xva)
        dist2 = np.mean(g2, axis=0)
        ent2 = compute_entropy(dist2)
        
        print(f"{seed:>5} | {at:>9.2f}% | {av4_1:>9.2f}% | {diff:>+7.2f}% | {ent2:>11.2f}")

    print("\n--- RESUMO ESTATÍSTICO (30 SEEDS) ---")
    trad_mean, trad_std = np.mean(results_trad), np.std(results_trad)
    v4_1_mean, v4_1_std = np.mean(results_v4_1), np.std(results_v4_1)
    
    print(f"Tradicional : {trad_mean:.2f}% ± {trad_std:.2f}%")
    print(f"V4.1 Sparse : {v4_1_mean:.2f}% ± {v4_1_std:.2f}%")
    
    wins_v4 = sum(1 for a, b in zip(results_v4_1, results_trad) if a > b)
    wins_trad = sum(1 for a, b in zip(results_v4_1, results_trad) if b > a)
    ties = n_seeds - wins_v4 - wins_trad
    
    print(f"\nVitórias V4.1: {wins_v4}/{n_seeds}")
    print(f"Vitórias Trad: {wins_trad}/{n_seeds}")
    print(f"Empates      : {ties}/{n_seeds}")

if __name__ == "__main__":
    run_v4_1_30_seeds()
