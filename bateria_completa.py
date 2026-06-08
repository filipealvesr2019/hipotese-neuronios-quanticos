"""
Bateria completa de 12 testes: MLP Tradicional vs MultiEstado (4 estados)
Tudo determinístico, 30 seeds por experimento.
"""
import numpy as np
import random
import time
import json
import os
from collections import defaultdict

# ============================================================
# GLOBALS
# ============================================================
N_SEEDS = 30
N_STATES = 4
EPOCHS = 200
RESULTS = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ============================================================
# LAYERS
# ============================================================
class LinearLayer:
    def __init__(self, input_dim, output_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
    def forward(self, x):
        self.x = x
        self.out = x @ self.W + self.b
        return self.out
    def backward(self, grad, lr, l2=1e-4):
        dW = (self.x.T @ grad) / grad.shape[0] + l2 * self.W
        db = np.mean(grad, axis=0)
        dx = grad @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx
    def params(self):
        return self.W.size + self.b.size

class MultiStateLayer:
    def __init__(self, input_dim, output_dim, n_states=4, seed=42):
        self.n_states = n_states
        self.odim = output_dim
        rng = np.random.RandomState(seed)
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]
        self.sw = np.ones(n_states) / n_states
    def forward(self, x):
        self.x = x
        self.states = [x @ w + b for w, b in zip(self.W, self.b)]
        self.out = sum(self.sw[i] * s for i, s in enumerate(self.states))
        return self.out
    def backward(self, grad, lr, l2=1e-4):
        B = grad.shape[0]
        dx = np.zeros((B, self.x.shape[1]))
        for i in range(self.n_states):
            g = grad * self.sw[i]
            dW = (self.x.T @ g) / B + l2 * self.W[i]
            db = np.mean(g, axis=0)
            dx += g @ self.W[i].T
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
            self.sw[i] -= lr * np.sum(grad * self.states[i]) / B
        self.sw = np.maximum(self.sw, 0)
        s = np.sum(self.sw)
        if s > 0:
            self.sw /= s
        return dx
    def params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b) + self.sw.size
    def get_states(self, x):
        return [x @ w + b for w, b in zip(self.W, self.b)]

# ============================================================
# MODELS
# ============================================================
class MLPTradicional:
    def __init__(self, input_dim, h, output_dim, seed=42):
        self.l1 = LinearLayer(input_dim, h, seed)
        self.l2 = LinearLayer(h, h, seed + 1)
        self.l3 = LinearLayer(h, output_dim, seed + 2)
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
    def copy(self):
        import copy
        return copy.deepcopy(self)

class MLPMultiEstado:
    def __init__(self, input_dim, h, output_dim, n_states=4, seed=42):
        self.l1 = MultiStateLayer(input_dim, h, n_states, seed)
        self.l2 = MultiStateLayer(h, h, n_states, seed + 10)
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
    def log_states(self, x):
        s1 = self.l1.get_states(x)
        s2 = self.l2.get_states(x)
        return s1, s2
    def copy(self):
        import copy
        return copy.deepcopy(self)

def get_ms_hidden(target_params, input_dim, output_dim, n_states):
    best = 1
    best_diff = float('inf')
    for h in range(1, 512):
        p = (n_states * (input_dim * h + h + 1)
             + n_states * (h * h + h + 1)
             + h * output_dim + output_dim)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best = h
    return best

def softmax_crossentropy(logits, y):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    n = y.shape[0]
    loss = -np.sum(np.log(probs[np.arange(n), y] + 1e-10)) / n
    grad = probs.copy()
    grad[np.arange(n), y] -= 1
    return loss, grad

def train(model, X, y, Xv, yv, epochs=200, bs=64, lr=0.01, log_epochs=False):
    n = X.shape[0]
    t0 = time.time()
    best = 0.0
    history = []
    for ep in range(epochs):
        lr_ = lr * (0.99 ** ep)
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            e = min(s + bs, n)
            ids = idx[s:e]
            Xb, yb = X[ids], y[ids]
            logits = model.forward(Xb)
            _, grad = softmax_crossentropy(logits, yb)
            model.backward(grad, lr_)
        acc = np.mean(model.predict(Xv) == yv)
        if acc > best:
            best = acc
        if log_epochs:
            history.append(acc)
    return best, time.time() - t0, history

# ============================================================
# DATA GENERATORS
# ============================================================
def make_xor(seed=42):
    rng = np.random.RandomState(seed)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([0,1,1,0], dtype=np.int64)
    # Repeat for more samples
    X = np.tile(X, (125, 1))
    y = np.tile(y, 125)
    return X, y

def make_parity(n_bits=8, n_samples=2000, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_bits)).astype(np.float32)
    y = (np.sum(X, axis=1) % 2).astype(np.int64)
    return X, y

def make_circles(n_samples=2000, noise=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    t = rng.uniform(0, 2*np.pi, n)
    r = rng.uniform(0, 1, n) * 0.5
    X1 = np.column_stack([r*np.cos(t), r*np.sin(t)])
    r2 = rng.uniform(0, 1, n) * 1.0 + 0.3
    X2 = np.column_stack([r2*np.cos(t), r2*np.sin(t)])
    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n, dtype=np.int64)
    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def make_spirals(n_samples=2000, noise=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    t = np.linspace(0, 4*np.pi, n)
    X1 = np.column_stack([t*np.cos(t)/8, t*np.sin(t)/8]) + rng.randn(n,2)*noise
    t2 = np.linspace(0, 4*np.pi, n) + np.pi
    X2 = np.column_stack([t2*np.cos(t2)/8, t2*np.sin(t2)/8]) + rng.randn(n,2)*noise
    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def make_moons(n_samples=2000, noise=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    t = np.linspace(0, np.pi, n)
    X1 = np.column_stack([np.cos(t), np.sin(t)]) + rng.randn(n,2)*noise
    X2 = np.column_stack([1-np.cos(t), -np.sin(t)]) + rng.randn(n,2)*noise + np.array([0, -0.3])
    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def add_noise(X, noise_level, rng):
    return X + rng.randn(*X.shape) * noise_level

# ============================================================
# UTILITY
# ============================================================
def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0)
        std = X.std(0) + 1e-8
    return (X - mean) / std, mean, std

def train_test_split(X, y, ratio=0.2):
    n = len(X)
    nv = int(n * ratio)
    return X[nv:], X[:nv], y[nv:], y[:nv]

def fmt_pct(v):
    return f"{v*100:.2f}%"

# ============================================================
# TEST 1: XOR
# ============================================================
def test_xor():
    print("\n" + "=" * 65)
    print("TESTE 1: XOR")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_xor(seed)
        X, mean, std = normalize(X)
        # Small net: only need tiny hidden dim for XOR
        h_trad = 8
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=200)
        am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=200)
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm})
    print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'Params Trad':>10} | {'Params MS':>9}")
    print("-" * 55)
    for r in rows:
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_params']:>10d} | {r['ms_params']:>9d}")
    ta = [r['trad_acc'] for r in rows]
    ma = [r['ms_acc'] for r in rows]
    print(f"\n  Trad: media={np.mean(ta):.2f}% | Multi: media={np.mean(ma):.2f}%")
    print(f"  Multi venceu: {sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])}/{N_SEEDS}")
    RESULTS["xor"] = rows
    return rows

# ============================================================
# TEST 2: PARITY
# ============================================================
def test_parity():
    print("\n" + "=" * 65)
    print("TESTE 2: PARIDADE (8 bits)")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_parity(n_bits=8, n_samples=2000, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 64
        m_t = MLPTradicional(8, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 8, 2, N_STATES)
        m_m = MLPMultiEstado(8, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm})
    print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'Params Trad':>10} | {'Params MS':>9}")
    print("-" * 55)
    for r in rows:
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_params']:>10d} | {r['ms_params']:>9d}")
    ta = [r['trad_acc'] for r in rows]
    ma = [r['ms_acc'] for r in rows]
    print(f"\n  Trad: media={np.mean(ta):.2f}% | Multi: media={np.mean(ma):.2f}%")
    RESULTS["parity"] = rows
    return rows

# ============================================================
# TEST 3: NON-LINEAR BOUNDARIES
# ============================================================
def test_nonlinear():
    print("\n" + "=" * 65)
    print("TESTE 3: FRONTEIRAS NAO LINEARES")
    print("=" * 65)
    datasets = {
        "circles": make_circles,
        "spirals": make_spirals,
        "moons": make_moons,
    }
    results = {}
    for dname, dfn in datasets.items():
        print(f"\n--- {dname.upper()} ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            X, y = dfn(n_samples=2000, noise=0.15, seed=seed)
            X, mean, std = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            h_trad = 64
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
            m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
            pm = m_m.params()
            at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                          "trad_params": pt, "ms_params": pm})
        ta = [r['trad_acc'] for r in rows]
        ma = [r['ms_acc'] for r in rows]
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | Diff")
        print("-" * 40)
        for r in rows:
            d = r['trad_acc'] - r['ms_acc']
            print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {d:>+6.2f}%")
        print(f"  Trad: {np.mean(ta):.2f}% | Multi: {np.mean(ma):.2f}% | MS wins: {mw}/{N_SEEDS}")
        results[dname] = rows
    RESULTS["nonlinear"] = results
    return results

# ============================================================
# TEST 4: COMPRESSION (Trad 100k vs MS 50k)
# ============================================================
def test_compression():
    print("\n" + "=" * 65)
    print("TESTE 4: COMPRESSAO (Trad ~100k vs MS ~50k)")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        # Traditional with ~100k params: find h
        h_trad = 200  # gives ~100k
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        # Multi with ~50k params
        target = pt // 2
        h_ms = get_ms_hidden(target, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        eff = (am / at) * 100 if at > 0 else 0
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm, "efficiency": round(eff,2)})
    print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'P.Trad':>7} | {'P.MS':>7} | {'Eff':>6}")
    print("-" * 55)
    for r in rows:
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_params']:>7d} | {r['ms_params']:>7d} | {r['efficiency']:>5.1f}%")
    ta = [r['trad_acc'] for r in rows]
    ma = [r['ms_acc'] for r in rows]
    ratio = (np.mean(ma) / np.mean(ta)) * 100 if np.mean(ta) > 0 else 0
    print(f"\n  Trad media: {np.mean(ta):.2f}% | MS media: {np.mean(ma):.2f}% | Eficiencia relativa: {ratio:.1f}%")
    RESULTS["compression"] = rows
    return rows

# ============================================================
# TEST 5: CAPACITY (Trad 100k vs MS 100k)
# ============================================================
def test_capacity():
    print("\n" + "=" * 65)
    print("TESTE 5: CAPACIDADE (Trad 100k vs MS 100k)")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.05, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 200
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm})
    ta = [r['trad_acc'] for r in rows]
    ma = [r['ms_acc'] for r in rows]
    print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'P.Trad':>7} | {'P.MS':>7} | Diff")
    print("-" * 55)
    for r in rows:
        d = r['trad_acc'] - r['ms_acc']
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_params']:>7d} | {r['ms_params']:>7d} | {d:>+6.2f}%")
    mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
    print(f"\n  Trad media: {np.mean(ta):.2f}% | MS media: {np.mean(ma):.2f}% | MS wins: {mw}/{N_SEEDS}")
    RESULTS["capacity"] = rows
    return rows

# ============================================================
# TEST 6: FEW DATA
# ============================================================
def test_few_data():
    print("\n" + "=" * 65)
    print("TESTE 6: POUCOS DADOS")
    print("=" * 65)
    fractions = [1.0, 0.5, 0.25, 0.1]
    results = {}
    for frac in fractions:
        print(f"\n--- {int(frac*100)}% dos dados ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
            X, mean, std = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            # Subsample training
            n_sub = max(10, int(len(Xtr) * frac))
            idx = np.random.choice(len(Xtr), n_sub, replace=False)
            Xtr_s = Xtr[idx]
            ytr_s = ytr[idx]
            h_trad = 64
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
            m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
            at, tt, _ = train(m_t, Xtr_s, ytr_s, Xva, yva, epochs=EPOCHS)
            am, tm, _ = train(m_m, Xtr_s, ytr_s, Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2)})
        ta = [r['trad_acc'] for r in rows]
        ma = [r['ms_acc'] for r in rows]
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"  Trad: {np.mean(ta):.2f}% | Multi: {np.mean(ma):.2f}% | MS wins: {mw}/{N_SEEDS}")
        results[f"{int(frac*100)}pct"] = rows
    RESULTS["few_data"] = results
    return results

# ============================================================
# TEST 7: NOISE
# ============================================================
def test_noise():
    print("\n" + "=" * 65)
    print("TESTE 7: RUIDO")
    print("=" * 65)
    noise_levels = [0.0, 0.1, 0.2, 0.4]
    results = {}
    for nl in noise_levels:
        print(f"\n--- Ruido: {nl*100:.0f}% ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            rng = np.random.RandomState(seed + 1000)
            X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
            X = add_noise(X, nl, rng)
            X, mean, std = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            h_trad = 64
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
            m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
            at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2)})
        ta = [r['trad_acc'] for r in rows]
        ma = [r['ms_acc'] for r in rows]
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"  Trad: {np.mean(ta):.2f}% | Multi: {np.mean(ma):.2f}% | MS wins: {mw}/{N_SEEDS}")
        results[f"noise_{nl}"] = rows
    RESULTS["noise"] = results
    return results

# ============================================================
# TEST 8: STATE SPECIALIZATION (logging)
# ============================================================
def test_state_specialization():
    print("\n" + "=" * 65)
    print("TESTE 8: ESPECIALIZACAO DOS ESTADOS")
    print("=" * 65)
    rows = []
    for seed in range(1, min(N_SEEDS + 1, 6)):  # 5 seeds for detailed logging
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 64
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, tm, hist = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS, log_epochs=True)

        # Log state weights after training
        sw1 = m_m.l1.sw.copy()
        sw2 = m_m.l2.sw.copy()
        rows.append({"seed": seed, "ms_acc": round(am*100,2), "state_weights_l1": sw1.tolist(),
                      "state_weights_l2": sw2.tolist()})
        print(f"  Seed {seed}: MS acc={am*100:.2f}%")
        print(f"    Layer1 state_weights: {np.array2string(sw1, precision=4)}")
        print(f"    Layer2 state_weights: {np.array2string(sw2, precision=4)}")
    RESULTS["specialization"] = rows
    return rows

# ============================================================
# TEST 9: ABLATION (n_states variation)
# ============================================================
def test_ablation():
    print("\n" + "=" * 65)
    print("TESTE 9: ABLACAO (1, 2, 4, 8, 16 estados)")
    print("=" * 65)
    state_counts = [1, 2, 4, 8, 16]
    results = {}
    for ns in state_counts:
        print(f"\n--- {ns} estado(s) ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
            X, mean, std = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            # Fix hidden to match params of trad with h=64
            h_trad = 64
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, ns)
            # Override n_states in model
            m_m = MLPMultiEstado(2, h_ms, 2, ns, seed)
            am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "ms_acc": round(am*100,2), "ms_params": m_m.params()})
        accs = [r['ms_acc'] for r in rows]
        print(f"  Media: {np.mean(accs):.2f}% | Std: {np.std(accs):.2f}% | Min: {min(accs):.2f}% | Max: {max(accs):.2f}%")
        results[ns] = rows
    RESULTS["ablation"] = results
    # Summary table
    print("\n--- TABELA DE ABLACAO ---")
    print(f"{'Estados':>7} | {'Media Acc':>9} | {'Std Dev':>7} | {'Params':>8}")
    print("-" * 40)
    for ns in state_counts:
        accs = [r['ms_acc'] for r in results[ns]]
        p = results[ns][0]['ms_params']
        print(f"{ns:>7d} | {np.mean(accs):>8.2f}% | {np.std(accs):>6.2f}% | {p:>8d}")
    return results

# ============================================================
# TEST 10: INFO PER PARAMETER
# ============================================================
def test_info_per_param():
    print("\n" + "=" * 65)
    print("TESTE 10: INFORMACAO POR PARAMETRO")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 64
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, tt, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        info_t = at / pt * 1e6
        info_m = am / pm * 1e6
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm,
                      "trad_info_per_1M": round(info_t, 4), "ms_info_per_1M": round(info_m, 4),
                      "ratio": round(info_m / info_t, 4) if info_t > 0 else 0})
    print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'Info Trad':>9} | {'Info MS':>9} | {'Ratio':>6}")
    print("-" * 55)
    for r in rows:
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_info_per_1M']:>8.2f} | {r['ms_info_per_1M']:>8.2f} | {r['ratio']:>5.2f}x")
    ta = [r['trad_acc'] for r in rows]
    ma = [r['ms_acc'] for r in rows]
    ri = [r['ratio'] for r in rows]
    print(f"\n  Trad media: {np.mean(ta):.2f}% | MS media: {np.mean(ma):.2f}%")
    print(f"  Eficiencia (Info/param) media do MS vs Trad: {np.mean(ri):.2f}x")
    RESULTS["info_per_param"] = rows
    return rows

# ============================================================
# TEST 11: DISTILLATION
# ============================================================
def test_distillation():
    print("\n" + "=" * 65)
    print("TESTE 11: DESTILACAO")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)

        # Teacher: bigger model
        h_teacher = 128
        teacher = MLPTradicional(2, h_teacher, 2, seed)
        train(teacher, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        # Get soft labels from teacher
        logits = teacher.forward(Xtr)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        soft_labels = exp / np.sum(exp, axis=1, keepdims=True)
        soft_labels = soft_labels.astype(np.float32)

        # Student: traditional
        h_student = 32
        student_t = MLPTradicional(2, h_student, 2, seed + 100)
        # Train with distillation (use soft labels as targets via argmax + weights)
        # Simplified: train on hard labels from teacher
        y_teacher = np.argmax(soft_labels, axis=1)
        # Also train on ground truth for comparison
        at_distill, _, _ = train(student_t, Xtr, y_teacher, Xva, yva, epochs=EPOCHS)
        pt_s = student_t.params()

        # Student: multi-state
        h_ms_s = get_ms_hidden(pt_s, 2, 2, N_STATES)
        student_ms = MLPMultiEstado(2, h_ms_s, 2, N_STATES, seed + 200)
        am_distill, _, _ = train(student_ms, Xtr, y_teacher, Xva, yva, epochs=EPOCHS)
        pm_s = student_ms.params()

        # Also train both on ground truth for comparison
        student_t2 = MLPTradicional(2, h_student, 2, seed + 300)
        at_gt, _, _ = train(student_t2, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        student_ms2 = MLPMultiEstado(2, h_ms_s, 2, N_STATES, seed + 400)
        am_gt, _, _ = train(student_ms2, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        rows.append({"seed": seed,
                      "trad_distill": round(at_distill*100,2), "ms_distill": round(am_distill*100,2),
                      "trad_gt": round(at_gt*100,2), "ms_gt": round(am_gt*100,2),
                      "trad_params": pt_s, "ms_params": pm_s})

        if seed <= 5 or seed == N_SEEDS:
            print(f"  Seed {seed:2d}: Trad(D={at_distill*100:.1f}%,GT={at_gt*100:.1f}%) | MS(D={am_distill*100:.1f}%,GT={am_gt*100:.1f}%)")

    td = [r['trad_distill'] for r in rows]
    md = [r['ms_distill'] for r in rows]
    tg = [r['trad_gt'] for r in rows]
    mg = [r['ms_gt'] for r in rows]
    print(f"\n  Destilacao - Trad: {np.mean(td):.2f}% | MS: {np.mean(md):.2f}%")
    print(f"  Ground Truth - Trad: {np.mean(tg):.2f}% | MS: {np.mean(mg):.2f}%")
    print(f"  Absorcao (Destilacao - GT): Trad={np.mean(td)-np.mean(tg):+.2f}% | MS={np.mean(md)-np.mean(mg):+.2f}%")
    RESULTS["distillation"] = rows
    return rows

# ============================================================
# TEST 12: STATE CORRELATION
# ============================================================
def test_state_correlation():
    print("\n" + "=" * 65)
    print("TESTE 12: CORRELACAO ENTRE ESTADOS")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 64
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        am, tm, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        # Get states for a validation batch
        s1, s2 = m_m.log_states(Xva[:200])

        # Compute correlations between states (within each layer)
        def avg_corr(states):
            n = len(states)
            corrs = []
            for i in range(n):
                for j in range(i+1, n):
                    si = states[i].ravel()
                    sj = states[j].ravel()
                    c = np.corrcoef(si, sj)[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
            return np.mean(corrs) if corrs else 0, corrs

        c1_mean, c1_all = avg_corr(s1)
        c2_mean, c2_all = avg_corr(s2)

        # Also check state weight divergence
        sw_div1 = np.std(m_m.l1.sw)
        sw_div2 = np.std(m_m.l2.sw)

        rows.append({"seed": seed, "ms_acc": round(am*100,2),
                      "corr_l1_mean": round(c1_mean, 4), "corr_l2_mean": round(c2_mean, 4),
                      "corr_l1_all": [round(c, 4) for c in c1_all],
                      "corr_l2_all": [round(c, 4) for c in c2_all],
                      "sw_std_l1": round(sw_div1, 4), "sw_std_l2": round(sw_div2, 4)})

        if seed <= 5 or seed == N_SEEDS:
            print(f"  Seed {seed:2d}: MS={am*100:.1f}% | L1 corr={c1_mean:.4f} L2 corr={c2_mean:.4f} | L1 sw std={sw_div1:.4f} L2 sw std={sw_div2:.4f}")

    c1 = [r['corr_l1_mean'] for r in rows]
    c2 = [r['corr_l2_mean'] for r in rows]
    print(f"\n  Correlacao media entre estados:")
    print(f"    Layer 1: {np.mean(c1):.4f} (std={np.std(c1):.4f})")
    print(f"    Layer 2: {np.mean(c2):.4f} (std={np.std(c2):.4f})")
    if np.mean(c1) > 0.8:
        print("  >>> ALERTA: Estados da Layer 1 sao quase copias (corr > 0.8)")
    if np.mean(c2) > 0.8:
        print("  >>> ALERTA: Estados da Layer 2 sao quase copias (corr > 0.8)")
    RESULTS["correlation"] = rows
    return rows

# ============================================================
# RUN ALL
# ============================================================
def run_all():
    print("=" * 65)
    print("BATERIA COMPLETA DE TESTES - MLP TRADICIONAL vs MULTIESTADO")
    print(f"  Seeds: 1-{N_SEEDS}, Estados: {N_STATES}, Epocas: {EPOCHS}")
    print("=" * 65)

    test_xor()
    test_parity()
    test_nonlinear()
    test_compression()
    test_capacity()
    test_few_data()
    test_noise()
    test_state_specialization()
    test_ablation()
    test_info_per_param()
    test_distillation()
    test_state_correlation()

    # Save all
    with open("bateria_resultados.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\n" + "=" * 65)
    print("Todos os resultados salvos em bateria_resultados.json")
    print("=" * 65)

    # Final verdict
    print("\n--- RESUMO FINAL ---")
    for key in ["xor", "parity", "compression", "capacity", "info_per_param"]:
        if key in RESULTS:
            data = RESULTS[key]
            if isinstance(data, list) and len(data) > 0:
                ta = np.mean([r['trad_acc'] for r in data]) if 'trad_acc' in data[0] else 0
                ma = np.mean([r['ms_acc'] for r in data]) if 'ms_acc' in data[0] else 0
                print(f"  {key:20s}: Trad={ta:.2f}% | MS={ma:.2f}%")

if __name__ == "__main__":
    run_all()
