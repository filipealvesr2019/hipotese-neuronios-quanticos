"""
Módulo comum: layers, modelos, datasets, treino, utilitários.
"""
import numpy as np
import random
import time
import json
import os

# ============================================================
# CONFIG (sobrescrito pelos testes)
# ============================================================
N_SEEDS = 5
N_STATES = 4
EPOCHS = 100
RESULT_DIR = "F:\\neuronios quanticos\\resultados"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)

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
        h = np.maximum(self.l1.forward(x), 0)
        s2 = self.l2.get_states(h)
        return s1, s2

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
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([0,1,1,0], dtype=np.int64)
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
    X1 = np.column_stack([rng.uniform(0,0.5,n)*np.cos(t), rng.uniform(0,0.5,n)*np.sin(t)])
    X2 = np.column_stack([(rng.uniform(0.3,1.3,n))*np.cos(t), (rng.uniform(0.3,1.3,n))*np.sin(t)])
    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n, dtype=np.int64)
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
    X2 = np.column_stack([1-np.cos(t), -np.sin(t)-0.3]) + rng.randn(n,2)*noise
    X = np.vstack([X1, X2])
    y = np.array([0]*n + [1]*n, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def add_noise(X, noise_level, rng):
    return X + rng.randn(*X.shape) * noise_level

def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0)
        std = X.std(0) + 1e-8
    return (X - mean) / std, mean, std

def train_test_split(X, y, ratio=0.2):
    n = len(X)
    nv = int(n * ratio)
    return X[nv:], X[:nv], y[nv:], y[:nv]

def save_results(name, data):
    ensure_dir()
    path = os.path.join(RESULT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Resultados salvos em {path}")

def print_table(rows, extra_cols=None):
    cols = ["seed", "trad_acc", "ms_acc", "trad_params", "ms_params"]
    header = f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'P.Trad':>7} | {'P.MS':>7}"
    if extra_cols:
        for c in extra_cols:
            header += f" | {c['name']:>{c['width']}}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_params']:>7d} | {r['ms_params']:>7d}"
        if extra_cols:
            for c in extra_cols:
                val = r[c['key']]
                if isinstance(val, float):
                    line += f" | {val:>{c['width']}.2f}"
                else:
                    line += f" | {val:>{c['width']}}"
        print(line)

def print_summary(rows, label="Resultado"):
    ta = [r['trad_acc'] for r in rows]
    ma = [r['ms_acc'] for r in rows]
    mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
    tw = sum(1 for r in rows if r['ms_acc'] < r['trad_acc'])
    print(f"\n  {label}: Trad={np.mean(ta):.2f}% | MS={np.mean(ma):.2f}% | MS venceu: {mw}/{len(rows)} | Trad venceu: {tw}/{len(rows)}")
