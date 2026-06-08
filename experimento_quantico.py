import numpy as np
import random
import time
import json

# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ============================================================
def make_dataset(n_samples=2000, n_features=20, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) * 0.5
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        score = (np.sum(X[i, :5]) + 0.5 * np.prod(X[i, :3])
                 + 0.3 * np.sin(X[i, 0] * X[i, 1]) + rng.randn() * noise)
        y[i] = 1 if score > 0 else 0
    return X, y

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

# ============================================================
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

# ============================================================
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

# ============================================================
def softmax_crossentropy(logits, y):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    n = y.shape[0]
    loss = -np.sum(np.log(probs[np.arange(n), y] + 1e-10)) / n
    grad = probs.copy()
    grad[np.arange(n), y] -= 1
    return loss, grad

def train(model, X, y, Xv, yv, epochs=100, bs=64, lr=0.01):
    n = X.shape[0]
    t0 = time.time()
    best = 0.0
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
    return best, time.time() - t0

# ============================================================
def compute_ms_hidden(target_params, input_dim, output_dim, n_states):
    """Find hidden dim for MultiState MLP to match target_params."""
    best_h = 1
    for h in range(1, 512):
        p = (n_states * (input_dim * h + h + 1)
             + n_states * (h * h + h + 1)
             + h * output_dim + output_dim)
        if abs(p - target_params) < abs(best_h * target_params - target_params * 0):
            # Just find closest
            pass

    # Solve analytically: 4h^2 + (4 + 4 + 2*output_dim + 2*input_dim*n_states)h + ...
    # Actually let's just brute-force iterate to find closest
    best = None
    best_diff = float('inf')
    for h in range(1, 512):
        ms = MLPMultiEstado(input_dim, h, output_dim, n_states, 0)
        p = ms.params()
        if abs(p - target_params) < best_diff:
            best_diff = abs(p - target_params)
            best = h
    return best

# ============================================================
def run_experiment(seed, input_dim=20, output_dim=2, epochs=100, n_states=4):
    set_seed(seed)
    X, y = make_dataset(n_samples=2000, n_features=input_dim, seed=seed)
    mean, std = X.mean(0), X.std(0) + 1e-8
    X = (X - mean) / std

    nv = int(len(X) * 0.2)
    Xtr, Xva = X[nv:], X[:nv]
    ytr, yva = y[nv:], y[:nv]

    # --- Traditional (hidden=128) ---
    base_h = 128
    model_t = MLPTradicional(input_dim, base_h, output_dim, seed)
    pt = model_t.params()
    at, tt = train(model_t, Xtr, ytr, Xva, yva, epochs)

    # --- Multi-State with matched params ---
    ms_h = compute_ms_hidden(pt, input_dim, output_dim, n_states)
    model_m = MLPMultiEstado(input_dim, ms_h, output_dim, n_states, seed)
    pm = model_m.params()
    am, tm = train(model_m, Xtr, ytr, Xva, yva, epochs)

    return {
        "seed": seed,
        "trad_acc": round(at * 100, 2),
        "trad_params": pt,
        "trad_time": round(tt, 2),
        "ms_acc": round(am * 100, 2),
        "ms_params": pm,
        "ms_time": round(tm, 2),
        "ms_hidden": ms_h,
    }

def main():
    n_seeds = 30
    results = []
    print(f"Running {n_seeds} deterministic experiments (params matched)...\n")
    h = f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'Trad Params':>10} | {'MS Params':>9} | {'Trad Time':>9} | {'MS Time':>9}"
    print(h)
    print("-" * len(h))

    for seed in range(1, n_seeds + 1):
        r = run_experiment(seed)
        results.append(r)
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | "
              f"{r['trad_params']:>10d} | {r['ms_params']:>9d} | "
              f"{r['trad_time']:>8.2f}s | {r['ms_time']:>8.2f}s")

    ta = [r["trad_acc"] for r in results]
    ma = [r["ms_acc"] for r in results]
    mw = sum(1 for r in results if r["ms_acc"] > r["trad_acc"])
    ti = sum(1 for r in results if r["ms_acc"] == r["trad_acc"])
    tw = n_seeds - mw - ti

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"Tradicional  - Média: {np.mean(ta):.2f}%, Std: {np.std(ta):.2f}%, "
          f"Min: {min(ta):.2f}%, Max: {max(ta):.2f}%")
    print(f"MultiEstado  - Média: {np.mean(ma):.2f}%, Std: {np.std(ma):.2f}%, "
          f"Min: {min(ma):.2f}%, Max: {max(ma):.2f}%")
    print(f"\nMultiEstado venceu em {mw}/{n_seeds} ({mw/n_seeds*100:.1f}%)")
    print(f"Tradicional venceu em {tw}/{n_seeds} ({tw/n_seeds*100:.1f}%)")
    print(f"Empates: {ti}/{n_seeds}")
    print(f"\nHidden: Trad=128, MS=~{results[0]['ms_hidden']} (for param matching)")

    with open("resultados.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em resultados.json")

if __name__ == "__main__":
    main()
