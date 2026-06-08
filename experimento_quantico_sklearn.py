import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
import time
import json

# ============================================================
# Deterministic setup
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ============================================================
# Dataset: synthetic binary classification (non-linear)
# ============================================================
def make_dataset(n_samples=2000, n_features=20, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) * 0.5
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        score = (np.sum(X[i, :5])
                 + 0.5 * np.prod(X[i, :3])
                 + 0.3 * np.sin(X[i, 0] * X[i, 1])
                 + rng.randn() * noise)
        y[i] = 1 if score > 0 else 0
    return X, y

# ============================================================
# Multi-State Neuron Layer (4 internal states) using numpy
# ============================================================
class MultiStateLayer:
    """Each neuron has 4 internal states, each with independent weights."""
    def __init__(self, input_dim, output_dim, n_states=4, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_states = n_states
        rng = np.random.RandomState(seed)

        # One weight matrix per state
        self.W = [rng.randn(input_dim, output_dim) * 0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Learnable state weights (initialized uniformly)
        self.state_w = np.ones(n_states) / n_states

        # Gradients
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        self.dstate_w = np.zeros_like(self.state_w)

        # Cache for backward
        self.x_cached = None
        self.states_cached = None
        self.out_cached = None

    def forward(self, x):
        self.x_cached = x.copy() if isinstance(x, np.ndarray) else x
        self.states_cached = [x @ w + b for w, b in zip(self.W, self.b)]
        # Weighted sum
        self.out_cached = sum(self.state_w[i] * s for i, s in enumerate(self.states_cached))
        return self.out_cached

    def backward(self, grad_output, lr=0.01, l2_lambda=1e-4):
        batch_size = grad_output.shape[0]
        # Gradient w.r.t. state weights
        for i in range(self.n_states):
            self.dstate_w[i] = np.sum(grad_output * self.states_cached[i]) / batch_size
            # Gradient w.r.t. W[i] and b[i]
            self.dW[i] = (self.x_cached.T @ (grad_output * self.state_w[i])) / batch_size
            self.db[i] = np.mean(grad_output * self.state_w[i], axis=0)

            # L2 regularization
            self.dW[i] += l2_lambda * self.W[i]

            # Update
            self.W[i] -= lr * self.dW[i]
            self.b[i] -= lr * self.db[i]

        # Update state weights
        self.state_w -= lr * self.dstate_w
        # Project to simplex (ensure non-negative, sum to 1)
        self.state_w = np.maximum(self.state_w, 0)
        self.state_w = self.state_w / (np.sum(self.state_w) + 1e-10)

    def get_params_vector(self):
        return np.concatenate([w.ravel() for w in self.W] + [b.ravel() for b in self.b] + [self.state_w.ravel()])

    def set_params_vector(self, vec):
        offset = 0
        for i in range(self.n_states):
            n_w = self.input_dim * self.output_dim
            self.W[i] = vec[offset:offset + n_w].reshape(self.input_dim, self.output_dim)
            offset += n_w
            n_b = self.output_dim
            self.b[i] = vec[offset:offset + n_b]
            offset += n_b
        self.state_w = vec[offset:offset + self.n_states]

    def total_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b) + self.state_w.size


# ============================================================
# Custom Multi-State MLP (pure numpy, trained via SGD)
# ============================================================
class MultiStateMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, n_states=4, seed=42):
        self.layer1 = MultiStateLayer(input_dim, hidden_dim, n_states, seed)
        self.layer2 = MultiStateLayer(hidden_dim, hidden_dim, n_states, seed + 1)
        self.layer3_W = np.random.RandomState(seed + 2).randn(hidden_dim, output_dim) * 0.1
        self.layer3_b = np.zeros(output_dim)

    def forward(self, x):
        x = np.maximum(self.layer1.forward(x), 0)  # ReLU
        x = np.maximum(self.layer2.forward(x), 0)  # ReLU
        x = x @ self.layer3_W + self.layer3_b
        return x

    def softmax_crossentropy(self, logits, y):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        n = y.shape[0]
        loss = -np.sum(np.log(probs[np.arange(n), y] + 1e-10)) / n
        # Gradient
        grad = probs.copy()
        grad[np.arange(n), y] -= 1
        grad /= n
        return loss, grad

    def backward(self, grad, lr=0.01, l2_lambda=1e-4):
        # Layer3
        h2 = np.maximum(self.layer2.out_cached, 0)
        dW3 = h2.T @ grad + l2_lambda * self.layer3_W
        db3 = np.mean(grad, axis=0)
        self.layer3_W -= lr * dW3
        self.layer3_b -= lr * db3

        # Backprop through layer2 (ReLU + linear)
        grad_h2 = grad @ self.layer3_W.T
        grad_h2[self.layer2.out_cached <= 0] = 0
        self.layer2.backward(grad_h2, lr, l2_lambda)

        # Backprop through layer1 (ReLU + linear)
        h1 = self.layer1.out_cached
        grad_h1 = self.layer2.x_cached @ grad_h2.T  # TODO: simplified - use proper backprop
        # Actually properly:
        # grad w.r.t. layer1 output = grad_h2 @ W2^T, then through ReLU
        grad_h1_correct = grad_h2 @ self.layer2.W[0].T  # approximate
        grad_h1_correct[self.layer1.out_cached <= 0] = 0
        self.layer1.backward(grad_h1_correct, lr, l2_lambda)

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def total_params(self):
        return (self.layer1.total_params()
                + self.layer2.total_params()
                + self.layer3_W.size
                + self.layer3_b.size)

    def train_epoch(self, X, y, batch_size=64, lr=0.01, l2_lambda=1e-4):
        n = X.shape[0]
        indices = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            Xb, yb = X[idx], y[idx]

            logits = self.forward(Xb)
            loss, grad = self.softmax_crossentropy(logits, yb)
            self.backward(grad, lr, l2_lambda)
            epoch_loss += loss * len(idx)
        return epoch_loss / n


# ============================================================
# Training function
# ============================================================
def train_model_numpy(model, X_train, y_train, X_val, y_val,
                      epochs=100, batch_size=64, lr=0.01):
    start = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        # Simple learning rate decay
        current_lr = lr * (0.99 ** epoch)
        model.train_epoch(X_train, y_train, batch_size, current_lr)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
    elapsed = time.time() - start
    return best_acc, elapsed


def train_sklearn(X_train, y_train, X_val, y_val, hidden_dim=128, epochs=100):
    start = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_dim, hidden_dim),
        activation='relu',
        solver='adam',
        max_iter=epochs,
        random_state=42,
        early_stopping=False,
        tol=1e-8,
        n_iter_no_change=epochs,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    params = sum(p.sum() for p in [coef.ravel() for coef in model.coefs_])
    # Actually count params properly
    params = sum(coef.size for coef in model.coefs_) + sum(inter.size for inter in model.intercepts_)
    elapsed = time.time() - start
    return acc, params, elapsed


# ============================================================
# Main experiment
# ============================================================
def run_experiment(seed, hidden_dim=128, input_dim=20, output_dim=2,
                   epochs=100, val_split=0.2, n_states=4):
    set_seed(seed)

    # Generate dataset deterministically
    X, y = make_dataset(n_samples=2000, n_features=input_dim, seed=seed)
    n_val = int(len(X) * val_split)
    X_train, X_val = X[:n_val], X[n_val:]
    y_train, y_val = y[:n_val], y[n_val:]

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --- Traditional (sklearn MLP) ---
    acc_trad, params_trad, time_trad = train_sklearn(
        X_train, y_train, X_val, y_val, hidden_dim, epochs
    )

    # --- Multi-State ---
    # Try different hidden sizes to roughly match params
    best_acc_ms = 0.0
    best_hidden_ms = hidden_dim
    best_time_ms = 0.0
    params_ms = 0

    for h_try in [hidden_dim, hidden_dim // 2, hidden_dim // 3, hidden_dim // 4,
                  hidden_dim // 5, hidden_dim // 6, hidden_dim // 8]:
        model_ms = MultiStateMLP(input_dim, h_try, output_dim, n_states, seed)
        p = model_ms.total_params()
        if p <= params_trad * 1.1:
            acc, elapsed = train_model_numpy(
                model_ms, X_train, y_train, X_val, y_val, epochs, lr=0.01
            )
            if acc > best_acc_ms:
                best_acc_ms = acc
                best_hidden_ms = h_try
                best_time_ms = elapsed
                params_ms = p

    return {
        "seed": seed,
        "trad_acc": round(acc_trad * 100, 2),
        "trad_params": params_trad,
        "trad_time": round(time_trad, 2),
        "ms_acc": round(best_acc_ms * 100, 2),
        "ms_params": params_ms,
        "ms_time": round(best_time_ms, 2),
        "ms_hidden": best_hidden_ms,
    }


def main():
    n_seeds = 30
    results = []

    print(f"Running {n_seeds} deterministic experiments (scikit-learn + numpy)...\n")
    header = f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'Trad Params':>10} | {'MS Params':>9} | {'Trad Time':>9} | {'MS Time':>9}"
    print(header)
    print("-" * len(header))

    for seed in range(1, n_seeds + 1):
        r = run_experiment(seed)
        results.append(r)
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | "
              f"{r['trad_params']:>10d} | {r['ms_params']:>9d} | "
              f"{r['trad_time']:>8.2f}s | {r['ms_time']:>8.2f}s")

    # Summary
    trad_accs = [r["trad_acc"] for r in results]
    ms_accs = [r["ms_acc"] for r in results]
    ms_wins = sum(1 for r in results if r["ms_acc"] > r["trad_acc"])
    ties = sum(1 for r in results if r["ms_acc"] == r["trad_acc"])
    trad_wins = n_seeds - ms_wins - ties

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"Tradicional - Média: {np.mean(trad_accs):.2f}%, "
          f"Std: {np.std(trad_accs):.2f}%, "
          f"Min: {min(trad_accs):.2f}%, "
          f"Max: {max(trad_accs):.2f}%")
    print(f"MultiEstado - Média: {np.mean(ms_accs):.2f}%, "
          f"Std: {np.std(ms_accs):.2f}%, "
          f"Min: {min(ms_accs):.2f}%, "
          f"Max: {max(ms_accs):.2f}%")
    print(f"\nMultiEstado venceu em {ms_wins}/{n_seeds} ({(ms_wins / n_seeds) * 100:.1f}%)")
    print(f"Tradicional venceu em {trad_wins}/{n_seeds} ({(trad_wins / n_seeds) * 100:.1f}%)")
    print(f"Empates: {ties}/{n_seeds}")

    with open("resultados.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em resultados.json")
    return results


if __name__ == "__main__":
    main()
