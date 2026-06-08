import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bateria_completa import MLPTradicional, make_circles, make_moons, make_spirals, normalize, train_test_split, set_seed, softmax_crossentropy
from v4_sparse_routing import MLPMultiEstadoV4, get_ms_v4_hidden

def make_custom_20features(n_samples=2000, n_features=20, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) * 0.5
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        # A complex combination of features
        score = (np.sum(X[i, :5]) + 0.5 * np.prod(X[i, :3]) 
                 + 0.3 * np.sin(X[i, 0] * X[i, 1]) + rng.randn() * noise)
        y[i] = 1 if score > 0 else 0
    return X, y

def run_stress_test():
    print("==================================================")
    print(" STRESS TEST: V4 SPARSE vs TRADICIONAL (30 SEEDS)")
    print(" Testando Robustez da Descoberta em Múltiplos Domínios")
    print("==================================================")
    
    n_seeds = 30
    epochs = 300
    lr = 0.01

    datasets = {
        "Moons (Ruído=0.1)": lambda s: make_moons(n_samples=2000, noise=0.1, seed=s),
        "Spirals (Ruído=0.2)": lambda s: make_spirals(n_samples=2000, noise=0.2, seed=s),
        "20-Features (Ruído=0.3)": lambda s: make_custom_20features(n_samples=2000, n_features=20, noise=0.3, seed=s)
    }

    for d_name, d_fn in datasets.items():
        print(f"\n--- DATASET: {d_name} ---")
        
        results_trad = []
        results_v4 = []
        
        for seed in range(1, n_seeds + 1):
            set_seed(seed)
            X, y = d_fn(seed)
            X, _, _ = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            n = Xtr.shape[0]
            input_dim = X.shape[1]

            # Tradicional
            mt = MLPTradicional(input_dim, 64, 2, seed)
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

            # V4 Sparse
            pt = mt.params()
            h_v4 = get_ms_v4_hidden(pt, input_dim, 2, 4, gate_hidden=16)
            mv4 = MLPMultiEstadoV4(input_dim, h_v4, 2, 4, seed=seed, temperature=1.0, gate_hidden=16)
            for ep in range(epochs):
                lr_ = lr * (0.99 ** ep)
                idx = np.random.permutation(n)
                for s in range(0, n, 64):
                    e = min(s + 64, n)
                    ids = idx[s:e]
                    Xb, yb = Xtr[ids], ytr[ids]
                    logits = mv4.forward(Xb)
                    _, grad = softmax_crossentropy(logits, yb)
                    mv4.backward(grad, lr_)
            av4 = np.mean(mv4.predict(Xva) == yva) * 100
            results_v4.append(av4)

        t_mean, t_std = np.mean(results_trad), np.std(results_trad)
        v_mean, v_std = np.mean(results_v4), np.std(results_v4)
        wins_v4 = sum(1 for a, b in zip(results_v4, results_trad) if a > b)
        wins_t = sum(1 for a, b in zip(results_v4, results_trad) if b > a)
        
        print(f"  Tradicional : {t_mean:.2f}% ± {t_std:.2f}%")
        print(f"  V4 Sparse   : {v_mean:.2f}% ± {v_std:.2f}%")
        print(f"  V4 Wins {wins_v4}/{n_seeds} | Trad Wins {wins_t}/{n_seeds}")

if __name__ == "__main__":
    run_stress_test()
