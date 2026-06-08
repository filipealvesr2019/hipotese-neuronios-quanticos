import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bateria_completa import MLPTradicional, make_circles, normalize, train_test_split, set_seed, softmax_crossentropy
from v4_sparse_routing import MLPMultiEstadoV4, get_ms_v4_hidden

def run_30_seeds():
    print("==================================================")
    print(" VALIDAÇÃO ESTATÍSTICA: V4 SPARSE vs TRADICIONAL")
    print(" 30 Seeds - Dataset: Círculos (2000 amostras, noise=0.1)")
    print("==================================================")
    
    n_seeds = 30
    results_trad = []
    results_v4 = []
    
    # Same hyperparams
    epochs = 300
    lr = 0.01
    
    print(f"{'Seed':>5} | {'Trad Acc':>10} | {'V4 Acc':>10} | {'Diff':>8}")
    print("-" * 43)

    for seed in range(1, n_seeds + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=2000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        n = Xtr.shape[0]

        # Tradicional
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

        # V4 Sparse
        pt = mt.params()
        h_v4 = get_ms_v4_hidden(pt, 2, 2, 4, gate_hidden=16)
        mv4 = MLPMultiEstadoV4(2, h_v4, 2, 4, seed=seed, temperature=1.0, gate_hidden=16)
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

        diff = av4 - at
        print(f"{seed:>5} | {at:>9.2f}% | {av4:>9.2f}% | {diff:>+7.2f}%")

    # Estatísticas Finais
    print("\n--- RESUMO ESTATÍSTICO (30 SEEDS) ---")
    trad_mean, trad_std = np.mean(results_trad), np.std(results_trad)
    v4_mean, v4_std = np.mean(results_v4), np.std(results_v4)
    
    print(f"Tradicional : {trad_mean:.2f}% ± {trad_std:.2f}%")
    print(f"V4 Sparse   : {v4_mean:.2f}% ± {v4_std:.2f}%")
    
    wins_v4 = sum(1 for a, b in zip(results_v4, results_trad) if a > b)
    wins_trad = sum(1 for a, b in zip(results_v4, results_trad) if b > a)
    ties = n_seeds - wins_v4 - wins_trad
    
    print(f"\nVitórias V4 : {wins_v4}/{n_seeds}")
    print(f"Vitórias Trad: {wins_trad}/{n_seeds}")
    print(f"Empates     : {ties}/{n_seeds}")

if __name__ == "__main__":
    run_30_seeds()
