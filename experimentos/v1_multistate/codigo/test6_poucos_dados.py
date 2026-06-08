"""
TESTE 6: POUCOS DADOS (100%, 50%, 25%, 10%)
O MultiEstado generaliza melhor com poucos dados?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 6: POUCOS DADOS")
    print("=" * 60)
    fractions = [1.0, 0.5, 0.25, 0.1]
    all_results = {}
    for frac in fractions:
        label = f"{int(frac*100)}%"
        print(f"\n--- {label} dos dados ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
            X, _, _ = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            n_sub = max(10, int(len(Xtr) * frac))
            idx = np.random.choice(len(Xtr), n_sub, replace=False)
            h_trad = 32
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
            m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
            at, _, _ = train(m_t, Xtr[idx], ytr[idx], Xva, yva, epochs=EPOCHS)
            am, _, _ = train(m_m, Xtr[idx], ytr[idx], Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                          "trad_params": pt, "ms_params": m_m.params()})
        ta = np.mean([r['trad_acc'] for r in rows])
        ma = np.mean([r['ms_acc'] for r in rows])
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"  Trad: {ta:.2f}% | MS: {ma:.2f}% | MS venceu: {mw}/{N_SEEDS}")
        all_results[label] = rows
    # Summary table
    print(f"\n{'Fracao':>6} | {'Trad Media':>9} | {'MS Media':>8} | {'MS Vence':>8}")
    print("-" * 40)
    for frac in fractions:
        label = f"{int(frac*100)}%"
        rows = all_results[label]
        ta = np.mean([r['trad_acc'] for r in rows])
        ma = np.mean([r['ms_acc'] for r in rows])
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"{label:>6} | {ta:>8.2f}% | {ma:>7.2f}% | {mw}/{N_SEEDS}")
    save_results("teste6_poucos_dados", all_results)
    return all_results

if __name__ == "__main__":
    run()
