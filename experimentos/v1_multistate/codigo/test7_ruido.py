"""
TESTE 7: RUIDO (0%, 10%, 20%, 40%)
Qual modelo e mais robusto a ruido?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 7: RUIDO")
    print("=" * 60)
    noise_levels = [0.0, 0.1, 0.2, 0.4]
    all_results = {}
    for nl in noise_levels:
        label = f"ruido_{int(nl*100)}%"
        print(f"\n--- Ruido: {int(nl*100)}% ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            rng = np.random.RandomState(seed + 1000)
            X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
            X = add_noise(X, nl, rng)
            X, _, _ = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            h_trad = 32
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
            m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
            at, _, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            am, _, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                          "trad_params": pt, "ms_params": m_m.params()})
        ta = np.mean([r['trad_acc'] for r in rows])
        ma = np.mean([r['ms_acc'] for r in rows])
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"  Trad: {ta:.2f}% | MS: {ma:.2f}% | MS venceu: {mw}/{N_SEEDS}")
        all_results[label] = rows
    print(f"\n{'Ruido':>6} | {'Trad Media':>9} | {'MS Media':>8} | {'MS Vence':>8}")
    print("-" * 40)
    for nl in noise_levels:
        label = f"{int(nl*100)}%"
        rows = all_results[f"ruido_{int(nl*100)}%"]
        ta = np.mean([r['trad_acc'] for r in rows])
        ma = np.mean([r['ms_acc'] for r in rows])
        mw = sum(1 for r in rows if r['ms_acc'] > r['trad_acc'])
        print(f"{label:>6} | {ta:>8.2f}% | {ma:>7.2f}% | {mw}/{N_SEEDS}")
    save_results("teste7_ruido", all_results)
    return all_results

if __name__ == "__main__":
    run()
