"""
TESTE 9: ABLACAO (1, 2, 4, 8, 16 estados)
Mais estados ajudam?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 9: ABLACAO (1, 2, 4, 8, 16 estados)")
    print("=" * 60)
    state_counts = [1, 2, 4, 8, 16]
    all_results = {}
    for ns in state_counts:
        print(f"\n--- {ns} estado(s) ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
            X, _, _ = normalize(X)
            Xtr, Xva, ytr, yva = train_test_split(X, y)
            h_trad = 32
            m_t = MLPTradicional(2, h_trad, 2, seed)
            pt = m_t.params()
            h_ms = get_ms_hidden(pt, 2, 2, ns)
            m_m = MLPMultiEstado(2, h_ms, 2, ns, seed)
            am, _, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
            rows.append({"seed": seed, "ms_acc": round(am*100,2), "ms_params": m_m.params()})
        accs = [r['ms_acc'] for r in rows]
        print(f"  Media: {np.mean(accs):.2f}% | Std: {np.std(accs):.2f}% | Params: {rows[0]['ms_params']}")
        all_results[str(ns)] = rows
    # Summary table
    print(f"\n{'Estados':>7} | {'Media Acc':>9} | {'Std Dev':>7} | {'Params':>8}")
    print("-" * 40)
    for ns in state_counts:
        accs = [r['ms_acc'] for r in all_results[str(ns)]]
        p = all_results[str(ns)][0]['ms_params']
        print(f"{ns:>7d} | {np.mean(accs):>8.2f}% | {np.std(accs):>6.2f}% | {p:>8d}")
    save_results("teste9_ablacao", all_results)
    return all_results

if __name__ == "__main__":
    run()
