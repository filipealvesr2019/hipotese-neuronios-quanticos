"""
TESTE 3: FRONTEIRAS NAO LINEARES (circulos, espirais, luas)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 3: FRONTEIRAS NAO LINEARES")
    print("=" * 60)
    datasets = {
        "circulos": (make_circles, 0.15),
        "espirais": (make_spirals, 0.2),
        "luas":     (make_moons, 0.1),
    }
    all_results = {}
    for dname, (dfn, noise) in datasets.items():
        print(f"\n--- {dname.upper()} ---")
        rows = []
        for seed in range(1, N_SEEDS + 1):
            set_seed(seed)
            X, y = dfn(n_samples=1000, noise=noise, seed=seed)
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
        print_table(rows)
        print_summary(rows, dname)
        all_results[dname] = rows
    save_results("teste3_fronteiras", all_results)
    return all_results

if __name__ == "__main__":
    run()
