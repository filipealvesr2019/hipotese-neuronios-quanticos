"""
TESTE 4: COMPRESSAO (Trad ~100k vs MS ~50k)
O MultiEstado mantem desempenho com metade dos parametros?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 4: COMPRESSAO (Trad 100k vs MS 50k)")
    print("=" * 60)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 200  # ~100k params
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt // 2, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, _, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, _, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        eficiencia = (am / at * 100) if at > 0 else 0
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm,
                      "eficiencia": round(eficiencia,2)})
    print_table(rows, extra_cols=[{"name":"Eficiencia","key":"eficiencia","width":10}])
    ta = np.mean([r['trad_acc'] for r in rows])
    ma = np.mean([r['ms_acc'] for r in rows])
    print(f"\n  Trad com {rows[0]['trad_params']} params: {ta:.2f}%")
    print(f"  MS  com {rows[0]['ms_params']} params: {ma:.2f}%")
    print(f"  Proporcao parametros: {rows[0]['ms_params']/rows[0]['trad_params']*100:.1f}%")
    print(f"  Eficiencia (MS/Trad): {ma/ta*100:.1f}%")
    save_results("teste4_compressao", rows)
    return rows

if __name__ == "__main__":
    run()
