"""
TESTE 10: INFORMACAO POR PARAMETRO
Accuracy / Numero de parametros
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 10: INFORMACAO POR PARAMETRO")
    print("=" * 60)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 32
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        pm = m_m.params()
        at, _, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, _, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        info_t = at / pt
        info_m = am / pm
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": pm,
                      "trad_info": round(info_t*1e4,4), "ms_info": round(info_m*1e4,4),
                      "ratio": round(info_m/info_t,4) if info_t>0 else 0})
    print_table(rows, extra_cols=[
        {"name":"Info Trad","key":"trad_info","width":10},
        {"name":"Info MS","key":"ms_info","width":9},
        {"name":"Ratio","key":"ratio","width":6},
    ])
    ratios = [r['ratio'] for r in rows]
    print(f"\n  Eficiencia media (MS/Trad): {np.mean(ratios):.2f}x")
    if np.mean(ratios) > 1.0:
        print("  >>> MultiEstado e mais eficiente (mais acc por parametro)")
    else:
        print("  >>> Tradicional e mais eficiente (mais acc por parametro)")
    save_results("teste10_info_por_param", rows)
    return rows

if __name__ == "__main__":
    run()
