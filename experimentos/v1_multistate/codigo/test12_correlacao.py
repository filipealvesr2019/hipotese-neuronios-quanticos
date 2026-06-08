"""
TESTE 12: CORRELACAO ENTRE ESTADOS
Os estados internos sao independentes ou copias?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 12: CORRELACAO ENTRE ESTADOS")
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
        am, _, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        # Compute state correlations
        s1, s2 = m_m.log_states(Xva[:100])

        def avg_corr(states):
            n = len(states)
            corrs = []
            for i in range(n):
                for j in range(i+1, n):
                    si = states[i].ravel()
                    sj = states[j].ravel()
                    c = np.corrcoef(si, sj)[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
            return np.mean(corrs) if corrs else 0, corrs

        c1_m, c1_all = avg_corr(s1)
        c2_m, c2_all = avg_corr(s2)
        sw_d1 = np.std(m_m.l1.sw)
        sw_d2 = np.std(m_m.l2.sw)

        rows.append({"seed": seed, "ms_acc": round(am*100,2),
                      "corr_l1": round(c1_m,4), "corr_l2": round(c2_m,4),
                      "corr_l1_pares": [round(c,4) for c in c1_all],
                      "corr_l2_pares": [round(c,4) for c in c2_all],
                      "sw_std_l1": round(sw_d1,4), "sw_std_l2": round(sw_d2,4)})

        c_str1 = ", ".join([f"{c:.3f}" for c in c1_all])
        c_str2 = ", ".join([f"{c:.3f}" for c in c2_all])
        print(f"  Seed {seed}: MS={am*100:.1f}% | L1 corr={c1_m:.4f} [{c_str1}] | L2 corr={c2_m:.4f} [{c_str2}]")

    c1 = [r['corr_l1'] for r in rows]
    c2 = [r['corr_l2'] for r in rows]
    print(f"\n  Correlacao media entre estados:")
    print(f"    Layer 1: {np.mean(c1):.4f} (std={np.std(c1):.4f})")
    print(f"    Layer 2: {np.mean(c2):.4f} (std={np.std(c2):.4f})")
    if np.mean(c1) > 0.7:
        print("  >>> Layer 1: estados ALTAMENTE CORRELACIONADOS (quase copias)")
    elif np.mean(c1) < 0.3:
        print("  <<< Layer 1: estados BAIXA correlacao (aprendendo coisas diferentes!)")
    if np.mean(c2) > 0.7:
        print("  >>> Layer 2: estados ALTAMENTE CORRELACIONADOS (quase copias)")
    elif np.mean(c2) < 0.3:
        print("  <<< Layer 2: estados BAIXA correlacao (aprendendo coisas diferentes!)")
    save_results("teste12_correlacao", rows)
    return rows

if __name__ == "__main__":
    run()
