"""
TESTE 8: ESPECIALIZACAO DOS ESTADOS
Log de state_weights: os estados aprendem coisas diferentes?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 8: ESPECIALIZACAO DOS ESTADOS")
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
        sw1 = m_m.l1.sw.copy()
        sw2 = m_m.l2.sw.copy()
        div1 = np.std(sw1)
        div2 = np.std(sw2)
        rows.append({"seed": seed, "ms_acc": round(am*100,2),
                      "sw_l1": sw1.tolist(), "sw_l2": sw2.tolist(),
                      "sw_std_l1": round(div1,4), "sw_std_l2": round(div2,4)})
        print(f"  Seed {seed}: MS={am*100:.2f}% | L1 sw={np.array2string(sw1,precision=3)} (std={div1:.4f}) | L2 sw={np.array2string(sw2,precision=3)} (std={div2:.4f})")
    # Overall analysis
    stds1 = [r['sw_std_l1'] for r in rows]
    stds2 = [r['sw_std_l2'] for r in rows]
    print(f"\n  Divergencia media dos state_weights:")
    print(f"    Layer 1: std medio = {np.mean(stds1):.4f}")
    print(f"    Layer 2: std medio = {np.mean(stds2):.4f}")
    if np.mean(stds1) < 0.05:
        print("  >>> Layer 1: estados quase uniformes (pesos iguais)")
    if np.mean(stds2) < 0.05:
        print("  >>> Layer 2: estados quase uniformes (pesos iguais)")
    if np.mean(stds1) > 0.2:
        print("  <<< Layer 1: estados com pesos bem diferenciados!")
    if np.mean(stds2) > 0.2:
        print("  <<< Layer 2: estados com pesos bem diferenciados!")
    save_results("teste8_especializacao", rows)
    return rows

if __name__ == "__main__":
    run()
