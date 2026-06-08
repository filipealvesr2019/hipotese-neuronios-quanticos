"""
TESTE 1: XOR
O neuronio multiestado consegue aprender interacoes simples?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 1: XOR")
    print("=" * 60)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_xor(seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 8
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_ms = get_ms_hidden(pt, 2, 2, N_STATES)
        m_m = MLPMultiEstado(2, h_ms, 2, N_STATES, seed)
        at, _, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        am, _, _ = train(m_m, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "ms_acc": round(am*100,2),
                      "trad_params": pt, "ms_params": m_m.params()})
    print_table(rows)
    print_summary(rows, "XOR")
    save_results("teste1_XOR", rows)
    return rows

if __name__ == "__main__":
    run()
