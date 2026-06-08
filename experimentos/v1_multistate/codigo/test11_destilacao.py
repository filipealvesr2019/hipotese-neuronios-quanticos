"""
TESTE 11: DESTILACAO
Professor -> Aluno (Trad vs MS): qual absorve mais conhecimento?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.common import *

N_SEEDS = 5
EPOCHS = 100

def run():
    print("\n" + "=" * 60)
    print("TESTE 11: DESTILACAO")
    print("=" * 60)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)

        # Teacher: bigger model
        teacher = MLPTradicional(2, 64, 2, seed)
        train(teacher, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        logits = teacher.forward(Xtr)
        y_teacher = np.argmax(logits, axis=1)

        # Student: traditional (small)
        h_s = 16
        st_t = MLPTradicional(2, h_s, 2, seed + 100)
        pt_s = st_t.params()
        at_dist, _, _ = train(st_t, Xtr, y_teacher, Xva, yva, epochs=EPOCHS)
        # GT
        st_t2 = MLPTradicional(2, h_s, 2, seed + 200)
        at_gt, _, _ = train(st_t2, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        # Student: multi-state
        h_ms_s = get_ms_hidden(pt_s, 2, 2, N_STATES)
        sm = MLPMultiEstado(2, h_ms_s, 2, N_STATES, seed + 300)
        pm_s = sm.params()
        am_dist, _, _ = train(sm, Xtr, y_teacher, Xva, yva, epochs=EPOCHS)
        sm2 = MLPMultiEstado(2, h_ms_s, 2, N_STATES, seed + 400)
        am_gt, _, _ = train(sm2, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        rows.append({"seed": seed,
                      "trad_distill": round(at_dist*100,2), "ms_distill": round(am_dist*100,2),
                      "trad_gt": round(at_gt*100,2), "ms_gt": round(am_gt*100,2),
                      "trad_params": pt_s, "ms_params": pm_s})

        if seed <= 2:
            print(f"  Seed {seed}: Trad(D={at_dist*100:.1f}%,GT={at_gt*100:.1f}%) | MS(D={am_dist*100:.1f}%,GT={am_gt*100:.1f}%)")

    td = [r['trad_distill'] for r in rows]
    md = [r['ms_distill'] for r in rows]
    tg = [r['trad_gt'] for r in rows]
    mg = [r['ms_gt'] for r in rows]
    print(f"\n  Destilado - Trad: {np.mean(td):.2f}% | MS: {np.mean(md):.2f}%")
    print(f"  GT        - Trad: {np.mean(tg):.2f}% | MS: {np.mean(mg):.2f}%")
    print(f"  Ganho destilacao - Trad: {np.mean(td)-np.mean(tg):+.2f}% | MS: {np.mean(md)-np.mean(mg):+.2f}%")
    save_results("teste11_destilacao", rows)
    return rows

if __name__ == "__main__":
    run()
