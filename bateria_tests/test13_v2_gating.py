"""
Teste 13A: State dominance — o gate seleciona estados diferentes
para entradas diferentes?
Teste 13B: Ablation — remover estados e medir impacto.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from v2_gating import *

N_SEEDS = 5
EPOCHS = 300
TEMPERATURE = 0.3  # <1 peaked gate, =1 standard, >1 uniform

def test_v2_baseline():
    """Compara V2 vs Trad em circulos com parametros equivalentes."""
    print("\n" + "=" * 65)
    print("BASELINE: MultiEstado V2 vs Tradicional (círculos)")
    print("=" * 65)
    rows = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
        X, _, _ = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 32
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_v2 = get_ms_v2_hidden(pt, 2, 2, N_STATES)
        m_v2 = MLPMultiEstadoV2(2, h_v2, 2, N_STATES, seed, TEMPERATURE)
        pv2 = m_v2.params()
        at, _, _ = train(m_t, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        av2, _, _ = train(m_v2, Xtr, ytr, Xva, yva, epochs=EPOCHS)
        rows.append({"seed": seed, "trad_acc": round(at*100,2), "v2_acc": round(av2*100,2),
                      "trad_params": pt, "v2_params": pv2, "v2_hidden": h_v2})
    print(f"{'Seed':>4} | {'Trad Acc':>8} | {'V2 Acc':>8} | {'P.Trad':>7} | {'P.V2':>7} | {'V2 h':>5}")
    print("-" * 55)
    for r in rows:
        print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['v2_acc']:>7.2f}% | {r['trad_params']:>7d} | {r['v2_params']:>7d} | {r['v2_hidden']:>5d}")
    ta = [r['trad_acc'] for r in rows]
    va = [r['v2_acc'] for r in rows]
    print(f"\n  Trad: {np.mean(ta):.2f}% | V2: {np.mean(va):.2f}% | V2 venceu: {sum(1 for r in rows if r['v2_acc'] > r['trad_acc'])}/{N_SEEDS}")
    return rows

def test_13a_gate_analysis():
    """
    Teste 13A: Para 1000 amostras, registrar qual estado domina o gate.
    O gate está selecionando estados diferentes para entradas diferentes?
    """
    print("\n" + "=" * 65)
    print("TESTE 13A: ANALISE DE DOMINANCIA DO GATE")
    print("=" * 65)
    all_data = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 32
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_v2 = get_ms_v2_hidden(pt, 2, 2, N_STATES)
        m_v2 = MLPMultiEstadoV2(2, h_v2, 2, N_STATES, seed, TEMPERATURE)
        av2, _, _ = train(m_v2, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        # Analisar gate em todas as amostras de treino
        g1, g2 = m_v2.analyze_gate(Xtr)

        # Estado dominante (argmax) para cada amostra
        dom_l1 = np.argmax(g1, axis=1)
        dom_l2 = np.argmax(g2, axis=1)

        # Distribuicao de dominancia
        dist_l1 = np.bincount(dom_l1, minlength=N_STATES) / len(dom_l1) * 100
        dist_l2 = np.bincount(dom_l2, minlength=N_STATES) / len(dom_l2) * 100

        # Entropia media do gate (medida de quao "decidido" o gate esta)
        eps = 1e-10
        ent_l1 = -np.mean(np.sum(g1 * np.log(g1 + eps), axis=1))
        ent_l2 = -np.mean(np.sum(g2 * np.log(g2 + eps), axis=1))
        max_ent = np.log(N_STATES)  # entropia maxima (distribuicao uniforme)

        # Quao longe da uniforme?
        kl_l1 = max_ent - ent_l1
        kl_l2 = max_ent - ent_l2

        # Media e std dos gate weights
        mean_g1 = np.mean(g1, axis=0)
        std_g1 = np.std(g1, axis=0)

        info = {
            "seed": seed, "v2_acc": round(av2*100,2),
            "dominance_l1": dist_l1.tolist(), "dominance_l2": dist_l2.tolist(),
            "entropy_l1": round(ent_l1,4), "entropy_l2": round(ent_l2,4),
            "max_entropy": round(max_ent,4),
            "kl_div_l1": round(kl_l1,4), "kl_div_l2": round(kl_l2,4),
            "gate_mean_l1": [round(v,4) for v in mean_g1],
            "gate_std_l1": [round(v,4) for v in std_g1],
        }
        all_data.append(info)

        # Print detalhado
        print(f"\n  Seed {seed}: V2 acc = {av2*100:.2f}%")
        print(f"  Gate L1 - Dominancia: E1={dist_l1[0]:.1f}% E2={dist_l1[1]:.1f}% E3={dist_l1[2]:.1f}% E4={dist_l1[3]:.1f}%")
        print(f"  Gate L2 - Dominancia: E1={dist_l2[0]:.1f}% E2={dist_l2[1]:.1f}% E3={dist_l2[2]:.1f}% E4={dist_l2[3]:.1f}%")
        print(f"  Entropia L1: {ent_l1:.4f} (max={max_ent:.4f}, KL={kl_l1:.4f}) | L2: {ent_l2:.4f} (KL={kl_l2:.4f})")
        print(f"  Gate mean L1: {np.array2string(mean_g1, precision=3)}")
        print(f"  Gate std  L1: {np.array2string(std_g1, precision=3)}")

        # Mostrar alguns exemplos
        print(f"  Exemplos de gate L1 (10 amostras):")
        print(f"  {np.array2string(g1[:10], precision=3, suppress_small=True)}")

    # Summary
    print(f"\n--- SUMARIO TESTE 13A ---")
    kl1 = [d['kl_div_l1'] for d in all_data]
    kl2 = [d['kl_div_l2'] for d in all_data]
    print(f"  KL divergencia media L1: {np.mean(kl1):.4f} (0 = uniforme, >0 = especializado)")
    print(f"  KL divergencia media L2: {np.mean(kl2):.4f}")
    if np.mean(kl1) < 0.1:
        print("  >>> Gate L1 colapsou para distribuicao uniforme")
    else:
        print("  <<< Gate L1 esta especializando!")
    if np.mean(kl2) < 0.1:
        print("  >>> Gate L2 colapsou para distribuicao uniforme")
    else:
        print("  <<< Gate L2 esta especializando!")
    save_results("teste13a_gate_analysis", all_data)
    return all_data

def test_13b_ablation():
    """
    Teste 13B: Remover cada estado pos-treino e medir impacto na acuracia.
    Se cada remocao causa dano diferente -> estados especializados.
    """
    print("\n" + "=" * 65)
    print("TESTE 13B: ABLACAO DE ESTADOS (remover cada estado)")
    print("=" * 65)
    all_data = []
    for seed in range(1, N_SEEDS + 1):
        set_seed(seed)
        X, y = make_circles(n_samples=1000, noise=0.1, seed=seed)
        X, mean, std = normalize(X)
        Xtr, Xva, ytr, yva = train_test_split(X, y)
        h_trad = 32
        m_t = MLPTradicional(2, h_trad, 2, seed)
        pt = m_t.params()
        h_v2 = get_ms_v2_hidden(pt, 2, 2, N_STATES)
        m_v2 = MLPMultiEstadoV2(2, h_v2, 2, N_STATES, seed, TEMPERATURE)

        # Treina modelo completo
        acc_full, _, _ = train(m_v2, Xtr, ytr, Xva, yva, epochs=EPOCHS)

        # Acuracia base com modelo completo (sem mais treino)
        preds_full = m_v2.predict(Xva)
        acc_full_val = np.mean(preds_full == yva)

        # Ablacao: remover cada estado de layer1 e layer2
        impacts = {}
        for layer_name, layer in [("L1", m_v2.l1), ("L2", m_v2.l2)]:
            for i_state in range(N_STATES):
                # Salva pesos originais
                orig_W = layer.W[i_state].copy()
                orig_b = layer.b[i_state].copy()

                # Zera o estado (remocao)
                layer.W[i_state] = np.zeros_like(orig_W)
                layer.b[i_state] = np.zeros_like(orig_b)

                # Mede impacto
                preds = m_v2.predict(Xva)
                acc_abl = np.mean(preds == yva)
                impact = (acc_full_val - acc_abl) * 100

                impacts[f"{layer_name}_E{i_state+1}"] = round(impact, 2)

                # Restaura
                layer.W[i_state] = orig_W
                layer.b[i_state] = orig_b

        info = {"seed": seed, "v2_acc": round(acc_full*100,2), "impacts": impacts}
        all_data.append(info)

        print(f"\n  Seed {seed}: V2 acc = {acc_full*100:.2f}%")
        print(f"  Impacto da remocao (queda em pp):")
        for k, v in impacts.items():
            print(f"    {k}: {v:+.2f}pp")
        impacts_list = list(impacts.values())
        print(f"    Std dos impactos: {np.std(impacts_list):.2f} (0=todos iguais, >0=diferenciados)")

    # Summary
    print(f"\n--- SUMARIO TESTE 13B ---")
    keys = [f"L{i//4+1}_E{i%4+1}" for i in range(8)]
    all_impacts = {k: [] for k in [f"L1_E{i+1}" for i in range(4)] + [f"L2_E{i+1}" for i in range(4)]}
    for d in all_data:
        for k in all_impacts:
            all_impacts[k].append(d['impacts'][k])

    print(f"\n{'Estado':>6} | {'Impacto medio':>13} | {'Std':>6}")
    print("-" * 30)
    total_std = []
    for k in all_impacts:
        vals = all_impacts[k]
        print(f"{k:>6} | {np.mean(vals):>+10.2f}pp | {np.std(vals):>5.2f}")
        total_std.extend(vals)
    print(f"\n  Std total dos impactos: {np.std(total_std):.2f}pp")
    if np.std(total_std) > 2.0:
        print("  <<< Estados tem impactos DIFERENTES quando removidos -> especializados!")
    else:
        print("  >>> Estados tem impactos SEMELHANTES quando removidos -> nao especializados")
    save_results("teste13b_ablation", all_data)
    return all_data

if __name__ == "__main__":
    baseline = test_v2_baseline()
    print("\n" + "#" * 65)
    data_a = test_13a_gate_analysis()
    print("\n" + "#" * 65)
    data_b = test_13b_ablation()

    # Final verdict
    print("\n" + "=" * 65)
    print("VEREDITO FINAL - MultiEstado V2 (com gating)")
    print("=" * 65)

    # Teste 13A
    kl1 = np.mean([d['kl_div_l1'] for d in data_a])
    kl2 = np.mean([d['kl_div_l2'] for d in data_a])
    print(f"\n  Gate especializou? KL-L1={kl1:.4f} KL-L2={kl2:.4f}")
    if kl1 > 0.5 or kl2 > 0.5:
        print("  >>> GATE ESPECIALIZOU - entradas diferentes escolhem estados diferentes")
    elif kl1 > 0.1:
        print("  >>> Gate moderadamente especializado")
    else:
        print("  >>> Gate NAO especializou (distribuicao quase uniforme)")

    # Teste 13B
    all_impacts = []
    for d in data_b:
        all_impacts.extend(d['impacts'].values())
    impact_std = np.std(all_impacts)
    print(f"\n  Ablacao - std dos impactos: {impact_std:.2f}pp")
    if impact_std > 2.0:
        print("  >>> ESTADOS ESPECIALIZADOS - remover cada um causa dano diferente")
    else:
        print("  >>> Estados NAO diferenciados - remover qualquer um causa dano similar")

    # Baseline
    va = [r['v2_acc'] for r in baseline]
    ta = [r['trad_acc'] for r in baseline]
    print(f"\n  Performance: Trad={np.mean(ta):.2f}% | V2={np.mean(va):.2f}%")
