import json, numpy as np, csv

mlp_accs, v4_accs = [], []
mlp_hist, v4_hist = [], []

for seed in range(1, 11):
    with open(f"resultados_finais/e1_mnist_validation/dense_h128_seed{seed}.json") as f:
        d = json.load(f)
    mlp_accs.append(d["dense"]["accuracy"])
    mlp_hist.append(d["dense"]["history"])

    with open(f"resultados_finais/e1_mnist_validation/v4_h128_s2_g8_noskip_seed{seed}.json") as f:
        d = json.load(f)
    v4_accs.append(d["v4_sparse"]["accuracy"])
    v4_hist.append(d["v4_sparse"]["history"])

# Clean L2_H (handle negative values)
def clean_l2(vh):
    gs = vh.get("gate_stats", {})
    l2 = gs.get("layer2", {}).get("normalized_entropy", 0)
    return max(0, l2)

print("=" * 60)
print("VALIDACAO V4 SPARSE ROUTING — 10 SEEDS x 10 EPOCAS")
print("=" * 60)
print(f"\nMLP 128 padrao: media={np.mean(mlp_accs)*100:.2f}%  std={np.std(mlp_accs)*100:.2f}%")
print(f"V4 h128/s2/g8/noskip: media={np.mean(v4_accs)*100:.2f}%  std={np.std(v4_accs)*100:.2f}%")

diffs = [m - v for m, v in zip(mlp_accs, v4_accs)]
print(f"Diff media (MLP - V4): {np.mean(diffs)*100:+.2f}pp")

print(f"\n{'Seed':>4} | {'MLP':>7} | {'V4':>7} | {'Diff':>7} | {'L1_H':>5} | {'L2_H':>5}")
print("-" * 50)
for i in range(10):
    gs = v4_hist[i][-1].get("gate_stats", {})
    l1h = gs.get("layer1", {}).get("normalized_entropy", 0)
    l2h = clean_l2(v4_hist[i][-1])
    print(f"{i+1:4d} | {mlp_accs[i]*100:>6.2f}% | {v4_accs[i]*100:>6.2f}% | {diffs[i]*100:>+6.2f}% | {l1h:.3f} | {l2h:.3f}")

wins_v4 = sum(1 for i in range(10) if v4_accs[i] > mlp_accs[i])
wins_mlp = sum(1 for i in range(10) if mlp_accs[i] > v4_accs[i])
print(f"\nConfronto direto: V4={wins_v4}/10  MLP={wins_mlp}/10  Empates={10-wins_v4-wins_mlp}")

# Save summary
summary = {
    "models": {
        "MLP 128": {
            "acc_mean": float(f"{np.mean(mlp_accs)*100:.2f}"),
            "acc_std": float(f"{np.std(mlp_accs)*100:.2f}"),
            "flops": 236032
        },
        "V4 h128/s2/g8/noskip": {
            "acc_mean": float(f"{np.mean(v4_accs)*100:.2f}"),
            "acc_std": float(f"{np.std(v4_accs)*100:.2f}"),
            "flops": 250688
        }
    },
    "confronto": {"V4": wins_v4, "MLP": wins_mlp},
    "seeds": [{"seed": i+1, "mlp": round(mlp_accs[i]*100, 2), "v4": round(v4_accs[i]*100, 2)} for i in range(10)]
}

with open("resultados_finais/e1_mnist_validation/summary_10seeds.json", "w") as f:
    json.dump(summary, f, indent=2)

# Per-epoch curves (mean over 10 seeds)
print("\n=== CURVAS MEDIAS (10 seeds) ===")
print(f"{'Ep':>3} | {'MLP Acc':>7} | {'V4 Acc':>7} | {'L1_H':>5} | {'L2_H':>5}")
print("-" * 42)
for ep in range(10):
    mlp_ep = np.mean([h[ep]["test_acc"] for h in mlp_hist])
    v4_ep = np.mean([h[ep]["test_acc"] for h in v4_hist])
    l1s = [v4_hist[s][ep].get("gate_stats", {}).get("layer1", {}).get("normalized_entropy", 0) for s in range(10)]
    l2s = [clean_l2(v4_hist[s][ep]) for s in range(10)]
    print(f"{ep+1:3d} | {mlp_ep*100:>6.2f}% | {v4_ep*100:>6.2f}% | {np.mean(l1s):.3f} | {np.mean(l2s):.3f}")

# Entropy summary
l1_final = [v4_hist[s][-1].get("gate_stats", {}).get("layer1", {}).get("normalized_entropy", 0) for s in range(10)]
l2_final = [clean_l2(v4_hist[s][-1]) for s in range(10)]
print(f"\n=== ENTROPIA FINAL (media 10 seeds) ===")
print(f"L1_H: media={np.mean(l1_final):.3f}  std={np.std(l1_final):.3f}")
print(f"L2_H: media={np.mean(l2_final):.3f}  std={np.std(l2_final):.3f}")
print(f"L2_H range: [{min(l2_final):.3f}, {max(l2_final):.3f}]")

# Group by entropy regime
print(f"\n=== REGIMES DE ENTROPIA ===")
for i in range(10):
    l2 = l2_final[i]
    regime = "COLAPSO" if l2 < 0.05 else ("MODERADO" if l2 < 0.5 else "ALTO")
    print(f"  seed {i+1}: L2_H={l2:.3f} [{regime}]")

print("\nOK — sumario salvo em summary_10seeds.json")
