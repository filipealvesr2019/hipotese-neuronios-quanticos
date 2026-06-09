import json, numpy as np

mlp_accs = []
v4_accs = []
mlp_histories = []
v4_histories = []

for seed in range(1, 6):
    with open(f"resultados_finais/e1_mnist_validation/dense_h128_seed{seed}.json") as f:
        d = json.load(f)
    mlp_accs.append(d["dense"]["accuracy"])
    mlp_histories.append(d["dense"]["history"])

    with open(f"resultados_finais/e1_mnist_validation/v4_h128_s2_g8_noskip_seed{seed}.json") as f:
        d = json.load(f)
    v4_accs.append(d["v4_sparse"]["accuracy"])
    v4_histories.append(d["v4_sparse"]["history"])

print("=== RESULTADOS PARCIAIS (seeds 1-5) ===")
print(f"MLP 128: media={np.mean(mlp_accs)*100:.2f}% std={np.std(mlp_accs)*100:.2f}%")
print(f"V4  h128/s2/g8/noskip: media={np.mean(v4_accs)*100:.2f}% std={np.std(v4_accs)*100:.2f}%")

for i in range(5):
    diff = mlp_accs[i] - v4_accs[i]
    print(f"  seed {i+1}: MLP={mlp_accs[i]*100:.2f}% V4={v4_accs[i]*100:.2f}% diff={diff*100:+.2f}pp")

wins_v4 = sum(1 for i in range(5) if v4_accs[i] > mlp_accs[i])
wins_mlp = sum(1 for i in range(5) if mlp_accs[i] > v4_accs[i])
print(f"V4 venceu: {wins_v4}/5 | MLP venceu: {wins_mlp}/5")

print()
print("=== CURVAS DE TREINO (seed 1) ===")
print(f"{'Ep':>3} | {'MLP Loss':>8} | {'MLP Acc':>7} | {'V4 Loss':>8} | {'V4 Acc':>7} | {'L1_H':>5} | {'L2_H':>5}")
print("-" * 55)
for ep in range(10):
    mh = mlp_histories[0][ep]
    vh = v4_histories[0][ep]
    gs = vh.get("gate_stats", {})
    l1h = gs.get("layer1", {}).get("normalized_entropy", 0)
    l2h = gs.get("layer2", {}).get("normalized_entropy", 0)
    print(f"{ep+1:3d} | {mh['loss']:>8.4f} | {mh['test_acc']*100:>6.2f}% | {vh['loss']:>8.4f} | {vh['test_acc']*100:>6.2f}% | {l1h:.3f} | {l2h:.3f}")

# Show L1/L2 per seed
print()
print("=== ENTROPIA FINAL POR SEED ===")
print(f"{'Seed':>4} | {'V4 Acc':>7} | {'L1_H':>5} | {'L2_H':>5}")
print("-" * 30)
for seed in range(5):
    vh = v4_histories[seed][-1]
    gs = vh.get("gate_stats", {})
    l1h = gs.get("layer1", {}).get("normalized_entropy", 0)
    l2h = gs.get("layer2", {}).get("normalized_entropy", 0)
    print(f"{seed+1:4d} | {v4_accs[seed]*100:>6.2f}% | {l1h:.3f} | {l2h:.3f}")
