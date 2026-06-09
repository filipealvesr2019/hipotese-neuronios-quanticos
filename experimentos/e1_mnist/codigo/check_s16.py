import json

with open("resultados_finais/e1_mnist_validation/v4_s16_g8_noskip_seed1.json") as f:
    d = json.load(f)
v4 = d["v4_sparse"]

acc = v4["accuracy"]
gs = v4["history"][-1].get("gate_stats", {})
l1h = gs.get("layer1", {}).get("normalized_entropy", 0)
l2h = gs.get("layer2", {}).get("normalized_entropy", 0)

print(f"Acc: {acc*100:.2f}%")
print(f"L1_H: {l1h:.3f}")
print(f"L2_H: {l2h:.3f}")
print(f"Params: {v4['params']}")
print(f"FLOPs sparse: {v4['estimated_sparse_inference_flops_per_sample']}")
print(f"Treino: {v4['train_time_sec']:.1f}s")

print()
print("Historico:")
for h in v4["history"]:
    g = h.get("gate_stats", {})
    l1 = g.get("layer1", {}).get("normalized_entropy", 0)
    l2 = g.get("layer2", {}).get("normalized_entropy", 0)
    print(f"  ep {h['epoch']:2d}: loss={h['loss']:.4f} acc={h['test_acc']*100:.2f}% L1={l1:.3f} L2={l2:.3f}")

# Usage by class
print()
print("Uso L1 por classe:")
u1 = v4["gate_usage_l1_by_class"]
for cls in range(10):
    total = sum(u1[cls])
    pct = [round(x/total*100) for x in u1[cls]]
    dom = max(range(len(u1[cls])), key=lambda i: u1[cls][i])
    print(f"  classe {cls}: expert {dom} ({pct[dom]:.0f}%)  dist={pct}")

print()
print("Uso L2 por classe:")
u2 = v4["gate_usage_l2_by_class"]
for cls in range(10):
    total = sum(u2[cls])
    pct = [round(x/total*100) for x in u2[cls]]
    dom = max(range(len(u2[cls])), key=lambda i: u2[cls][i])
    print(f"  classe {cls}: expert {dom} ({pct[dom]:.0f}%)  dist={pct}")
