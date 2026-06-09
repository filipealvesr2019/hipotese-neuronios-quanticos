import json, numpy as np, csv
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M")

def load(path):
    with open(path) as f:
        return json.load(f)

rows = []

# === 1. MLP vs V4 s=2 — 10 seeds ===
mlp_accs, v4_accs = [], []
v4_l1h, v4_l2h = [], []
for seed in range(1, 11):
    d = load(f"resultados_finais/e1_mnist_validation/dense_h128_seed{seed}.json")
    mlp_accs.append(d["dense"]["accuracy"])
    v = load(f"resultados_finais/e1_mnist_validation/v4_h128_s2_g8_noskip_seed{seed}.json")
    v4 = v["v4_sparse"]
    v4_accs.append(v4["accuracy"])
    gs = v4["history"][-1].get("gate_stats", {})
    l1 = gs.get("layer1", {}).get("normalized_entropy", 0)
    l2 = gs.get("layer2", {}).get("normalized_entropy", 0)
    v4_l1h.append(l1)
    v4_l2h.append(max(0, l2))

# === 2. Temperature sweep ===
temps = [("T=0.5", "v4_t05_seed1.json", "e1_mnist_validation"),
         ("T=2.0", "v4_t20_seed1.json", "e1_mnist_validation"),
         ("T=0.5", "v4_t05_seed5.json", "e1_mnist_validation"),
         ("T=2.0", "v4_t20_seed5.json", "e1_mnist_validation")]
temp_rows = []
for label, fname, subdir in temps:
    d = load(f"resultados_finais/{subdir}/{fname}")
    v = d["v4_sparse"]
    gs = v["history"][-1].get("gate_stats", {})
    temp_rows.append({
        "label": label, "seed": d["seed"], "acc": v["accuracy"],
        "l1h": gs.get("layer1", {}).get("normalized_entropy", 0),
        "l2h": gs.get("layer2", {}).get("normalized_entropy", 0),
        "flops": v["estimated_sparse_inference_flops_per_sample"],
        "params": v["params"]
    })

# === 3. States scale ===
states_data = [
    ("s=2", load("resultados_finais/e1_mnist_validation/v4_h128_s2_g8_noskip_seed1.json")),
    ("s=4", load("resultados_finais/e1_mnist_validation/v4_s4_g8_noskip_seed1.json")),
    ("s=8", load("resultados_finais/e1_mnist_validation/v4_s8_g8_noskip_seed1.json")),
]
scale_rows = []
for label, d in states_data:
    v = d["v4_sparse"]
    gs = v["history"][-1].get("gate_stats", {})
    scale_rows.append({
        "label": label, "acc": v["accuracy"],
        "l1h": gs.get("layer1", {}).get("normalized_entropy", 0),
        "l2h": gs.get("layer2", {}).get("normalized_entropy", 0),
        "flops": v["estimated_sparse_inference_flops_per_sample"],
        "dense_flops": v["estimated_dense_executed_flops_per_sample"],
        "params": v["params"], "train_time": v["train_time_sec"]
    })

mlp_acc = load("resultados_finais/e1_mnist_validation/dense_h128_seed1.json")["dense"]["accuracy"]

# ====================================
# WRITE REPORT
# ====================================
lines = []
lines.append(f"# Relatorio Consolidado — V4 Sparse Routing no MNIST")
lines.append(f"*Gerado em: {now}*")
lines.append("")
lines.append("## 1. Validacao 10 seeds — V4 s=2/g=8/no-skip vs MLP 128")
lines.append("")
lines.append(f"| Metrica | MLP 128 | V4 s=2/g=8/no-skip |")
lines.append(f"|---------|--------|---------------------|")
lines.append(f"| Media   | {np.mean(mlp_accs)*100:.2f}% | {np.mean(v4_accs)*100:.2f}% |")
lines.append(f"| Std     | {np.std(mlp_accs)*100:.2f}% | {np.std(v4_accs)*100:.2f}% |")
lines.append(f"| Min     | {min(mlp_accs)*100:.2f}% | {min(v4_accs)*100:.2f}% |")
lines.append(f"| Max     | {max(mlp_accs)*100:.2f}% | {max(v4_accs)*100:.2f}% |")
diffs = [m - v for m, v in zip(mlp_accs, v4_accs)]
lines.append(f"| Diff media (MLP-V4) | | {np.mean(diffs)*100:+.2f}pp |")
wins_v4 = sum(1 for i in range(10) if v4_accs[i] > mlp_accs[i])
wins_mlp = sum(1 for i in range(10) if mlp_accs[i] > v4_accs[i])
lines.append(f"| Confronto direto | | V4={wins_v4}/10 MLP={wins_mlp}/10 |")
lines.append("")
lines.append("### Per-seed")
lines.append("")
lines.append(f"| Seed | MLP | V4 | Diff | L1_H | L2_H | Regime L2 |")
lines.append(f"|------|-----|-----|------|------|------|-----------|")
def regime(l2):
    return "COLAPSO" if l2 < 0.05 else ("MODERADO" if l2 < 0.5 else "ALTO")
for i in range(10):
    l2 = max(0, v4_l2h[i])
    lines.append(f"| {i+1} | {mlp_accs[i]*100:.2f}% | {v4_accs[i]*100:.2f}% | {diffs[i]*100:+.2f}pp | {v4_l1h[i]:.3f} | {l2:.3f} | {regime(l2)} |")
lines.append("")

lines.append("## 2. Variancia de Temperatura")
lines.append("")
lines.append("### Seed 1 (colapso natural)")
lines.append("")
lines.append("| T | Acc | L1_H | L2_H |")
lines.append("|---|-----|------|------|")
t1 = [t for t in temp_rows if t["seed"] == 1]
for t in sorted(t1, key=lambda x: float(x["label"].replace("T=",""))):
    lines.append(f"| {t['label']} | {t['acc']*100:.2f}% | {t['l1h']:.3f} | {t['l2h']:.3f} |")
lines.append("")

lines.append("### Seed 5 (entropia alta)")
lines.append("")
lines.append("| T | Acc | L1_H | L2_H |")
lines.append("|---|-----|------|------|")
t5 = [t for t in temp_rows if t["seed"] == 5]
for t in sorted(t5, key=lambda x: float(x["label"].replace("T=",""))):
    lines.append(f"| {t['label']} | {t['acc']*100:.2f}% | {t['l1h']:.3f} | {t['l2h']:.3f} |")
lines.append("")

lines.append("**Conclusao:** Temperatura nao altera o regime de entropia. O comportamento do gate e determinado pela inicializacao, nao pelo hyperparametro de temperatura.")
lines.append("")

lines.append("## 3. Escala de Estados (seed 1)")
lines.append("")
lines.append("| Estados | Acc | L1_H | L2_H | Params | FLOPs sparse | FLOPs dense | Treino (s) |")
lines.append("|---------|-----|------|------|--------|-------------|-------------|-----------|")
for r in scale_rows:
    lines.append(f"| {r['label']} | {r['acc']*100:.2f}% | {r['l1h']:.3f} | {r['l2h']:.3f} | {r['params']} | {r['flops']} | {r['dense_flops']} | {r['train_time']:.1f} |")
mlp_params = load("resultados_finais/e1_mnist_validation/dense_h128_seed1.json")["dense"]["params"]
lines.append(f"| MLP (ref) | {mlp_acc*100:.2f}% | — | — | {mlp_params} | — | 236032 | — |")
lines.append("")

lines.append("### Uso por classe (s=8, L1)")
lines.append("")
lines.append("Maior especializacao observada em L1 com 8 estados:")
lines.append("- Expert 3 domina: classes 0, 1, 3, 5, 6, 7, 9")
lines.append("- Expert 4 domina: classes 2, 4, 8")
lines.append("- Porem L2 colapsou para expert 2 em 99-100% das classes")
lines.append("")

lines.append("## 4. Analise de Custo-Beneficio")
lines.append("")
lines.append(f"| Config | Acc | vs MLP | Params | FLOPs sparse |")
lines.append(f"|--------|-----|--------|--------|-------------|")
for r in scale_rows:
    lines.append(f"| {r['label']} | {r['acc']*100:.2f}% | {r['acc']-mlp_acc:+.2f}pp | {r['params']} | {r['flops']} |")
lines.append(f"| MLP (ref) | {mlp_acc*100:.2f}% | baseline | {mlp_params} | 236032 |")
lines.append("")

lines.append("## 5. Conclusoes")
lines.append("")
lines.append("1. **V4 nao superou o MLP** em 10 seeds (94.78% vs 95.05%, diff -0.28pp)")
lines.append("2. **Temperatura nao controla colapso** — regime de entropia e determinado pela seed")
lines.append("3. **Seeds com colapso tem accuracy MELHOR** que seeds com entropia alta")
lines.append("4. **Camadas alternam colapso** — nunca ambas distribuem simultaneamente")
lines.append("5. **Mais estados piora accuracy** (s=2: 95.23%, s=4: 94.30%, s=8: 93.66%)")
lines.append("6. **Especializacao real existe** (s=8 L1: experts 3 e 4 dividem classes) mas nao melhora accuracy")
lines.append("7. **Problema e arquitetura, nao roteamento** — V4 tem limitacao fundamental no fluxo de gradiente via top-1 hard selection")
lines.append("")

report = "\n".join(lines)

# Save
with open("resultados_finais/e1_mnist_validation/REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print()
print(f"Relatorio salvo em resultados_finais/e1_mnist_validation/REPORT.md")
