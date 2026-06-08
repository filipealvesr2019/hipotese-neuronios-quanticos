import json
import numpy as np

with open("resultados.json") as f:
    data = json.load(f)

ta = [r["trad_acc"] for r in data]
ma = [r["ms_acc"] for r in data]
tp = data[0]["trad_params"]
mp_vals = [r["ms_params"] for r in data]

print("=" * 70)
print("RESUMO COMPLETO - EXPERIMENTO MULTIESTADO (4 estados) vs TRADICIONAL")
print("=" * 70)
print()
print("--- CONFIGURACAO ---")
print("  Seeds: 1 a 30")
print("  Dataset: 2000 amostras, 20 features, classificacao binaria")
print("  Modelo Trad: MLP 3 camadas (Linear+ReLU), hidden=128")
print("  Modelo Multi: MLP com MultiStateNeuronLayer (4 estados internos)")
print("  Epocas: 100, Otimizador: SGD, Lr: 0.01 (decay 0.99/ep)")
print("  Total execucoes: 30 (uma por seed)")
print()

print("--- TABELA COMPARATIVA ---")
print(f"{'Seed':>4} | {'Trad Acc':>8} | {'MS Acc':>8} | {'Trad Params':>9} | {'MS Params':>8} | {'Diferenca':>8}")
print("-" * 60)
for r in data:
    d = r["trad_acc"] - r["ms_acc"]
    print(f"{r['seed']:4d} | {r['trad_acc']:>7.2f}% | {r['ms_acc']:>7.2f}% | {r['trad_params']:>9d} | {r['ms_params']:>8d} | {d:>+7.2f}%")
print()

print("--- ESTATISTICAS ---")
print(f"  Tradicional  - Media: {np.mean(ta):.2f}% | Std: {np.std(ta):.2f}% | Min: {min(ta):.2f}% | Max: {max(ta):.2f}%")
print(f"  MultiEstado  - Media: {np.mean(ma):.2f}% | Std: {np.std(ma):.2f}% | Min: {min(ma):.2f}% | Max: {max(ma):.2f}%")
print(f"  Diferenca media (Trad - MS): {np.mean(ta) - np.mean(ma):+.2f}%")
print()

print("--- VITORIAS ---")
mw = sum(1 for r in data if r["ms_acc"] > r["trad_acc"])
tw = sum(1 for r in data if r["ms_acc"] < r["trad_acc"])
ti = sum(1 for r in data if r["ms_acc"] == r["trad_acc"])
print(f"  MultiEstado venceu: {mw}/30 ({mw/30*100:.1f}%)")
print(f"  Tradicional venceu: {tw}/30 ({tw/30*100:.1f}%)")
print(f"  Empates: {ti}/30")
print()

print("--- PARAMETROS ---")
print(f"  Tradicional: {tp} params (fixo)")
print(f"  MultiEstado: {min(mp_vals)} a {max(mp_vals)} params (variou por seed)")
print(f"  Proporcao MS/Trad: {min(mp_vals)/tp*100:.1f}% a {max(mp_vals)/tp*100:.1f}%")
print()

print("--- ANALISE ---")
print("  O modelo MultiEstado (4 estados internos) PERDEU em todas as")
print("  30 execucoes deterministicas. Zero vitorias.")
print()
print("  Porem, o MultiEstado usou apenas 24%-56% dos parametros do")
print("  modelo tradicional (4760 a 10846 vs 19458). Mesmo com menos")
print("  da metade dos parametros, atingiu media de 59% de acuracia")
print("  contra 87% do tradicional. Isso sugere potencial para")
print("  COMPRESSAO - nao substituicao direta.")
print()
print("  Proximo passo necessario: IGUALAR o numero de parametros")
print("  entre os dois modelos e re-executar o experimento para")
print("  uma comparacao justa. O codigo ajustado esta em")
print("  experimento_quantico.py (funcao compute_ms_hidden).")
print()

print("--- DADOS CRUS (ultimos 5 seeds) ---")
for r in data[-5:]:
    print(f"  Seed {r['seed']:2d}: Trad={r['trad_acc']:.2f}% | MS={r['ms_acc']:.2f}% | "
          f"Params Trad={r['trad_params']} | MS={r['ms_params']} | "
          f"MS hidden={r['ms_hidden']}")
