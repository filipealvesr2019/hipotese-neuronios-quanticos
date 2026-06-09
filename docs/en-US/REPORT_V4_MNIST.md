# Relatorio Consolidado — V4 Sparse Routing no MNIST
*Gerado em: 2026-06-08 22:00*

## 1. Validacao 10 seeds — V4 s=2/g=8/no-skip vs MLP 128

| Metrica | MLP 128 | V4 s=2/g=8/no-skip |
|---------|--------|---------------------|
| Media   | 95.05% | 94.78% |
| Std     | 0.15% | 0.36% |
| Min     | 94.84% | 93.97% |
| Max     | 95.32% | 95.23% |
| Diff media (MLP-V4) | | +0.28pp |
| Confronto direto | | V4=2/10 MLP=8/10 |

### Per-seed

| Seed | MLP | V4 | Diff | L1_H | L2_H | Regime L2 |
|------|-----|-----|------|------|------|-----------|
| 1 | 94.97% | 95.23% | -0.26pp | 0.087 | 0.011 | COLAPSO |
| 2 | 94.91% | 95.03% | -0.12pp | 0.118 | 0.001 | COLAPSO |
| 3 | 95.32% | 95.11% | +0.21pp | 0.012 | 0.004 | COLAPSO |
| 4 | 95.18% | 94.47% | +0.71pp | 0.992 | 0.000 | COLAPSO |
| 5 | 94.97% | 93.97% | +1.00pp | 0.988 | 0.869 | ALTO |
| 6 | 94.84% | 94.52% | +0.32pp | 0.751 | 0.441 | MODERADO |
| 7 | 95.16% | 94.78% | +0.38pp | 0.100 | 0.348 | MODERADO |
| 8 | 94.96% | 94.75% | +0.21pp | 0.571 | 0.521 | ALTO |
| 9 | 95.00% | 94.86% | +0.14pp | 0.004 | 0.418 | MODERADO |
| 10 | 95.23% | 95.07% | +0.16pp | 0.009 | 0.547 | ALTO |

## 2. Variancia de Temperatura

### Seed 1 (colapso natural)

| T | Acc | L1_H | L2_H |
|---|-----|------|------|
| T=0.5 | 95.13% | 0.201 | 0.074 |
| T=2.0 | 95.04% | 0.206 | -0.000 |

### Seed 5 (entropia alta)

| T | Acc | L1_H | L2_H |
|---|-----|------|------|
| T=0.5 | 94.13% | 0.972 | 0.741 |
| T=2.0 | 94.18% | 0.999 | 0.801 |

**Conclusao:** Temperatura nao altera o regime de entropia. O comportamento do gate e determinado pela inicializacao, nao pelo hyperparametro de temperatura.

## 3. Escala de Estados (seed 1)

| Estados | Acc | L1_H | L2_H | Params | FLOPs sparse | FLOPs dense | Treino (s) |
|---------|-----|------|------|--------|-------------|-------------|-----------|
| s=2 | 95.23% | 0.087 | 0.011 | 242622 | 250688 | 484160 | 73.0 |
| s=4 | 94.30% | 0.025 | 0.591 | 476642 | 250752 | 951168 | 120.2 |
| s=8 | 93.66% | 0.486 | 0.081 | 944682 | 250880 | 1885184 | 227.5 |
| MLP (ref) | 94.97% | — | — | 118282 | — | 236032 | — |

### Uso por classe (s=8, L1)

Maior especializacao observada em L1 com 8 estados:
- Expert 3 domina: classes 0, 1, 3, 5, 6, 7, 9
- Expert 4 domina: classes 2, 4, 8
- Porem L2 colapsou para expert 2 em 99-100% das classes

## 4. Analise de Custo-Beneficio

| Config | Acc | vs MLP | Params | FLOPs sparse |
|--------|-----|--------|--------|-------------|
| s=2 | 95.23% | +0.00pp | 242622 | 250688 |
| s=4 | 94.30% | -0.01pp | 476642 | 250752 |
| s=8 | 93.66% | -0.01pp | 944682 | 250880 |
| MLP (ref) | 94.97% | baseline | 118282 | 236032 |

## 5. Conclusoes

1. **V4 nao superou o MLP** em 10 seeds (94.78% vs 95.05%, diff -0.28pp)
2. **Temperatura nao controla colapso** — regime de entropia e determinado pela seed
3. **Seeds com colapso tem accuracy MELHOR** que seeds com entropia alta
4. **Camadas alternam colapso** — nunca ambas distribuem simultaneamente
5. **Mais estados piora accuracy** (s=2: 95.23%, s=4: 94.30%, s=8: 93.66%)
6. **Especializacao real existe** (s=8 L1: experts 3 e 4 dividem classes) mas nao melhora accuracy
7. **Problema e arquitetura, nao roteamento** — V4 tem limitacao fundamental no fluxo de gradiente via top-1 hard selection
