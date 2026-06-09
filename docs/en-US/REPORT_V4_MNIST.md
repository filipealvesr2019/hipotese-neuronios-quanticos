# Consolidated Report — V4 Sparse Routing on MNIST
*Generated: 2026-06-08 22:00*

## 1. 10-Seed Validation — V4 s=2/g=8/no-skip vs MLP 128

| Metric | MLP 128 | V4 s=2/g=8/no-skip |
|---------|--------|---------------------|
| Mean   | 95.05% | 94.78% |
| Std     | 0.15% | 0.36% |
| Min     | 94.84% | 93.97% |
| Max     | 95.32% | 95.23% |
| Mean diff (MLP-V4) | | +0.28pp |
| Head-to-head | | V4=2/10 MLP=8/10 |

### Per-seed

| Seed | MLP | V4 | Diff | L1_H | L2_H | L2 Regime |
|------|-----|-----|------|------|------|-----------|
| 1 | 94.97% | 95.23% | -0.26pp | 0.087 | 0.011 | COLLAPSE |
| 2 | 94.91% | 95.03% | -0.12pp | 0.118 | 0.001 | COLLAPSE |
| 3 | 95.32% | 95.11% | +0.21pp | 0.012 | 0.004 | COLLAPSE |
| 4 | 95.18% | 94.47% | +0.71pp | 0.992 | 0.000 | COLLAPSE |
| 5 | 94.97% | 93.97% | +1.00pp | 0.988 | 0.869 | HIGH |
| 6 | 94.84% | 94.52% | +0.32pp | 0.751 | 0.441 | MODERATE |
| 7 | 95.16% | 94.78% | +0.38pp | 0.100 | 0.348 | MODERATE |
| 8 | 94.96% | 94.75% | +0.21pp | 0.571 | 0.521 | HIGH |
| 9 | 95.00% | 94.86% | +0.14pp | 0.004 | 0.418 | MODERATE |
| 10 | 95.23% | 95.07% | +0.16pp | 0.009 | 0.547 | HIGH |

## 2. Temperature Sweep

### Seed 1 (naturally collapsed)

| T | Acc | L1_H | L2_H |
|---|-----|------|------|
| 0.5 | 95.13% | 0.201 | 0.074 |
| 1.0 | **95.23%** | 0.087 | 0.011 |
| 2.0 | 95.04% | 0.206 | ~0.000 |

### Seed 5 (naturally high entropy)

| T | Acc | L1_H | L2_H |
|---|-----|------|------|
| 0.5 | 94.13% | 0.972 | 0.741 |
| 1.0 | 93.97% | 0.988 | 0.869 |
| 2.0 | 94.18% | 0.999 | 0.801 |

**Conclusion:** Temperature does not change the entropy regime. Gate behavior is determined by initialization, not by the temperature hyperparameter.

## 3. States Scale (seed 1)

| States | Acc | L1_H | L2_H | Params | FLOPs sparse | FLOPs dense | Train (s) |
|---------|-----|------|------|--------|-------------|-------------|-----------|
| s=2 | 95.23% | 0.087 (collapse) | 0.011 (collapse) | 242622 | 250688 | 484160 | 73.0 |
| s=4 | 94.30% | 0.025 (collapse) | **0.591** (distributed) | 476642 | 250752 | 951168 | 120.2 |
| s=8 | 93.66% | **0.486** (moderate) | 0.081 (collapse) | 944682 | 250880 | 1885184 | 227.5 |
| s=16 | **93.34%** | 0.350 (moderate) | 0.336 (moderate) | 1.88M | 251136 | 3.75M | 558 |
| MLP (ref) | 94.97% | — | — | 118282 | — | 236032 | — |

**Pattern:** Layers alternate collapse at s=4 and s=8. Only at s=16 do both layers maintain moderate entropy, but accuracy is the worst.

### Class usage (s=16, L1)

- Expert 4 dominates: classes 1, 2, 3, 5, 6, 8
- Expert 12 dominates: classes 0, 4, 7, 9

### Class usage (s=16, L2)

- Expert 8 dominates: classes 1, 2, 3, 5, 6, 8
- Expert 14 dominates: classes 4, 7, 9
- Expert 10 leads: class 0 (47%)
- First time both layers distribute simultaneously, but accuracy does not improve

## 4. Cost-Benefit Analysis

| Config | Acc | vs MLP | Params | FLOPs sparse |
|--------|-----|--------|--------|-------------|
| s=2 | 95.23% | +0.00pp | 242622 | 250688 |
| s=4 | 94.30% | -0.01pp | 476642 | 250752 |
| s=8 | 93.66% | -0.01pp | 944682 | 250880 |
| s=16 | 93.34% | -0.02pp | 1.88M | 251136 |
| MLP (ref) | 94.97% | baseline | 118282 | 236032 |

## 5. Conclusions

1. **V4 did not outperform MLP** across 10 seeds (94.78% vs 95.05%, diff -0.28pp)
2. **Temperature does not control collapse** — entropy regime is determined by the seed
3. **Collapsed seeds have BETTER accuracy** than high-entropy seeds
4. **Layers alternate collapse** — they never both distribute simultaneously until s=16
5. **More states worsens accuracy** (s=2: 95.23%, s=4: 94.30%, s=8: 93.66%, s=16: 93.34%)
6. **Real specialization exists** (s=16 L1: experts 4 and 12; L2: experts 8 and 14) but does not improve accuracy
7. **Problem is architecture, not routing** — V4 has a fundamental limitation in gradient flow through top-1 hard selection
8. **Layers alternate collapse until s=16**, when both finally distribute, but with the worst accuracy — the gate spreads out by force, not by necessity
