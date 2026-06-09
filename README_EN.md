# MultiState Neurons – Sparse Routing Research

*Leia isto em [Português](README.md).*

> **Disclaimer:** I am not making any academic claims or definitive discoveries here. I am just a computer scientist building and exploring new ideas out of curiosity. Everything here is purely experimental.

**Description:**  
This project explores the hypothesis that architectures with compact internal experts, combined with intelligent routing (Sparse Routing) and skip connections, can match the performance of traditional neural networks (MLPs) while requiring significantly lower computational cost (reduced FLOPs). The project evolved from the exploratory idea of "multi-state neurons" into a Mixture of Experts (MoE) micro-framework.

**Repository structure:**
- [`experimentos/`](./experimentos/) → Code and results isolated by version (V1, V2, V3, V4, V4.1)
- [`resultados_finais/`](./resultados_finais/) → Consolidated logs, charts, and performance tables
- [`docs/`](./docs/) → Reports, validated conclusions, and research diary (PT-BR / EN-US)
- [`datasets/`](./datasets/) → Datasets used in benchmarks (Moons, Spirals, Circles)
- [`scripts/`](./scripts/) → Scripts to run test suites and cross-validation

**Research status:**  
⚠️ **Experimental**  
Rigorous stress tests across 30 seeds demonstrated a **technical tie** in accuracy between V4 (Sparse Routing Top-1) and the traditional MLP across multiple complex domains. However, the V4 architecture achieves this limit consuming **approximately 50% fewer FLOPs** during inference, as the routing successfully eliminates the activation and leakage of unused states (noise).

**How to run:**  
- Run `python scripts/bateria_completa.py` or the specific scripts inside `experimentos/`.
- See [`docs/en-US/diary.md`](./docs/en-US/diary.md) and [`docs/en-US/conclusions.md`](./docs/en-US/conclusions.md) for detailed reports, hypothesis evolution, and ablation data.

---

## V4 MNIST Validation Results (10 seeds × 10 epochs)

**Setup:** V4 h128/s2/g8/no-skip (250k FLOPs) vs MLP 128 (236k FLOPs). Full MNIST (60k train, 10k test). Batch=128, lr=0.01, l2=1e-4.

### 10-seed head-to-head

| Metric | MLP 128 | V4 s=2 |
|--------|---------|--------|
| Mean accuracy | **95.05%** ± 0.15% | 94.78% ± 0.36% |
| Head-to-head | MLP wins 8/10 | V4 wins 2/10 |
| Mean diff | | +0.28pp (MLP leads) |

### States scale (seed 1)

| States | Acc | L1_H | L2_H | Params |
|--------|-----|------|------|--------|
| MLP ref | 94.97% | — | — | 118K |
| s=2 | **95.23%** | 0.087 (collapse) | 0.011 (collapse) | 242K |
| s=4 | 94.30% | 0.025 (collapse) | **0.591** (distributed) | 476K |
| s=8 | 93.66% | **0.486** (moderate) | 0.081 (collapse) | 944K |
| s=16 | 93.34% | 0.350 (moderate) | 0.336 (moderate) | 1.88M |

### Temperature sweep (seed 5, naturally high entropy)

| T | Acc | L2_H |
|---|-----|------|
| 0.5 | 94.13% | 0.741 |
| 1.0 | 93.97% | 0.869 |
| 2.0 | 94.18% | 0.801 |

### E3 — Parameter control (seed 1)

| Model | Params | Acc | FLOPs |
|-------|--------|-----|-------|
| MLP 128 | 118K | 94.97% | 236K |
| MLP 235 | **242K** | 95.15% | 483K |
| V4 s=2 | **242K** | **95.23%** | **250K** |

**At equal params, V4 ties MLP in accuracy (95.23% vs 95.15%) but uses 48% fewer FLOPs.** V4's real advantage is efficiency, not accuracy.

### Key findings

1. **V4 did not outperform MLP** (94.78% vs 95.05%, -0.28pp)
2. **Temperature does not control collapse** — gate regime is seed-dependent
3. **Collapsed seeds have BETTER accuracy** than high-entropy seeds
4. **More states → worse accuracy** (s=2: 95.23% → s=16: 93.34%)
5. **Real specialization exists** (s=16: experts 4/12 in L1, 8/14 in L2) but does not improve accuracy
6. **Problem is architecture, not routing** — V4 has a fundamental limitation in gradient flow via top-1 hard selection
7. **At equal parameters, V4 ties MLP** — the gain is efficiency (half the FLOPs), not accuracy

Full report: [`docs/en-US/REPORT_V4_MNIST.md`](docs/en-US/REPORT_V4_MNIST.md)
Roadmap: [`docs/en-US/ROADMAP.md`](docs/en-US/ROADMAP.md)
