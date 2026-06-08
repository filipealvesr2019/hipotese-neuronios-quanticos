# MultiState Neurons – Sparse Routing Research

*Leia isto em [Português](README.md).*

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
