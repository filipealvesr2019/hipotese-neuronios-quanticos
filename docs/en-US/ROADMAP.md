# Research Roadmap — Sparse Routing and Mixture of Experts (V4+)

*Updated: 2026-06-08 — after MNIST 10-seed validation + states scale*

---

## Current State (executive summary)

**V4 s=2/g=8/no-skip vs MLP 128 — 10 seeds MNIST:**

| Metric | MLP 128 | MLP 235 (242k params) | V4 s=2 (242k params) |
|--------|---------|-----------------------|----------------------|
| Mean   | 95.05%  | **95.19%**            | 94.78%               |
| Std    | ±0.15%  | ±0.10%                | ±0.36%               |
| FLOPs  | 236k    | 483k                  | **250k**             |
| Acc/MFLOP | 4.03 | 1.97                   | **3.79**             |

**Key validated findings:**

1. **V4 does not outperform MLP in accuracy** (−0.28pp vs MLP 128, −0.41pp vs MLP 235)
2. **But it is 48% more FLOP-efficient** than an equal-parameter MLP
3. **Temperature does not control collapse** — entropy regime is seed-defined
4. **Collapsed seeds have BETTER accuracy** than high-entropy seeds
5. **More states worsens accuracy** (s=2: 95.23% → s=16: 93.34%)
6. **Real specialization exists but does not improve results**
7. **Problem is architecture, not routing** — fundamental limitation in top-1 hard selection

---

## Next Steps (priority-ordered)

### 1️⃣ Confirm efficiency and stability

**Question:** Does the 48% FLOP reduction with ~0.4pp accuracy loss hold across more seeds?

- Run seeds 11–20 for V4 s=2 and MLP 235
- Test V4 s=2 vs s=4 in economical configurations
- Measure std of accuracy and L2 entropy
- **Key metric:** Acc/MFLOP

### 2️⃣ Explore gate and state trade-offs

**Question:** Is there an optimal accuracy vs efficiency point?

- Temperature (0.5, 1.0, 2.0) on naturally high-entropy seeds
- States scale (2, 4, 8, 16) vs L1/L2 entropy
- Log expert distribution per class
- **Key metric:** accuracy × FLOPs curve

### 3️⃣ V4 Economical architecture optimization

**Question:** Which config minimizes accuracy loss while maximizing FLOP savings?

- Hidden size tuned for balanced FLOPs
- Smaller gate (4–8)
- Fewer states (2–4)
- No skip
- **Goal:** find the sweet spot on the accuracy/FLOPs curve

### 4️⃣ Generalization tests

**Question:** Does V4 efficiency hold on other datasets?

- Moons, Spirals, 20 Features (synthetic)
- MNIST full ✅ (completed)
- Fashion-MNIST
- CIFAR-10 (future)
- **Key metric:** Acc/MFLOP across datasets

### 5️⃣ Arena of architectures (V5–V9)

**Question:** Is there a better routing/specialization architecture?

| Challenger | Core idea |
|------------|-----------|
| V5 | Experts compete without external gate |
| V6 | Top-2 sparse routing |
| V7 | Hierarchical gate tree |
| V8 | Gate with usage memory |
| V9 | Low-rank experts |

**Rule:** same seeds, dataset, epochs, batch, lr, l2.

### 6️⃣ Multi-agent / system integration

**Question:** Does FLOP savings translate to real-world gains in multi-agent systems?

- Each expert as an independent "agent"
- Measure real savings in local pipelines
- Evaluate impact on circuit simulations

### 7️⃣ Documentation and visualization

- Consolidate logs: entropy, distribution, Acc/MFLOP
- Charts: Accuracy vs FLOPs, entropy per layer
- PT-BR/EN-US diaries ready for publication

---

## Completed experiments

| Experiment | Status | Main result |
|-----------|--------|-------------|
| E1 — MNIST 10 seeds V4 s=2 | ✅ | 94.78% ±0.36% vs MLP 95.05% ±0.15% |
| E2 — Temperature | ✅ | Does not alter entropy regime |
| E3 — Parameter control | ✅ | V4 ties equal-param MLP in accuracy at half the FLOPs |
| E4 — States scale | ✅ | More states → worse accuracy (s=2: 95.23% → s=16: 93.34%) |
| E5 — Class specialization | ✅ | Exists but does not improve accuracy |

---

## Refuted hypotheses

1. ~~"More experts = better accuracy"~~ → False. More states worsens.
2. ~~"Temperature controls gate collapse"~~ → False. Regime is seed-defined.
3. ~~"V4 buys performance with more parameters"~~ → Partial. At equal params, ties; the real gain is efficiency.

## Hypotheses under test

1. "Economical V4 (s=2, small gate) maximizes Acc/MFLOP"
2. "Efficiency holds on other datasets"
3. "V5–V9 can beat V4 in accuracy or efficiency"
