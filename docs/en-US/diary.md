# Research Diary

## 2026-06-08

### Experiment 1: Baseline Hypothesis Check (Test 12)
**Hypothesis:** Independent internal states can exist without collapsing into a single linear transformation.
**Result:** Confirmed.
**Observation:** Correlation between states is close to zero (~0.01). The states do not become identical copies of each other.

---

### Experiment 2: State Specialization (V2 & V3)
**Hypothesis:** An input-dependent Gate can specialize the usage of these internal states.
**Result:** Confirmed.
**Observation:** In Layer 1, the gate successfully routes different inputs to different states. However, in Layer 2, we observed *Mode Collapse*, where a single state handles ~80% of the traffic.

---

### Experiment 3: Sparse Routing & Noise Ablation (V4)
**Hypothesis:** The Softmax gate leaks noise from "bad" states. Using a Top-1 Sparse Routing will prevent this leakage.
**Result:** Confirmed.
**Observation:** Removing state 1 (L1_E1) in V4 has 0.00pp impact, proving it was perfectly gated out and produced no noise. V4 successfully eliminates the noise leakage observed in V3.

---

### Experiment 4: 30-Seed Validation (V4 vs Traditional)
**Hypothesis:** The V4 architecture can match the traditional MLP performance.
**Result:** Confirmed.
**Observation:** 
- Traditional MLP: 87.84% ± 1.36%
- V4 Sparse: 87.67% ± 1.15%
**Conclusion:** Technical tie in accuracy. However, V4 achieves this using approximately 50% of the FLOPs during inference, validating that sparse routing and internal experts can drastically reduce computational cost while maintaining the same intelligence.

---

### Experiment 5: Stress Testing (Moons, Spirals, 20-Features)
**Hypothesis:** The efficiency discovered in V4 holds across complex domains.
**Result:** Confirmed.
**Conclusion:** V4 matched the Traditional MLP across all tested complex datasets, solidifying the architecture's validity.
