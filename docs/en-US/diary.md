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

---

### Phase 1 Synthesis: Hypothesis Reformulation
**Research status:** Promising, but not yet scaled.

**Original hypothesis:** MultiState neurons are intrinsically superior to traditional neurons.
**Result:** Refuted.

**Revised hypothesis:** Compact experts with Top-1 sparse routing can reproduce the performance of dense networks while using less inference compute.
**Result:** Partially supported on small datasets.

**Accumulated evidence:**
- V1 failed: linear state aggregation collapses into an effectively dense transformation.
- V2 partially failed: the Softmax gate specializes in the first layer, but leaks noise.
- V3 approached the baseline: MLP gate + skip connections recovered performance.
- V4 reached a technical tie: Top-1 Sparse Routing eliminated leakage from bad states.
- 30-seed validation reduced the risk that the result came from a favorable seed.
- Stress tests on Moons, Spirals, and 20-feature data preserved the technical tie.

**Current conclusion:** V4 has not shown absolute accuracy superiority. The positive result is efficiency: similar performance to a traditional MLP with roughly half the inference FLOPs.

**Main risk:** All experiments are still small, with low dimensionality and few classes. There is not enough evidence yet to claim scalability to real computer vision or Transformers.

**Next critical milestone:** MNIST.

---

### Experiment 6: E1 MNIST Preliminary Run (Single Seed)
**Hypothesis:** V4 Sparse can maintain performance close to a traditional MLP on MNIST while using fewer inference FLOPs.
**Status:** Partially supported, still inconclusive.

**Setup:**
- Dataset: full MNIST (60,000 train, 10,000 test)
- Seed: 1
- Epochs: 5
- Traditional MLP: hidden=128
- V4 Sparse: hidden=53, 4 experts, gate_hidden=32
- V4 criterion: largest automatic hidden size below the baseline FLOPs ceiling

**Result:**
- Traditional MLP: 93.80% accuracy, 236,032 estimated FLOPs/sample
- V4 Sparse: 92.97% accuracy, 232,584 estimated sparse FLOPs/sample
- Accuracy difference: -0.83pp for V4

**Observation:** V4 survived MNIST and stayed close to the MLP, but the FLOPs gain in this configuration was small (~1.5%). In wall-clock time, the NumPy V4 implementation was slower because training still computes all experts and sparse inference uses Python/NumPy masking rather than optimized kernels.

**Provisional conclusion:** MNIST did not refute V4, but it also did not yet confirm the strong hypothesis of same accuracy with a substantial FLOPs reduction. The next step is an Accuracy/FLOPs curve across multiple hidden sizes and seeds.

**Recurring behavior:** MNIST repeated the pattern observed on synthetic datasets: Layer 1 shows more distributed specialization, while Layer 2 tends to collapse into a small number of experts. Since this pattern appeared on Circles, Moons, Spirals, 20-Features, and MNIST, it should be treated as an architectural property until proven otherwise, not as an isolated accident.

**Strong-result criterion:** The result that would meaningfully strengthen the hypothesis is a smaller V4 matching a larger MLP, for example:

```text
MLP 128 ≈ V4 64
or
MLP 256 ≈ V4 128
```

That would suggest specialization is replacing part of the active width of the dense network.

**Next official milestone:** close MNIST with the `10 seeds × 6 configurations` matrix before moving to CIFAR.

---

### Experiment 7: E2 MNIST Accuracy/FLOPs Matrix (Seed 1)
**Hypothesis:** On MNIST, some V4 configuration (64, 128, 256 hidden) beats the efficiency curve of the traditional MLP (64, 128, 256 hidden).
**Result:** Refuted in this initial configuration.

**Setup:**
- Dataset: full MNIST (60,000 train, 10,000 test)
- Seed: 1
- Epochs: 5
- MLP: hidden 64, 128, 256
- V4 Sparse: hidden 64, 128, 256; 4 experts; gate_hidden=32

**Results:**

| Model | Hidden | Accuracy | FLOPs/sample | Params | Acc/MFLOP |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 64 | 93.71% | 109,824 | 55,050 | 8.5327 |
| MLP | 128 | 93.80% | 236,032 | 118,282 | 3.9740 |
| MLP | 256 | 94.13% | 537,600 | 269,322 | 1.7509 |
| V4 | 64 | 92.70% | 273,152 | 300,114 | 3.3937 |
| V4 | 128 | 93.30% | 528,384 | 615,762 | 1.7658 |
| V4 | 256 | 93.31% | 1,137,152 | 1,369,938 | 0.8206 |

**Conclusion:** V4 learns MNIST and stays close in accuracy, but it does not beat the efficiency curve. MLP 64 was both cheaper and more accurate than every V4 tested in this matrix.

**Interpretation:** In high-dimensional inputs, the cost of the gate, skip path, and expert structure may outweigh the savings from Top-1 routing. The next step should not be CIFAR or blind 10-seed repetition of this configuration; it should be a more economical V4 variant for MNIST.
