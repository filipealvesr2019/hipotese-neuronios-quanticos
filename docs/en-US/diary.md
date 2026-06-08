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

---

### Experiment 8: MNIST Economic V4 (Seed 1)
**Hypothesis:** Reducing V4 overhead (fewer states, smaller gate, optional skip) recovers competitiveness on the Accuracy/FLOPs curve.
**Result:** Partially confirmed.

**Setup:**
- Dataset: full MNIST
- Seed: 1
- Epochs: 5
- Baselines: MLP 64 and MLP 128
- V4: hidden 64/128, states 2/4, gate_hidden 8/16, skip on/off

**Best results:**

| Model | Hidden | States | Gate | Skip | Accuracy | FLOPs/sample | Acc/MFLOP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| MLP | 64 | - | - | - | 93.71% | 109,824 | 8.5327 |
| MLP | 128 | - | - | - | 93.80% | 236,032 | 3.9740 |
| V4 | 128 | 2 | 8 | no | 93.92% | 250,688 | 3.7465 |
| V4 | 64 | 2 | 8 | no | 93.31% | 123,456 | 7.5582 |

**Conclusion:** The economic direction is correct: 2 states, a small gate, and no skip greatly improve cost-effectiveness. The best economic V4 beat MLP 128 in accuracy (+0.12pp), but still used ~6.2% more FLOPs. The economic V4 64 got close to MLP 64, but still lost in both accuracy and FLOPs.

**Next target:** test V4 hidden 96, 2 states, gate 4/8, no skip. The goal is to find an intermediate point that keeps ~93.7-93.9% with fewer FLOPs than MLP 128.

---

### Experiment 9: MNIST Intermediate Economic V4 (Hidden 96/112)
**Hypothesis:** An intermediate point between V4 64 and V4 128 can preserve accuracy close to MLP 128 with fewer FLOPs.
**Result:** Not confirmed yet.

**Setup:**
- Dataset: full MNIST
- Seed: 1
- Epochs: 5
- V4: 2 states, no skip
- Hidden tested: 96 and 112
- Gate tested: 4 and 8

**Main results:**

| Model | Hidden | States | Gate | Skip | Accuracy | FLOPs/sample | Acc/MFLOP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| MLP | 128 | - | - | - | 93.80% | 236,032 | 3.9740 |
| V4 | 96 | 2 | 8 | no | 93.59% | 185,024 | 5.0583 |
| V4 | 112 | 2 | 8 | no | 93.08% | 217,344 | 4.2826 |

**Conclusion:** V4 96/gate 8 used fewer FLOPs than MLP 128 and stayed relatively close in accuracy (-0.21pp), but did not reach the target. V4 112 got worse, suggesting the curve is not monotonic with hidden size and the current problem is likely optimization/routing, not just width.

**Next direction:** test more epochs, lower learning rate, gate temperature, and per-epoch entropy/expert usage for the best economic configurations.

---

### Experiment 10: MNIST Economic V4 Gate 6 + Entropy Logging
**Hypothesis:** Gate 6 may be a sweet spot between gate 4 and gate 8, preserving accuracy with fewer FLOPs.
**Result:** Not confirmed.

**Setup:**
- Dataset: full MNIST
- Seed: 1
- Epochs: 5
- V4: 2 states, no skip
- Hidden: 64, 96, 112, 128
- Gate: 6

**Results:**

| Model | Hidden | Gate | Accuracy | FLOPs/sample | Acc/MFLOP | L1 H | L2 H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 | 64 | 6 | 92.55% | 120,048 | 7.7094 | 0.993 | 0.234 |
| V4 | 96 | 6 | 93.15% | 181,488 | 5.1326 | 0.945 | 0.029 |
| V4 | 112 | 6 | 93.07% | 213,744 | 4.3543 | 0.999 | 0.110 |
| V4 | 128 | 6 | 93.02% | 247,024 | 3.7656 | 0.901 | 0.736 |

**Conclusion:** Gate 6 did not beat gate 8. Entropy logging confirmed the structural pattern again: Layer 1 tends to use experts in a distributed way, while Layer 2 frequently collapses. The partial exception was `hidden=128/gate=6`, with a more distributed Layer 2, but without an accuracy gain.

**Next direction:** rerun the best configurations with the new logging (`h96/g8` and `h128/g8`), test more epochs, and tune learning rate/temperature.

---

### Experiment 11: Architecture Arena Definition
**Motivation:** Avoid overfitting the research process to V4. Economic V4 showed promising signs, but the bottleneck may be in the architectural family itself.

**Decision:** Create an arena where MLP, V4, and new variants compete under the same protocol.

**Rules:**
- Same dataset
- Same seeds
- Same number of epochs
- Same batch size
- Same optimizer
- Same FLOPs calculation
- Same logging of accuracy, time, parameters, and entropy

**Planned challengers:**
- V5: direct expert competition, without an external gate
- V6: Top-2 sparse routing
- V7: hierarchical expert tree
- V8: usage-memory gate
- V9: low-rank experts

**Conclusion:** V4 will not be abandoned. It becomes the experimental baseline of the arena. New architectures only survive if they beat V4/MLP in accuracy, FLOPs, or routing stability.
