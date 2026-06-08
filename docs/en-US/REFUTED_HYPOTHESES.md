# Refuted Hypotheses

## H001: More States = Higher Absolute Intelligence
**Hypothesis:** A neuron with multiple states can represent more information per parameter and surpass the upper accuracy limit of a traditional MLP.
**Result:** False.
**Evidence:** In the 30-seed stress tests across complex datasets (Moons, Spirals, 20-Features), the V4 Sparse Architecture matched, but did not significantly surpass, the Traditional MLP accuracy. The advantage lies in efficiency (FLOPs), not absolute accuracy.

---

## H002: Simple Weighted Sum is Sufficient for States
**Hypothesis:** Simply adding the outputs of different internal states (V1) is enough to leverage their independent representations.
**Result:** False.
**Evidence:** V1 severely underperformed the baseline (84.27% vs 90.26% on Circles). A simple sum acts as a single collapsed linear transformation. An active, input-dependent Gate (V2+) is required to extract value from independent states.

---

## H003: Softmax Gate is Optimal
**Hypothesis:** A standard Softmax gate properly isolates state representations.
**Result:** False.
**Evidence:** Post-training ablation in V3 showed that removing a specific state *improved* performance (+2.75pp). This proved that Softmax leaks noise from bad states. Top-1 Hard Routing (V4) was required to fix this leakage.
