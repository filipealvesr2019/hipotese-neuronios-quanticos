# Development Guide for the Heterogeneous MoE Architecture

## Purpose

This document provides guidance for researchers and developers interested in reproducing, validating, or extending the Heterogeneous Mixture-of-Experts (MoE) architecture developed in this project.

The goal is not only to maximize accuracy but also to understand the mechanisms behind emergent specialization.

---

# Project Philosophy

Throughout versions V1 to V7, we observed that system behavior cannot be explained solely by expert quality.

Three recurring principles emerged.

---

## Law 1 — Router Primacy

The Router defines the geometry of specialization.

Without an effective routing mechanism, experts tend to converge toward similar functions.

---

## Law 2 — Experts as a Secondary Condition

Experts do not create specialization by themselves.

They adapt to partitions imposed by the Router.

---

## Law 3 — Boundary Specialization

Specialization emerges at routing boundaries.

It does not reside within individual experts.

---

# Recommended Architecture

## Heterogeneous Experts

Avoid identical experts.

Example:

```python
hidden_sizes = [64, 128, 128, 256, 512]
```

Architectural diversity was one of the strongest contributors to functional specialization.

---

## Contextual Router

Use a multi-layer routing network.

Avoid purely linear gates.

Example:

```text
Input
↓
Linear
↓
ReLU
↓
Linear
↓
ReLU
↓
Linear
↓
Softmax
```

---

## Residual Routing

Promote a primary expert.

Example:

```python
Top1 = 1.0
Top2 = 0.5
Top3 = 0.5
```

Normalized:

```python
[0.5, 0.25, 0.25]
```

This reduces responsibility dilution.

---

# Required Metrics

Do not rely on accuracy alone.

Always track:

## Accuracy

Final predictive performance.

---

## Expert Redundancy Index (ERI)

Measures predictive overlap between experts.

Desired:

```text
Low ERI
```

---

## Gini Index

Measures expert usage concentration.

Interpretation:

* Low = uniform utilization
* High = emergent pruning

---

## Routing Stability (RS)

Measures routing consistency over time.

Desired:

```text
RS → 0
```

---

# Recommended Experiments

## Ablation Study

Remove individual components:

* Contextual Router
* Adam optimization
* Heterogeneous Experts
* Residual Routing

Purpose:

Determine causal contribution.

---

## Freeze Study

Evaluate:

* Frozen Router
* Frozen Experts

This experiment produced the central discovery of the project.

---

## Sweet Spot Test

Evaluate extreme scale:

```python
N_experts = [5, 10, 20, 40, 80]
```

Track:

* Accuracy
* FLOPs
* ERI
* Gini

**Sweet Spot Guidance (U-Curve of Sparse Scaling)**:
- Always test multiple values for `N_experts` simultaneously.
- Identify the exact point where adding more experts begins to generate redundancy and accuracy loss (the valley of the U-Curve, often between N=10 and N=20).
- Monitor the Gini Index to confirm **Emergent Pruning**. If the Gini fails to spike (e.g., > 0.85) when scaling to N=80, the router is failing to actively partition the manifold.
- The absolute Sweet Spot occurs at extreme saturation (very high N) where accuracy fully recovers via active pruning without penalizing inference FLOPs (since Top-K is rigidly small). Plot `Accuracy vs N_experts` to visually isolate the optimal recovery point.

---

# Negative Results and Lessons Learned

These insights help researchers avoid repeating months of unnecessary experimentation:

* **Identical experts converge**: Without heterogeneity or Soft Cosine Diversity, networks with the exact same architecture tend to converge to similar functions, defeating the purpose of an MoE.
* **Blind scaling fails**: Adding an arbitrary number of experts does not guarantee an increase in accuracy if the router fails to isolate functions. There is a *U-Curve of Sparse Scaling* that must be respected.
* **Frozen routers destroy sparsity**: Allowing experts to learn without the router's active guidance (as seen in the Freeze Study) turns them all into redundant generalists, making any future efficient manifold partitioning impossible.
* **Poorly tuned Top-K**: An excessively large Top-K (e.g. all experts active) forces higher FLOP costs and dilutes responsibility, which can severely reduce final accuracy.
* **The Sweet Spot is empirical**: Model capacity should always be assessed by simultaneously analyzing Accuracy, FLOPs, and Gini Index to determine exactly when adding experts turns into computational dead weight.

---

# Future Directions

* CIFAR-10
* CIFAR-100
* SVHN
* Tiny ImageNet
* Real-world tabular datasets
* Small Transformer MoE

---

# Conclusion

Current evidence suggests that MoE systems should be understood primarily as manifold partitioning architectures.

The Router is not merely a selector.

It actively shapes the learning geometry.
