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

# Practical Recommendations and Lessons Learned

## 1. Do not increase experts before increasing the Router

Experiments V6.2 (Freeze Study) and V6.4 (Router Capacity Study) indicate that the router's capacity impacts the final quality of the system far more than the raw number of experts.

Recommended order:

```text
1. Improve the Router
2. Improve Router training
3. Improve Expert heterogeneity
4. Only then increase N_experts
```

Avoid:

```text
5 → 20 → 80 experts
```

without increasing Router capacity.

---

## 2. There is a Sweet Spot

Experiment V6.3 revealed a U-Curve.

Few experts:
```text
High efficiency
Low pruning
Good accuracy
```

Intermediate quantity:
```text
Greater Router confusion
Temporary performance drop
```

Many experts:
```text
Emergent pruning
Accuracy recovery
Higher specialization
```

There is no universal number. The Sweet Spot depends on:
* Dataset
* Router
* Top-K
* Expert Capacity

Always run a Sweet Spot Study before scaling up.

---

## 3. The Gini Index is a critical metric

Monitoring Accuracy alone can hide problems. Also track:

### Accuracy
Final quality.

### Gini
Expert usage concentration.

### ERI
Redundancy between experts.

### Routing Stability
Routing consistency.

A healthy model typically displays:
```text
Accuracy ↑
Gini ↑
ERI ↓
RS ↓
```

---

## 4. Router First, Experts Second

The Freeze Study showed:

### Router First
```text
Specialization emerges
High Gini
Better Accuracy
```

### Experts First
```text
Experts become generalists
Low Gini
Worse Accuracy
```

Specialization does not arise spontaneously. It is induced by the Router.

---

## 5. Heterogeneity is better than clones

Avoid:
```python
[128] * 20
```

Prefer:
```python
[32, 64, 128, 256, 512]
```
or
```python
[64]*4 + [128]*4 + [256]*4 + [512]*4
```

Different experts learn different regions of the manifold.

---

## 6. Do not assume more Experts means more Intelligence

A major discovery of the project:

```text
More Experts ≠ More Intelligence
```

In practice:
```text
Weak Router + 80 experts = Poor system
```
but
```text
Strong Router + 20 experts = Efficient system
```

The Router is the dominant factor.

---

## 7. Future Direction: Experts as Memory

An interesting unvalidated hypothesis:

```text
Top-1 Expert → Active Computation
Other Experts → Passive Memory
```

Possible architecture:
```text
Input
  ↓
Router
  ↓
Primary Expert
  ↓
Memory Query
  ↓
Output
```

Potential benefits:
* Fewer FLOPs
* More scaling
* Less competition between experts
* Better interpretability

This direction remains open for future experiments.

---

### Executive Summary

After the V6 and V7 batteries, the main practical conclusion of the entire project is:

> When building Sparse Mixture-of-Experts architectures, investing in a better router mathematically yields greater gains than simply adding more experts.
