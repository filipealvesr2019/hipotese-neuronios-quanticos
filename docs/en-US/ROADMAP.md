# Research Roadmap — Sparse Routing & Mixture of Experts (V4+)

This document organizes the next experiments by scientific question. The goal is to avoid random experiments and directly test the revised hypothesis: can compact experts with sparse routing preserve performance while reducing compute?

## E1 — MNIST

**Question:** Does V4 preserve MLP performance on a real 10-class dataset?

**Initial matrix:**

| Architecture | Hidden | Seeds |
| --- | ---: | ---: |
| MLP | 64 | 10 |
| MLP | 128 | 10 |
| MLP | 256 | 10 |
| V4 Sparse | 64 | 10 |
| V4 Sparse | 128 | 10 |
| V4 Sparse | 256 | 10 |

**Metrics:** accuracy, final loss, training time, inference time, estimated FLOPs, and parameters.

**Status:** started. The first single-seed run showed V4 close to the MLP, but not yet with a meaningful FLOPs reduction.

## E2 — Accuracy/FLOPs Curve

**Question:** Where does V4 start to be worth it?

Build the `accuracy vs FLOPs` curve and measure `accuracy per MFLOP`. A strong result would look like:

```text
MLP 128 ≈ V4 64
same accuracy
fewer FLOPs
```

## E3 — Real Specialization

**Question:** Do the experts actually learn different things?

During inference, record class, image, and selected expert. Generate tables and heatmaps by class:

```text
class 0 -> expert 2
class 1 -> expert 1
class 2 -> expert 3
```

## E4 — Fashion-MNIST

**Question:** Does V4 work beyond digits?

Repeat the MNIST battery on clothing, shoes, and related objects. This separates digit-specific behavior from more semantic visual generalization.

## E5 — State Count Scaling

**Question:** Is there an optimal number of experts?

Test:

```text
2, 4, 8, 16, 32 states
```

Record accuracy, FLOPs, expert usage, and entropy. This is one of the most important experiments because it directly tests whether additional states help or only create collapse/waste.

## E6 — Entropy Curve

**Question:** When does expert collapse happen?

Record by epoch:

```text
layer1 entropy
layer2 entropy
traffic distribution per expert
```

## E7 — Robustness

**Question:** Are experts more robust to noise?

Test MNIST with noise:

```text
10%, 20%, 30%
```

Compare degradation between MLP and V4.

## E8 — Transformer Bridge

**Question:** Does the idea survive when the dense layer becomes a Transformer FFN?

Start small:

```text
Tiny Transformer
TinyStories
local GPT mini
```

Measure loss, perplexity, tokens/s, VRAM, and estimated FLOPs.
