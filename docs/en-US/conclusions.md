# Current Conclusions

## Research Status

**Original hypothesis:** MultiState neurons are intrinsically superior to traditional neurons.  
**Status:** refuted.

**Revised hypothesis:** compact experts with Top-1 sparse routing can preserve dense-network performance while reducing inference compute.  
**Status:** partially supported on small datasets.

**Most honest formulation:** the hypothesis of intrinsic MultiState neuron superiority was refuted; the narrower hypothesis of efficient conditional computation through sparse experts received positive experimental evidence on small problems.

---

✅ **States learn different representations:** Correlation between states is near zero. They are independent experts.

✅ **The Gate strongly influences the outcome:** Softmax routing successfully distributes workload in the first layer but suffers from mode collapse in deeper layers.

✅ **Sparse Routing reduces FLOPs:** By using Top-1 Hard Routing (V4), inference computational cost is reduced by approximately 50% compared to a dense network.

⚠️ **No evidence of absolute accuracy gain:** The architecture matches the traditional MLP, but does not surpass its upper bound of intelligence on the tested datasets.

⚠️ **Load Balancing struggles in shallow networks:** The MoE auxiliary loss did not improve the mode collapse in Layer 2, likely due to the limited depth and capacity of the network.

⚠️ **Preliminary MNIST does not yet confirm the strong hypothesis:** In a single seed run, V4 stayed close to the MLP (92.97% vs 93.80%), but the estimated FLOPs reduction was small (~1.5%). This is promising as MNIST survival, but still requires an Accuracy/FLOPs curve and multiple seeds.

---

## Maturity Assessment

**Original hypothesis:** 0% alive.  
It was refuted by V1 and the subsequent tests.

**V4 architecture:** promising.  
The V1 → V2 → V3 → V4 sequence forms a coherent technical narrative: each version addressed a bottleneck observed in the previous one.

**Evidence of practical advantage:** still limited.  
There is a strong FLOPs reduction on small problems, but the preliminary MNIST run has not yet shown substantial savings.

**Evidence of a real phenomenon:** strong.  
Expert specialization, effective routing, and recurring Layer 2 collapse appear across multiple domains. The open question is whether this scales.

**Current priority:** close MNIST with an Accuracy/FLOPs curve before moving to CIFAR.

**E2 MNIST update:** the first seed-1 Accuracy/FLOPs matrix did not favor V4. MLP 64 was more accurate and cheaper than V4 64/128/256. This suggests the current V4 is efficient on small problems, but pays too much overhead on MNIST.

**Economic V4 update:** reducing overhead partially worked. The `hidden=128, 2 states, gate=8, no skip` configuration reached 93.92% vs 93.80% for MLP 128, but still used ~6.2% more FLOPs. The promising direction is now to explore intermediate points such as `hidden=96, 2 states, gate=4/8, no skip`.

**Intermediate update:** `hidden=96, 2 states, gate=8, no skip` reached 93.59% with 185,024 FLOPs, cheaper than MLP 128 but -0.21pp in accuracy. `hidden=112` did not improve. This suggests a non-monotonic curve and shifts the next question toward optimization/routing.
