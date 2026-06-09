import json, numpy as np

def load_seed(seed):
    path = f"resultados_finais/e1_mnist_validation/v4_h128_s2_g8_noskip_seed{seed}.json"
    with open(path) as f:
        d = json.load(f)
    return d["v4_sparse"]

classes = list(range(10))
n_states = 2

# Aggregate per seed
print("=== ESPECIALIZACAO POR CLASSE — L1 (10 seeds) ===")
print()
for seed in range(1, 11):
    v4 = load_seed(seed)
    usage = np.array(v4["gate_usage_l1_by_class"])  # shape (10, n_states)
    acc = v4["accuracy"]
    gs = v4["history"][-1].get("gate_stats", {})
    l2h = gs.get("layer2", {}).get("normalized_entropy", 0)

    total = usage.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    pct = usage / total * 100  # percentage per class

    # Dominant expert per class
    dominant = np.argmax(usage, axis=1)
    dom_pct = np.max(pct, axis=1)

    # How many classes prefer expert 0 vs expert 1
    pref0 = np.sum(dominant == 0)
    pref1 = np.sum(dominant == 1)

    print(f"Seed {seed}: acc={acc*100:.2f}%  L2_H={l2h:.3f}")
    for cls in classes:
        bar = "=" * int(dom_pct[cls] / 5)
        print(f"  classe {cls}: expert {dominant[cls]} ({dom_pct[cls]:.0f}%) {bar}")
    print(f"  -> expert0 lidera em {pref0}/10 classes, expert1 lidera em {pref1}/10 classes")
    print()

# Aggregate across seeds
print("=== AGREGACAO — L1 (media 10 seeds) ===")
print()
all_usage = np.zeros((10, 10, n_states))
for seed in range(1, 11):
    v4 = load_seed(seed)
    all_usage[seed-1] = np.array(v4["gate_usage_l1_by_class"])

total_across = all_usage.sum(axis=1)  # sum over seeds
total_across_seeds = total_across.sum(axis=1, keepdims=True)
total_across_seeds[total_across_seeds == 0] = 1
avg_pct = total_across / total_across_seeds * 100

for cls in classes:
    e0 = avg_pct[cls, 0]
    e1 = avg_pct[cls, 1]
    dominant = 0 if e0 > e1 else 1
    dom_pct = max(e0, e1)
    bar = "=" * int(dom_pct / 3)
    print(f"  classe {cls}: expert0={e0:.0f}%  expert1={e1:.0f}%  -> prefere expert {dominant} ({dom_pct:.0f}%) {bar}")

print()
print("=== ESPECIALIZACAO POR CLASSE — L2 (10 seeds) ===")
print()
for seed in range(1, 11):
    v4 = load_seed(seed)
    usage = np.array(v4["gate_usage_l2_by_class"])
    n_states_l2 = usage.shape[1]
    acc = v4["accuracy"]

    total = usage.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    pct = usage / total * 100

    dominant = np.argmax(usage, axis=1)
    dom_pct = np.max(pct, axis=1)

    pref_counts = [np.sum(dominant == i) for i in range(n_states_l2)]
    print(f"Seed {seed}: acc={acc*100:.2f}%  estados={n_states_l2}")
    for cls in classes:
        bar = "=" * int(dom_pct[cls] / 5)
        print(f"  classe {cls}: expert {dominant[cls]} ({dom_pct[cls]:.0f}%)")
    print(f"  distrib: {', '.join(f'expert{i}={pref_counts[i]}/10' for i in range(n_states_l2))}")
    print()

# Aggregate L2
print("=== AGREGACAO — L2 (media 10 seeds) ===")
print()
all_usage_l2 = []
for seed in range(1, 11):
    v4 = load_seed(seed)
    all_usage_l2.append(np.array(v4["gate_usage_l2_by_class"]))

# Check if all have same n_states
n_s = all_usage_l2[0].shape[1]
total_l2 = np.zeros((10, n_s))
for u in all_usage_l2:
    total_l2 += u

avg_l2_pct = total_l2 / total_l2.sum(axis=1, keepdims=True) * 100
for cls in classes:
    parts = [f"expert{i}={avg_l2_pct[cls,i]:.0f}%" for i in range(n_s)]
    print(f"  classe {cls}: " + "  ".join(parts))
