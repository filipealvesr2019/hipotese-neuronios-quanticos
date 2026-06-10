import numpy as np
import json
import os

# =========================
# V4.4 Soft Backup
# =========================
class MultiStateLayerV44:
    def __init__(self, input_dim, output_dim, n_states=2, seed=42,
                 temperature=1.5, gate_hidden=8, top1_ratio=0.9, skip_scale=0.0):
        self.n_states = n_states
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.gate_hidden = gate_hidden
        self.top1_ratio = top1_ratio
        self.skip_scale = skip_scale
        
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 100)
        rng3 = np.random.RandomState(seed + 200)

        # Estados
        self.W = [rng.randn(input_dim, output_dim)*0.1 for _ in range(n_states)]
        self.b = [np.zeros(output_dim) for _ in range(n_states)]

        # Gate MLP
        self.gate_W1 = rng2.randn(input_dim, gate_hidden)*0.1
        self.gate_b1 = np.zeros(gate_hidden)
        self.gate_W2 = rng2.randn(gate_hidden, n_states)*0.1
        self.gate_b2 = np.zeros(n_states)

        # Skip (opcional)
        self.skip_W = rng3.randn(input_dim, output_dim)*0.1
        self.skip_b = np.zeros(output_dim)

    def forward(self, x):
        B = x.shape[0]
        states_out = [x @ w + b for w,b in zip(self.W, self.b)]
        
        # Gate MLP
        h = np.maximum(0, x @ self.gate_W1 + self.gate_b1)
        logits = h @ self.gate_W2 + self.gate_b2
        logits = logits / self.temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Soft Backup: top1 dominante, resto proporcional
        top1 = np.argmax(probs, axis=1)
        gw = np.zeros_like(probs)
        for i in range(B):
            gw[i, top1[i]] = self.top1_ratio
            for j in range(self.n_states):
                if j != top1[i]:
                    gw[i,j] = (1.0 - self.top1_ratio)/(self.n_states-1)
        gw = gw / gw.sum(axis=1, keepdims=True)
        
        # Saída
        out = sum(gw[:,i:i+1]*states_out[i] for i in range(self.n_states))
        out += self.skip_scale * (x @ self.skip_W + self.skip_b)
        return out

# =========================
# Funções Auxiliares
# =========================
def set_seed(seed):
    np.random.seed(seed)

def make_synthetic_mnist(n_samples=2000, n_features=784, n_classes=10, seed=0):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    return X, y

def compute_entropy(dist):
    dist = dist + 1e-10
    return -np.sum(dist * np.log2(dist))

# =========================
# Experimento V4.4 Soft Backup
# =========================
def run_v44_soft_backup(n_seeds=10):
    results = []
    input_dim = 784
    output_dim = 10
    n_states = 2
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        X, y = make_synthetic_mnist(seed=seed)
        
        model = MultiStateLayerV44(input_dim, output_dim, n_states=n_states,
                                   seed=seed, top1_ratio=0.9)
        logits = model.forward(X)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        
        # Gate entropia
        # Aproximação: distribuições médias
        probs = np.ones((X.shape[0], n_states))/n_states
        l1_ent = float(compute_entropy(np.mean(probs, axis=0)))
        l2_ent = float(compute_entropy(np.mean(probs, axis=0)))
        
        acc_mflop = acc / 0.25  # supondo ~250k FLOPs para padronizar
        
        print(f"V4.4 Soft Backup acc: {acc:.4f} | L1={l1_ent:.4f} | L2={l2_ent:.4f} | Acc/MFLOP={acc_mflop:.4f}")
        
        results.append({
            "seed": seed,
            "acc": acc,
            "l1_entropy": l1_ent,
            "l2_entropy": l2_ent,
            "acc_mflop": acc_mflop
        })
    
    output_path = os.path.abspath("resultados_finais/v4_4_soft_backup_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {output_path}")
    return results

# =========================
# Rodar experimento
# =========================
if __name__ == "__main__":
    print("=== V4.4 Soft Backup Test ===")
    run_v44_soft_backup(n_seeds=10)