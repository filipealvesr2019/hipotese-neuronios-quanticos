import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from time import time

def gini_index(x):
    diffsum = 0
    for i in x:
        for j in x:
            diffsum += abs(i - j)
    return diffsum / (2 * len(x)**2 * np.mean(x) + 1e-9)

class V65_Memory_MoE:
    def __init__(self, input_dim, hidden_sizes, output_dim,
                 lr=0.01, gate_lr=0.01, ema=0.9):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.n_experts = len(hidden_sizes)
        self.top_k = 1 # FIXO: Apenas 1 Expert processa
        self.lr = lr
        self.gate_lr = gate_lr
        self.ema = ema
        
        rng = np.random.RandomState(42)

        self.W1 = [rng.randn(input_dim, h) * 0.1 for h in hidden_sizes]
        self.b1 = [np.zeros(h) for h in hidden_sizes]
        self.W2 = [rng.randn(h, output_dim) * 0.1 for h in hidden_sizes]
        self.b2 = [np.zeros(output_dim) for _ in range(self.n_experts)]

        self.gW1 = rng.randn(input_dim, 64) * 0.1
        self.gb1 = np.zeros(64)
        self.gW2 = rng.randn(64, 32) * 0.1
        self.gb2 = np.zeros(32)
        self.gW3 = rng.randn(32, self.n_experts) * 0.1
        self.gb3 = np.zeros(self.n_experts)

        self.m_gW1, self.v_gW1 = np.zeros_like(self.gW1), np.zeros_like(self.gW1)
        self.m_gb1, self.v_gb1 = np.zeros_like(self.gb1), np.zeros_like(self.gb1)
        self.m_gW2, self.v_gW2 = np.zeros_like(self.gW2), np.zeros_like(self.gW2)
        self.m_gb2, self.v_gb2 = np.zeros_like(self.gb2), np.zeros_like(self.gb2)
        self.m_gW3, self.v_gW3 = np.zeros_like(self.gW3), np.zeros_like(self.gW3)
        self.m_gb3, self.v_gb3 = np.zeros_like(self.gb3), np.zeros_like(self.gb3)
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0

        self.expert_perf = np.zeros(self.n_experts)
        self.expert_memory = np.ones((self.n_experts, output_dim))

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X, temperature=1.0):
        B = X.shape[0]
        h_list, out_list = [], []
        for i in range(self.n_experts):
            h = np.maximum(0, X @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_list.append(h)
            out_list.append(out)

        g1 = np.maximum(0, X @ self.gW1 + self.gb1)
        g2 = np.maximum(0, g1 @ self.gW2 + self.gb2)
        logits = g2 @ self.gW3 + self.gb3
        
        logits_scaled = logits / temperature
        gate_probs = self.softmax(logits_scaled)

        # TOP-1 Absoluto (Computação Ativa Única)
        top1_idx = np.argmax(gate_probs, axis=1)
        gate = np.zeros_like(gate_probs)
        for b in range(B):
            gate[b, top1_idx[b]] = 1.0

        out = sum(gate[:, i:i+1] * out_list[i] for i in range(self.n_experts))
        return out, gate, out_list, h_list, g1, g2, gate_probs, logits_scaled

    def train_step(self, X, y, temperature=1.0):
        B = X.shape[0]
        logits, gate, out_list, h_list, g1, g2, gate_probs, logits_scaled = self.forward(X, temperature)
        probs_out = self.softmax(logits)
        probs_out[np.arange(B), y] -= 1
        probs_out /= B

        expert_acc = np.zeros(self.n_experts)
        expert_correct = np.zeros((B, self.n_experts))

        for i in range(self.n_experts):
            grad_out = probs_out * gate[:, i:i+1]
            dh = grad_out @ self.W2[i].T
            dh[h_list[i] <= 0] = 0
            
            dW2 = h_list[i].T @ grad_out
            db2 = np.sum(grad_out, axis=0)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            self.W2[i] -= self.lr * dW2
            self.b2[i] -= self.lr * db2
            self.W1[i] -= self.lr * dW1
            self.b1[i] -= self.lr * db1

            pred = np.argmax(out_list[i], axis=1)
            expert_correct[:, i] = (pred == y).astype(float)
            expert_acc[i] = np.mean(expert_correct[:, i])
            
            for c in range(self.output_dim):
                self.expert_memory[i, c] += np.sum(expert_correct[:, i] * (y == c))

        self.expert_perf = self.ema * self.expert_perf + (1 - self.ema) * expert_acc

        sum_correct = np.sum(expert_correct, axis=1, keepdims=True)
        local_target = np.where(sum_correct > 0, expert_correct / np.where(sum_correct > 0, sum_correct, 1), 1.0 / self.n_experts)
        
        memory_prior = np.zeros((B, self.n_experts))
        for b in range(B):
            memory_prior[b] = self.expert_memory[:, y[b]]
        memory_sum = np.sum(memory_prior, axis=1, keepdims=True)
        memory_prior /= np.where(memory_sum > 0, memory_sum, 1)
        
        target_probs = 0.5 * local_target + 0.5 * memory_prior
        
        # SEM LOAD BALANCE LOSS (deixando o router podar livremente a memória estrutural)
        
        d_logits_scaled = (gate_probs - target_probs) / B
        d_logits = d_logits_scaled / temperature
        
        d_gW3 = g2.T @ d_logits
        d_gb3 = np.sum(d_logits, axis=0)
        d_g2 = d_logits @ self.gW3.T
        d_g2[g2 <= 0] = 0
        d_gW2 = g1.T @ d_g2
        d_gb2 = np.sum(d_g2, axis=0)
        d_g1 = d_g2 @ self.gW2.T
        d_g1[g1 <= 0] = 0
        d_gW1 = X.T @ d_g1
        d_gb1 = np.sum(d_g1, axis=0)
        
        self.t += 1
        def adam_upd(p, g, m, v):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_h = m / (1 - self.beta1 ** self.t)
            v_h = v / (1 - self.beta2 ** self.t)
            p -= self.gate_lr * m_h / (np.sqrt(v_h) + self.eps)
            return p, m, v
        self.gW3, self.m_gW3, self.v_gW3 = adam_upd(self.gW3, d_gW3, self.m_gW3, self.v_gW3)
        self.gb3, self.m_gb3, self.v_gb3 = adam_upd(self.gb3, d_gb3, self.m_gb3, self.v_gb3)
        self.gW2, self.m_gW2, self.v_gW2 = adam_upd(self.gW2, d_gW2, self.m_gW2, self.v_gW2)
        self.gb2, self.m_gb2, self.v_gb2 = adam_upd(self.gb2, d_gb2, self.m_gb2, self.v_gb2)
        self.gW1, self.m_gW1, self.v_gW1 = adam_upd(self.gW1, d_gW1, self.m_gW1, self.v_gW1)
        self.gb1, self.m_gb1, self.v_gb1 = adam_upd(self.gb1, d_gb1, self.m_gb1, self.v_gb1)

def run_sweet_spot_memory():
    print("Gerando Dataset High-Dim...")
    X, y = make_classification(n_samples=3000, n_features=200, n_informative=50, n_classes=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_experts_list = [10, 20, 40, 80, 160]
    base_sizes = [32, 64, 128, 256, 512]
    
    epochs = 15
    batch_size = 64
    
    acc_results = []
    
    print("\nIniciando Sweet Spot Memory (Fixado Top-K=1 Computando)...")
    for n in n_experts_list:
        multiplier = max(1, n // len(base_sizes))
        hidden_sizes = []
        for size in base_sizes:
            hidden_sizes.extend([size] * multiplier)
            
        print(f"\n[Treinando N={n} Experts | Computando={1}]")
        model = V65_Memory_MoE(input_dim=X.shape[1], hidden_sizes=hidden_sizes, output_dim=10)
        
        t0 = time()
        for epoch in range(epochs):
            temperature = max(0.7, 2.0 * (0.85 ** epoch))
            idx = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch = idx[i:i+batch_size]
                if len(batch) == 0: continue
                model.train_step(X_train[batch], y_train[batch], temperature)
                
            if epoch == epochs - 1:
                logits_val, gate_val, _, _, _, _, _, _ = model.forward(X_test, temperature=0.7)
                val_acc = np.mean(np.argmax(logits_val, axis=1) == y_test)
                print(f"  Resultados | Val ACC: {val_acc:.4f} | Tempo total: {time()-t0:.1f}s")
                acc_results.append(val_acc)

    # Plot
    os.makedirs('graficos', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(n_experts_list, acc_results, marker='o', linewidth=2, color='purple')
    plt.title('Sweet Spot Memory: Accuracy Scaling with ONLY 1 Active Expert')
    plt.xlabel('Número Total de Experts na Memória')
    plt.ylabel('Val Accuracy (Top-1 Routing)')
    plt.grid(True)
    plt.savefig('graficos/sweet_spot_memory.png')
    plt.close()
    
    print("\nVisualização salva em graficos/sweet_spot_memory.png")

if __name__ == "__main__":
    run_sweet_spot_memory()
