import numpy as np
import os
import sys

# Add parent to path for common imports if any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bateria_completa import make_circles, normalize, train_test_split, set_seed, softmax_crossentropy
from v3_skip_mlp_gate import MLPMultiEstadoV3, get_ms_v3_hidden

def compute_entropy(dist):
    """Calcula entropia de uma distribuição de probabilidade."""
    # Adiciona pequeno epsilon para evitar log(0)
    dist = dist + 1e-10
    return -np.sum(dist * np.log2(dist))

def train_and_track_entropy(model, X, y, Xv, epochs=300, bs=64, lr=0.01):
    n = X.shape[0]
    history = []
    
    # max entropia para 4 estados = log2(4) = 2.0
    
    for ep in range(epochs):
        lr_ = lr * (0.99 ** ep)
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            e = min(s + bs, n)
            ids = idx[s:e]
            Xb, yb = X[ids], y[ids]
            logits = model.forward(Xb)
            _, grad = softmax_crossentropy(logits, yb)
            model.backward(grad, lr_)
            
        # A cada 10 épocas, medir a distribuição média do gate no conjunto de validação
        if ep % 10 == 0 or ep == epochs - 1:
            g1, g2 = model.analyze_gate(Xv)
            
            # Distribuição média
            dist1 = np.mean(g1, axis=0)
            dist2 = np.mean(g2, axis=0)
            
            # Entropia média de cada amostra vs entropia da média global
            # Aqui medimos a entropia da distribuição global de ativação
            h1 = compute_entropy(dist1)
            h2 = compute_entropy(dist2)
            
            history.append({
                'epoch': ep,
                'l1_dist': dist1.tolist(),
                'l2_dist': dist2.tolist(),
                'l1_entropy': float(h1),
                'l2_entropy': float(h2)
            })
            
    return history

if __name__ == "__main__":
    print("==================================================")
    print(" TESTE D: Entropia e Evolução do Gate no Treino")
    print("==================================================")
    
    set_seed(42)
    X, y = make_circles(n_samples=2000, noise=0.1, seed=42)
    X, _, _ = normalize(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y)

    # Pegando mesmo hidden_dim do V3 anterior (~4570 param)
    h_v3 = get_ms_v3_hidden(4482, 2, 2, 4, gate_hidden=16)
    
    mv3 = MLPMultiEstadoV3(2, h_v3, 2, 4, seed=42, temperature=0.3, gate_hidden=16)
    
    print(f"Treinando V3 e rastreando distribuição (Hidden: {h_v3}, Params: {mv3.params()})")
    
    history = train_and_track_entropy(mv3, Xtr, ytr, Xva, epochs=300, lr=0.01)
    
    print("\n--- Evolução da Distribuição dos Estados (A cada 30 épocas) ---")
    print(f"{'Epoca':>5} | {'Layer 1 (E1, E2, E3, E4)':>30} | {'Layer 2 (E1, E2, E3, E4)':>30}")
    print("-" * 75)
    
    for item in history:
        ep = item['epoch']
        if ep % 30 == 0 or ep == 299:
            d1 = item['l1_dist']
            d2 = item['l2_dist']
            s1 = f"{d1[0]:.2f}, {d1[1]:.2f}, {d1[2]:.2f}, {d1[3]:.2f}"
            s2 = f"{d2[0]:.2f}, {d2[1]:.2f}, {d2[2]:.2f}, {d2[3]:.2f}"
            print(f"{ep:>5} | {s1:>30} | {s2:>30}")

    print("\n--- Entropia Global de Uso (Max = 2.0) ---")
    print(f"{'Epoca':>5} | {'Layer 1':>15} | {'Layer 2':>15}")
    print("-" * 45)
    for item in history:
        ep = item['epoch']
        if ep % 30 == 0 or ep == 299:
            print(f"{ep:>5} | {item['l1_entropy']:>15.4f} | {item['l2_entropy']:>15.4f}")
            
    print("\nConclusão visual:")
    print("  Se a entropia for próxima a 2.0, uso é uniforme (25% cada).")
    print("  Se for próxima a 0, houve colapso total (100% num único estado).")
