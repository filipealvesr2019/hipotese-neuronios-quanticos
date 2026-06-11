import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import importlib.util
import os

# Carregar V5.9
spec = importlib.util.spec_from_file_location("V59", "experimentos/V5.9.py")
v59 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v59)

def run_visualization():
    os.makedirs("graficos", exist_ok=True)
    
    # 1. Dataset
    X, y = v59.make_spiral(n=1000, noise=0.2)
    out_dim = len(np.unique(y))
    hidden_sizes = [64, 128, 128, 256, 512]
    
    # 2. Treinar
    print("Treinando V5.9 para Visualização (Spiral)...")
    model = v59.V59_MoE(input_dim=X.shape[1], hidden_sizes=hidden_sizes, output_dim=out_dim)
    
    epochs = 20
    usage_history = []
    entropy_history = []
    
    for epoch in range(epochs):
        temperature = max(0.7, 2.0 * (0.85 ** epoch))
        idx = np.random.permutation(len(X))
        batch_size = 64
        epoch_gate_probs = []
        
        for i in range(0, len(X), batch_size):
            batch = idx[i:i+batch_size]
            if len(batch) == 0: continue
            loss_val, expert_acc = model.train_step(X[batch], y[batch], temperature)
            
            _, _, _, _, _, _, gate_probs, _ = model.forward(X[batch], temperature)
            epoch_gate_probs.append(gate_probs)
            
        all_probs = np.vstack(epoch_gate_probs)
        mean_usage = np.mean(all_probs, axis=0)
        ent = -np.sum(mean_usage * np.log(mean_usage + 1e-9))
        
        usage_history.append(mean_usage)
        entropy_history.append(ent)
        print(f"Epoch {epoch+1}/{epochs} - Entropia: {ent:.4f}")

    # 3. Coletar dados finais
    logits, gate, out_list, h_list, g1, g2, gate_probs, logits_scaled = model.forward(X, temperature=0.7)
    
    # Heatmap Classe x Expert
    class_routing = np.zeros((out_dim, model.n_experts))
    for b in range(len(y)):
        class_routing[y[b]] += gate[b]
    for c in range(out_dim):
        s = np.sum(class_routing[c])
        if s > 0: class_routing[c] /= s

    plt.figure(figsize=(8, 6))
    sns.heatmap(class_routing, annot=True, cmap="YlGnBu", xticklabels=[f"Expert {i} ({h})" for i, h in enumerate(hidden_sizes)], yticklabels=[f"Class {c}" for c in range(out_dim)])
    plt.title("Routing per Class (V5.9 Contextual Gate)")
    plt.xlabel("Experts")
    plt.ylabel("Classes")
    plt.tight_layout()
    plt.savefig("graficos/heatmap_class_expert.png")
    plt.close()
    
    # Trajetória Temporal do Gate
    plt.figure(figsize=(10, 5))
    usage_history = np.array(usage_history)
    for i in range(model.n_experts):
        plt.plot(usage_history[:, i], label=f"Expert {i} ({hidden_sizes[i]})")
    plt.title("Gate Usage over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Average Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficos/gate_usage_history.png")
    plt.close()
    
    # t-SNE do Gate Contextual (g2)
    print("Calculando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    g2_tsne = tsne.fit_transform(g2)
    
    top1_expert = np.argmax(gate, axis=1)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Colorido por Classe
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(g2_tsne[:, 0], g2_tsne[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.title("Gate Context (g2) Manifold - By Class")
    plt.colorbar(scatter, ticks=range(out_dim))
    
    # Subplot 2: Colorido por Expert Escolhido (Top 1)
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(g2_tsne[:, 0], g2_tsne[:, 1], c=top1_expert, cmap="tab10", alpha=0.7)
    plt.title("Gate Context (g2) Manifold - By Top-1 Expert")
    plt.colorbar(scatter2, ticks=range(model.n_experts))
    
    plt.tight_layout()
    plt.savefig("graficos/tsne_gate_manifold.png")
    plt.close()
    
    print("Gráficos salvos na pasta 'graficos/'!")

if __name__ == "__main__":
    run_visualization()
