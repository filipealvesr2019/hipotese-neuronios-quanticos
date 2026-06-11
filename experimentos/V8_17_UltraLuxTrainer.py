import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from V8_2_MultimodalArchitecture import V8_ImageToCode_Model

class UltraLuxTokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        self.add_token("<LANG_HTML>")
        self.add_token("<LANG_REACT>")
        self.add_token("<LANG_NEXTJS>")
        self.add_token("<LANG_TAILWIND>")
        self.add_token(self.pad_token)
        self.add_token(self.sos_token)
        self.add_token(self.eos_token)
        self.add_token(self.unk_token)
        
    def add_token(self, token):
        if token not in self.word2id:
            idx = len(self.word2id)
            self.word2id[token] = idx
            self.id2word[idx] = token
            
    def _split_code(self, text):
        tokens = re.findall(r'<LANG_[A-Z]+>|</?[a-zA-Z0-9_-]+>|style=\{{[^}]+\}}|className="[^"]+"|class="[^"]+"|[a-zA-Z0-9_-]+|[^\s\w]', text)
        return [t.strip() for t in tokens if t.strip()]

    def build_vocab(self, texts):
        for text in texts:
            words = self._split_code(text)
            for w in words:
                self.add_token(w)
                
    def encode(self, text):
        words = self._split_code(text)
        return [self.word2id[self.sos_token]] + [self.word2id.get(w, self.word2id[self.unk_token]) for w in words] + [self.word2id[self.eos_token]]

class UltraLuxDataset(Dataset):
    def __init__(self, metadata, data_dir, tokenizer, transform=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.metadata = metadata
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        item = self.metadata[idx]
        target_code = item["target_code"]
        img_file = item["image_file"]
            
        img_path = os.path.join(self.data_dir, "images", img_file)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        encoded_html = self.tokenizer.encode(target_code)
        
        return image, torch.tensor(encoded_html, dtype=torch.long), item

def collate_fn(batch, pad_idx):
    images, seqs, items = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    
    padded_seqs = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded_seqs[i, :len(seq)] = seq
        
    return images, padded_seqs, items

def print_stats_table(title, counts_dict, expert_sizes):
    print("\n" + "="*60)
    print(f"RELATÓRIO ESTATÍSTICO: {title}")
    print("="*60)
    
    matrix_to_plot = []
    labels = []
    
    for key, expert_counts in counts_dict.items():
        total = sum(expert_counts.values())
        print(f"\n[{str(key).upper()}] ({total} tokens):")
        
        row_probs = []
        for expert_id, count in expert_counts.items():
            pct = (count / total) * 100 if total > 0 else 0
            row_probs.append(pct / 100.0)
            if pct > 0:
                print(f"  -> Expert {expert_id} ({expert_sizes[expert_id]}n): {pct:.1f}%")
                
        matrix_to_plot.append(row_probs)
        labels.append(str(key))
        
    print("="*60)
    return np.array(matrix_to_plot), labels

def plot_heatmap(matrix, y_labels, expert_sizes, title, filename):
    os.makedirs('graficos', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="magma", 
                xticklabels=[f"E{i} ({s}n)" for i, s in enumerate(expert_sizes)], 
                yticklabels=y_labels)
    plt.title(title)
    plt.xlabel("Experts (Capacidade)")
    plt.ylabel("Categoria")
    plt.tight_layout()
    plt.savefig(f'graficos/{filename}')
    plt.close()

def categorize_depth(depth):
    if depth <= 2: return "Raso (1-2)"
    if depth <= 4: return "Médio (3-4)"
    return "Profundo (5+)"

def categorize_tags(tags):
    if tags <= 5: return "Pequeno (<=5)"
    if tags <= 15: return "Médio (6-15)"
    return "Grande (16+)"

def main():
    print("Iniciando V8_17: Treinamento Ultra-Luxo (Alto Load Balancing + Metadados Estruturais)...")
    
    data_dir = r"F:\themes_dataset_ultralux"
    meta_path = os.path.join(data_dir, "metadata_ultralux.json")
    
    if not os.path.exists(meta_path):
        print(f"Erro: Metadata não encontrada em {meta_path}. Rode o V8_16 primeiro.")
        return
        
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    all_codes = [item["target_code"] for item in metadata]
            
    tokenizer = UltraLuxTokenizer()
    tokenizer.build_vocab(all_codes)
    vocab_size = len(tokenizer.word2id)
    print(f"Vocabulário Construído: {vocab_size} tokens.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = UltraLuxDataset(metadata, data_dir, tokenizer, transform)
    pad_idx = tokenizer.word2id[tokenizer.pad_token]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    
    expert_sizes = [16, 32, 64, 128, 256]
    
    # Criando o modelo com Load Balancing Weight ALTO (0.5 ao inves do padrao para forcar os outros experts)
    # Note que a assinatura de V8_ImageToCode_Model nao recebe balance_weight no init diretamente, 
    # entao vamos multiplicar a bal_loss no loop!
    model = V8_ImageToCode_Model(vocab_size=vocab_size, expert_sizes=expert_sizes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    epochs = 10
    print(f"\nIniciando Treinamento ({epochs} Épocas com Alta Penalidade de Colapso)...")
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, seqs, _) in enumerate(dataloader):
            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]
            
            optimizer.zero_grad()
            logits, gate_probs, bal_loss = model(images, inputs)
            
            ce_loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Penalidade 10x maior para balanceamento de carga!
            loss = ce_loss + (bal_loss * 10.0) 
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f"Época {epoch+1} | Batch {batch_idx} | CE Loss: {ce_loss.item():.4f} | Bal Loss: {bal_loss.item():.4f}")
                
    print("\nExtraindo Matrizes de Especialização Visual e Estrutural...")
    model.eval()
    
    counts = {
        "component": {},
        "theme": {},
        "style": {},
        "depth_dom": {},
        "num_tags": {}
    }
    
    with torch.no_grad():
        for batch_idx, (images, seqs, items) in enumerate(dataloader):
            if batch_idx > 50: break
                
            inputs = seqs[:, :-1]
            logits, gate_probs, _ = model(images, inputs) 
            
            for b in range(inputs.size(0)):
                item = items[b]
                item["depth_dom"] = categorize_depth(item["depth_dom"])
                item["num_tags"] = categorize_tags(item["num_tags"])
                
                for category in counts.keys():
                    val = item[category]
                    if val not in counts[category]:
                        counts[category][val] = {i: 0 for i in range(5)}
                
                for t in range(inputs.size(1)):
                    token_id = inputs[b, t].item()
                    if token_id in [pad_idx, tokenizer.word2id[tokenizer.sos_token], tokenizer.word2id[tokenizer.eos_token]]:
                        continue
                    
                    winner = gate_probs[b, t].argmax().item()
                    for category in counts.keys():
                        val = item[category]
                        counts[category][val][winner] += 1

    for category in counts.keys():
        matrix, labels = print_stats_table(f"POR {category.upper()}", counts[category], expert_sizes)
        plot_heatmap(matrix, labels, expert_sizes, f"Especialização por {category.upper()}", f"expert_ultralux_{category}.png")
        print(f"[Artefato Crítico] Heatmap gerado em: graficos/expert_ultralux_{category}.png")

if __name__ == "__main__":
    main()
