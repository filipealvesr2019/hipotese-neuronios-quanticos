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

class WordLevelTokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        self.add_token(self.pad_token)
        self.add_token(self.sos_token)
        self.add_token(self.eos_token)
        self.add_token(self.unk_token)
        
    def add_token(self, token):
        if token not in self.word2id:
            idx = len(self.word2id)
            self.word2id[token] = idx
            self.id2word[idx] = token
            
    def _split_html(self, text):
        tokens = re.findall(r'</?[a-z0-9]+>|style="[^"]+"|#?[A-Za-z0-9]+|[^\s\w]', text)
        return [t.strip() for t in tokens if t.strip()]

    def build_vocab(self, texts):
        for text in texts:
            words = self._split_html(text)
            for w in words:
                self.add_token(w)
                
    def encode(self, text):
        words = self._split_html(text)
        return [self.word2id[self.sos_token]] + [self.word2id.get(w, self.word2id[self.unk_token]) for w in words] + [self.word2id[self.eos_token]]
        
    def decode(self, ids):
        words = [self.id2word.get(int(i), self.unk_token) for i in ids]
        return " ".join([w for w in words if w not in [self.pad_token, self.sos_token, self.eos_token]])
        
    def __len__(self):
        return len(self.word2id)

class MixedComponentDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
        self.meta_paths = sorted(glob.glob(os.path.join(data_dir, "metadata", "*.json")))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        meta_path = self.meta_paths[idx]
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            html_target = meta["html_body"]
            family = meta["family"]
            
        encoded_html = self.tokenizer.encode(html_target)
        
        return image, torch.tensor(encoded_html, dtype=torch.long), family

def collate_fn(batch, pad_idx):
    images, seqs, families = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    
    padded_seqs = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded_seqs[i, :len(seq)] = seq
        
    return images, padded_seqs, families

def main():
    print("Iniciando V8_7: Análise de Especialização por Componente (Conceito vs Token)...")
    
    data_dir = "dataset_mixed"
    meta_files = glob.glob(os.path.join(data_dir, "metadata", "*.json"))
    if not meta_files:
        print(f"Erro: Nenhum dado encontrado em {data_dir}.")
        return
        
    all_htmls = []
    for m in meta_files:
        with open(m, 'r') as f:
            all_htmls.append(json.load(f)["html_body"])
            
    tokenizer = WordLevelTokenizer()
    tokenizer.build_vocab(all_htmls)
    vocab_size = len(tokenizer)
    print(f"Vocabulário Construído: {vocab_size} tokens.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MixedComponentDataset(data_dir, tokenizer, transform)
    pad_idx = tokenizer.word2id[tokenizer.pad_token]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    
    expert_sizes = [16, 32, 64, 128, 256]
    model = V8_ImageToCode_Model(vocab_size=vocab_size, expert_sizes=expert_sizes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    epochs = 10
    print(f"\nIniciando Treinamento ({epochs} Épocas)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, seqs, _) in enumerate(dataloader):
            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]
            
            optimizer.zero_grad()
            logits, gate_probs, bal_loss = model(images, inputs)
            
            ce_loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss = ce_loss + bal_loss
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f"Época {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    print("\nTreinamento concluído. Extraindo Mapa de Especialização por Componente (Família)...")
    
    model.eval()
    
    # Estrutura para armazenar contagem de chamadas aos experts por família
    family_expert_counts = {}
    
    with torch.no_grad():
        # Vamos rodar em vários batches para ter uma visão geral do dataset
        for batch_idx, (images, seqs, families) in enumerate(dataloader):
            if batch_idx > 10: # Limita a avaliação a uns ~160 exemplos para ser rápido
                break
                
            inputs = seqs[:, :-1]
            logits, gate_probs, _ = model(images, inputs) 
            
            for b in range(inputs.size(0)):
                family = families[b]
                if family not in family_expert_counts:
                    family_expert_counts[family] = {i: 0 for i in range(5)}
                
                # Para cada componente, contamos qual expert processou os tokens válidos
                for t in range(inputs.size(1)):
                    token_id = inputs[b, t].item()
                    if token_id in [pad_idx, tokenizer.word2id[tokenizer.sos_token], tokenizer.word2id[tokenizer.eos_token]]:
                        continue
                        
                    winning_expert = gate_probs[b, t].argmax().item()
                    family_expert_counts[family][winning_expert] += 1

    print("\n" + "="*60)
    print("RELATÓRIO ESTATÍSTICO: ESPECIALIZAÇÃO POR CONCEITO (FAMÍLIA)")
    print("="*60)
    
    matrix_to_plot = []
    family_labels = []
    
    for family, expert_counts in family_expert_counts.items():
        total_tokens = sum(expert_counts.values())
        print(f"\n[{family.upper()}] (Baseado em {total_tokens} tokens avaliados):")
        
        row_probs = []
        for expert_id, count in expert_counts.items():
            pct = (count / total_tokens) * 100 if total_tokens > 0 else 0
            row_probs.append(pct / 100.0) # normaliza para o heatmap
            if pct > 0:
                print(f"  -> Expert {expert_id} ({expert_sizes[expert_id]} neurônios) processou {pct:.1f}% do componente.")
                
        matrix_to_plot.append(row_probs)
        family_labels.append(family)
        
    print("="*60)
    
    # Gerar Heatmap Família vs Expert
    matrix_to_plot = np.array(matrix_to_plot)
    os.makedirs('graficos', exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_to_plot, annot=True, fmt=".2f", cmap="Purples", 
                xticklabels=[f"E{i} ({s}n)" for i, s in enumerate(expert_sizes)], 
                yticklabels=family_labels)
    plt.title("Especialização por Conceito: Família vs Expert")
    plt.xlabel("Experts (Capacidade)")
    plt.ylabel("Componente de UI")
    plt.tight_layout()
    plt.savefig('graficos/expert_concept_heatmap.png')
    plt.close()
    
    print("\n[Artefato Crítico Gerado] Heatmap salvo em: graficos/expert_concept_heatmap.png")
    print("Com isso validamos se o modelo separou por caractere/token sintático ou se separou por conceito de interface!")

if __name__ == "__main__":
    main()
