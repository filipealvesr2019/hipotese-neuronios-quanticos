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
from V8_2_MultimodalArchitecture import V8_ImageToCode_Model

# Tokenizador Word-Level Simples (Expressão Regular para separar tags e estilos)
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
        # Separa tags HTML, atributos CSS e palavras
        # Padrão: pega <tag>, </tag>, propriedades CSS como padding: 20px 40px;, etc.
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

class HeaderDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
        self.meta_paths = sorted(glob.glob(os.path.join(data_dir, "metadata", "*.json")))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Imagem
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Alvo: html_body lido do JSON (desprezamos a casca <html>)
        meta_path = self.meta_paths[idx]
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            html_target = meta["html_body"]
            
        # Tokenizar (Word Level)
        encoded_html = self.tokenizer.encode(html_target)
        
        return image, torch.tensor(encoded_html, dtype=torch.long)

def collate_fn(batch, pad_idx):
    images, seqs = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    
    padded_seqs = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded_seqs[i, :len(seq)] = seq
        
    return images, padded_seqs

def main():
    print("Iniciando V8_4: Treinamento e Extração de Gate Probs (Word-Level)...")
    
    data_dir = "dataset_headers"
    meta_files = glob.glob(os.path.join(data_dir, "metadata", "*.json"))
    if not meta_files:
        print("Erro: Nenhum header encontrado. Rode V8_3_HeaderDataset.py primeiro.")
        return
        
    all_htmls = []
    for m in meta_files:
        with open(m, 'r') as f:
            all_htmls.append(json.load(f)["html_body"])
            
    tokenizer = WordLevelTokenizer()
    tokenizer.build_vocab(all_htmls)
    vocab_size = len(tokenizer)
    print(f"Vocabulário Word-Level Construído. Tamanho: {vocab_size} tokens.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = HeaderDataset(data_dir, tokenizer, transform)
    pad_idx = tokenizer.word2id[tokenizer.pad_token]
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    
    # Modelo V8_2 com 5 Experts Variáveis
    model = V8_ImageToCode_Model(vocab_size=vocab_size, expert_sizes=[16, 32, 64, 128, 256])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    epochs = 10
    print("\nIniciando Treinamento Intenso (10 Épocas para decantação estatística)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, seqs) in enumerate(dataloader):
            # Para Language Modeling, input é a sequência[:-1] e target é a sequência[1:]
            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]
            
            optimizer.zero_grad()
            logits, gate_probs, bal_loss = model(images, inputs)
            
            # Formatar para CrossEntropy: [Batch * Seq_Len, Vocab]
            ce_loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss = ce_loss + bal_loss # Adiciona a penalidade de desbalanceamento
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Época {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    print("\nTreinamento concluído. Extraindo Mapa de Especialização Funcional...")
    
    # Avaliação: Pegar um batch inteiro e rastrear quem digitou o quê
    model.eval()
    expert_usage_per_token = {}
    
    with torch.no_grad():
        images, seqs = next(iter(dataloader))
        inputs = seqs[:, :-1]
        logits, gate_probs, _ = model(images, inputs) # gate_probs: [Batch, Seq, Experts=5]
        
        # Iterar sobre todos os batches e sequencias
        for b in range(inputs.size(0)):
            for t in range(inputs.size(1)):
                token_id = inputs[b, t].item()
                if token_id in [pad_idx, tokenizer.word2id[tokenizer.sos_token], tokenizer.word2id[tokenizer.eos_token]]:
                    continue
                
                token_str = tokenizer.id2word[token_id]
                probs = gate_probs[b, t].numpy()
                
                if token_str not in expert_usage_per_token:
                    expert_usage_per_token[token_str] = []
                expert_usage_per_token[token_str].append(probs)

    # Calcular Média de Uso de Expert por Token
    tokens_to_plot = []
    matrix_to_plot = []
    
    # Dicionários para contagem estatística sugerida
    token_categories = {
        "HTML_ESTRUTURAL": ["<header>", "</header>", "<h1>", "</h1>", "<nav>", "</nav>", "<span>", "</span>", "<button>", "</button>"],
        "CSS_LAYOUT": ["width:", "box-sizing:", "display:", "justify-content:", "align-items:", "padding:", "margin:"],
        "CSS_VISUAL": ["background-color:", "color:", "font-family:", "border-radius:", "border:", "cursor:", "font-weight:", "style="],
        "CONTEUDO_TEXTO": ["Logo", "Brand", "MySite", "App", "Home", "About", "Services", "Contact", "Dashboard", "Login", "Sign", "Enter", "&nbsp;"]
    }
    
    expert_category_dominance = {i: {cat: 0.0 for cat in token_categories.keys()} for i in range(5)}
    
    for token, prob_list in list(expert_usage_per_token.items()):
        avg_probs = sum(prob_list) / len(prob_list)
        
        # Atribui o token ao expert vencedor
        winning_expert = int(torch.tensor(avg_probs).argmax().item())
        
        # Mapear para categoria
        for cat, kw_list in token_categories.items():
            if any(kw in token for kw in kw_list):
                expert_category_dominance[winning_expert][cat] += 1
                break
                
        if len(tokens_to_plot) < 25:
            tokens_to_plot.append(token[:15])
            matrix_to_plot.append(avg_probs)
            
    print("\n" + "="*50)
    print("RELATÓRIO ESTATÍSTICO DE ESPECIALIZAÇÃO FUNCIONAL")
    print("="*50)
    for expert_id, cats in expert_category_dominance.items():
        total_tokens_won = sum(cats.values())
        if total_tokens_won > 0:
            print(f"\n[Expert {expert_id}] Dominou {total_tokens_won} tokens únicos:")
            for cat, count in cats.items():
                if count > 0:
                    pct = (count / total_tokens_won) * 100
                    print(f"  -> {pct:.1f}% pertencem à categoria: {cat}")
        else:
            print(f"\n[Expert {expert_id}] Não dominou estatisticamente nenhuma categoria principal (Possível Colapso ou Memória Passiva).")
    print("="*50)
    
    import numpy as np
    matrix_to_plot = np.array(matrix_to_plot)
    
    # Gerar Heatmap
    os.makedirs('graficos', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_to_plot, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=[f"Expert {i}" for i in range(5)], yticklabels=tokens_to_plot)
    plt.title("Especialização Emergente: Uso de Experts por Token (Word-Level)")
    plt.xlabel("Experts (Tamanhos Crescentes)")
    plt.ylabel("Tokens Funcionais")
    plt.tight_layout()
    plt.savefig('graficos/expert_heatmap_v8.png')
    plt.close()
    
    print("\n[Artefato Crítico Gerado] Heatmap salvo em: graficos/expert_heatmap_v8.png")

if __name__ == "__main__":
    main()
