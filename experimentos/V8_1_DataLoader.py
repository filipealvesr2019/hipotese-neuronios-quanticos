import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CharTokenizer:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        
        self.add_token(self.pad_token)
        self.add_token(self.sos_token)
        self.add_token(self.eos_token)
        
    def add_token(self, token):
        if token not in self.char2id:
            idx = len(self.char2id)
            self.char2id[token] = idx
            self.id2char[idx] = token
            
    def build_vocab(self, texts):
        for text in texts:
            for char in text:
                self.add_token(char)
                
    def encode(self, text):
        return [self.char2id[self.sos_token]] + [self.char2id[c] for c in text] + [self.char2id[self.eos_token]]
        
    def decode(self, ids):
        chars = []
        for i in ids:
            token = self.id2char.get(int(i), "")
            if token in [self.pad_token, self.sos_token, self.eos_token]:
                continue
            chars.append(token)
        return "".join(chars)
        
    def __len__(self):
        return len(self.char2id)

class UIDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
        self.html_paths = sorted(glob.glob(os.path.join(data_dir, "code", "*.html")))
        
        assert len(self.image_paths) == len(self.html_paths), "Erro crítico: Desalinhamento entre imagens e códigos HTML!"
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Load HTML
        html_path = self.html_paths[idx]
        with open(html_path, 'r', encoding='utf-8') as f:
            html_text = f.read().strip()
            
        # Tokenize (Adiciona SOS no início e EOS no fim)
        encoded_html = self.tokenizer.encode(html_text)
        
        return image, torch.tensor(encoded_html, dtype=torch.long)

def collate_fn(batch, pad_idx):
    images, seqs = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad sequences para que todas tenham o mesmo tamanho no batch
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    
    padded_seqs = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded_seqs[i, :len(seq)] = seq
        
    return images, padded_seqs

def main():
    print("Iniciando V8_1: Construção do Tokenizador e DataLoader...")
    
    data_dir = "dataset"
    html_files = glob.glob(os.path.join(data_dir, "code", "*.html"))
    
    if not html_files:
        print(f"Erro: Nenhum arquivo HTML encontrado em {data_dir}/code. Gere o dataset primeiro.")
        return
        
    # Build Vocabulary
    all_texts = []
    for p in html_files:
        with open(p, 'r', encoding='utf-8') as f:
            all_texts.append(f.read().strip())
            
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(all_texts)
    print(f"Vocabulário construído. Tamanho do Vocabulário (Char-Level): {len(tokenizer)}")
    
    # Define Image Transforms (Resize para 224x224 para ResNet/ViT, converte para tensor, normaliza ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Instantiate Dataset and DataLoader
    dataset = UIDataset(data_dir, tokenizer, transform=transform)
    pad_idx = tokenizer.char2id[tokenizer.pad_token]
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    print(f"Dataset pareado carregado com sucesso: {len(dataset)} amostras prontas.")
    print("Iniciando extração do primeiro batch de teste...")
    
    for images, seqs in dataloader:
        print("\n--- Teste de Batch Bem-Sucedido ---")
        print(f"Shape do Tensor de Imagem: {images.shape} -> [Batch, Canais, Altura, Largura]")
        print(f"Shape do Tensor de Texto:  {seqs.shape} -> [Batch, Sequência]")
        
        # Decode first sequence in batch to verify integrity
        decoded_text = tokenizer.decode(seqs[0])
        print("\nDecodificação da amostra 0 do batch (Teste de Integridade):")
        print(decoded_text[:150] + "...\n[Truncado para visualização]")
        break
        
    print("\nV8_1 Infraestrutura de Dados Completa e Validada. O modelo Multimodal já pode ser construído.")

if __name__ == "__main__":
    main()
