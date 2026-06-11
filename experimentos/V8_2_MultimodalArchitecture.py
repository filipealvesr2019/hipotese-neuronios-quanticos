import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class HeterogeneousMoE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, top_k=1):
        super().__init__()
        self.n_experts = len(hidden_sizes)
        self.top_k = top_k
        
        # O Roteador: Decide qual expert vai processar o token atual
        self.router = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_experts)
        )
        
        # Os Experts: Redes densas de tamanhos variados
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, h),
                nn.ReLU(),
                nn.Linear(h, output_dim)
            ) for h in hidden_sizes
        ])

    def forward(self, x):
        # x shape: [Batch, Seq, Dim] ou [Batch, Dim]
        orig_shape = x.shape
        if len(orig_shape) == 3:
            x = x.view(-1, orig_shape[-1])
            
        logits = self.router(x)
        
        # 1. Noisy Routing (apenas no treinamento) para forçar exploração
        if self.training:
            noise = torch.randn_like(logits) * 0.05
            logits = logits + noise
            
        gate_probs = F.softmax(logits, dim=-1)
        
        # 2. Load Balancing Loss para punir colapso
        importance = gate_probs.sum(dim=0)
        mean_importance = importance.mean()
        bal_loss = 0.05 * ((importance - mean_importance)**2).mean()
        
        # Top-K Routing
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        out = torch.zeros(x.shape[0], self.experts[0][2].out_features, device=x.device)
        
        for i, expert in enumerate(self.experts):
            # Encontra quais amostras no batch escolheram o expert 'i'
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                # Pega a probabilidade que o roteador deu para esse expert
                # Para simplificar, assumimos top_k=1
                prob = gate_probs[mask, i].unsqueeze(-1)
                out[mask] += expert_out * prob
                
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            gate_probs = gate_probs.view(orig_shape[0], orig_shape[1], -1)
            
        return out, gate_probs, bal_loss

class V8_ImageToCode_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, lstm_dim=256, expert_sizes=[16, 32, 64, 128, 256]):
        super().__init__()
        
        # 1. Vision Encoder (O Olho)
        # Usando ResNet18 pré-treinada para extrair features base
        resnet = models.resnet18(pretrained=True)
        # Removemos a camada de classificação final e o pooling
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Projeta a saída visual (512) para a dimensão do nosso modelo (256)
        self.vision_proj = nn.Linear(512, lstm_dim)
        
        # 2. Tokenizer Embedding (O Vocabulário)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 3. Lógica Sequencial (A Memória de Curto Prazo)
        # Processa o token anterior
        self.lstm = nn.LSTM(embed_dim, lstm_dim, batch_first=True)
        
        # 4. Decoder MoE (O Cérebro de Especialização)
        # Recebe (Contexto Visual + Saída do LSTM) -> MoE -> Previsão
        self.moe = HeterogeneousMoE(
            input_dim=lstm_dim * 2, # Concatenamos Visual + Texto
            hidden_sizes=expert_sizes,
            output_dim=lstm_dim,
            top_k=1
        )
        
        # 5. Classificador Final
        self.fc_out = nn.Linear(lstm_dim, vocab_size)
        
    def forward(self, images, seq_tokens):
        # 1. Processa a Imagem
        # images: [B, 3, 224, 224]
        v_feat = self.vision_encoder(images) # [B, 512, 1, 1]
        v_feat = v_feat.view(v_feat.size(0), -1) # [B, 512]
        v_context = self.vision_proj(v_feat) # [B, lstm_dim]
        
        # 2. Processa o Texto Sequencial
        # seq_tokens: [B, Seq]
        embedded = self.embedding(seq_tokens) # [B, Seq, embed_dim]
        lstm_out, _ = self.lstm(embedded) # [B, Seq, lstm_dim]
        
        # 3. Concatena o Contexto Visual com cada passo de tempo do LSTM
        # Expande o v_context para todos os passos da sequência
        v_context_expanded = v_context.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        combined_context = torch.cat([v_context_expanded, lstm_out], dim=-1) # [B, Seq, lstm_dim * 2]
        
        # 4. Roteamento pelo MoE
        moe_out, gate_probs, bal_loss = self.moe(combined_context) # [B, Seq, lstm_dim]
        
        # 5. Previsão do Próximo Token
        logits = self.fc_out(moe_out) # [B, Seq, vocab_size]
        
        return logits, gate_probs, bal_loss

def test_architecture():
    print("Testando Arquitetura Multimodal V8_2...")
    vocab_size = 80 # Exemplo do nosso DataLoader (78 tokens)
    batch_size = 4
    seq_len = 50
    
    # Mock tensores simulando o DataLoader
    images = torch.randn(batch_size, 3, 224, 224)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    model = V8_ImageToCode_Model(vocab_size=vocab_size)
    
    print(f"Alimentando modelo com Imagens {images.shape} e Tokens {tokens.shape}")
    logits, gate_probs, bal_loss = model(images, tokens)
    
    print("\n[Sucesso] Forward Pass Concluído.")
    print(f"Logits gerados: {logits.shape} -> [Batch, Seq_Len, Vocab_Size]")
    print(f"Gate Probs:     {gate_probs.shape} -> [Batch, Seq_Len, N_Experts]")
    print(f"Balancing Loss: {bal_loss.item():.4f}")
    print("\nA Arquitetura está pronta para medir a especialização visual vs sintática.")

if __name__ == "__main__":
    test_architecture()
