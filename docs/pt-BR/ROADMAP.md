# Roadmap de Pesquisa — Roteamento Esparso e Mistura de Especialistas (V4+)

Este documento organiza os próximos experimentos por pergunta científica. A meta é evitar testes aleatórios e medir diretamente a hipótese revisada: especialistas compactos com roteamento esparso podem manter desempenho reduzindo computação?

## E1 — MNIST

**Pergunta:** A V4 mantém a performance do MLP em um dataset real de 10 classes?

**Matriz inicial:**

| Arquitetura | Hidden | Seeds |
| --- | ---: | ---: |
| MLP | 64 | 10 |
| MLP | 128 | 10 |
| MLP | 256 | 10 |
| V4 Sparse | 64 | 10 |
| V4 Sparse | 128 | 10 |
| V4 Sparse | 256 | 10 |

**Métricas:** accuracy, loss final, tempo de treino, tempo de inferência, FLOPs estimados e parâmetros.

**Status:** iniciado. O primeiro single seed mostrou V4 próxima do MLP, mas ainda sem redução expressiva de FLOPs.

## E2 — Curva Accuracy/FLOPs

**Pergunta:** Onde a V4 começa a valer a pena?

Construir a curva `accuracy vs FLOPs` e medir `accuracy por MFLOP`. O resultado forte seria uma configuração como:

```text
MLP 128 ≈ V4 64
mesma accuracy
menos FLOPs
```

**Grade econômica atual:**

```text
hidden = 64, 96, 112, 128
states = 2
gate = 4, 6, 8
skip = off
seeds = 1-3
epochs = 5 inicialmente
```

Cada execução V4 deve registrar accuracy, FLOPs, parâmetros, tempo e entropia/uso dos especialistas por época.

## E3 — Especialização Real

**Pergunta:** Os especialistas realmente aprendem coisas diferentes?

Durante inferência, registrar classe, imagem e especialista escolhido. Gerar tabelas e heatmaps por classe:

```text
classe 0 -> especialista 2
classe 1 -> especialista 1
classe 2 -> especialista 3
```

## E4 — Fashion-MNIST

**Pergunta:** A V4 funciona fora de dígitos?

Repetir a bateria de MNIST em roupas, sapatos e objetos similares. Esse teste separa memorização de dígitos de generalização visual um pouco mais semântica.

## E5 — Escala dos Estados

**Pergunta:** Existe um número ótimo de especialistas?

Testar:

```text
2, 4, 8, 16, 32 estados
```

Registrar accuracy, FLOPs, uso dos especialistas e entropia. Este é um dos experimentos mais importantes, porque testa diretamente se mais estados ajudam ou apenas criam colapso/desperdício.

## E6 — Curva de Entropia

**Pergunta:** Quando ocorre o colapso dos especialistas?

Registrar por época:

```text
entropia layer1
entropia layer2
distribuição de tráfego por especialista
```

**Status:** logging implementado no experimento MNIST para V4. Cada linha de histórico agora inclui distribuição e entropia normalizada das duas camadas.

## E7 — Robustez

**Pergunta:** Especialistas são mais robustos a ruído?

Testar MNIST com ruído:

```text
10%, 20%, 30%
```

Comparar degradação de MLP vs V4.

## E8 — Ponte para Transformers

**Pergunta:** A ideia sobrevive quando a camada densa vira FFN de Transformer?

Começar pequeno:

```text
Tiny Transformer
TinyStories
GPT mini local
```

Medir loss, perplexity, tokens/s, VRAM e FLOPs estimados.

## E9 — Arena de Arquiteturas

**Pergunta:** A resposta está dentro da família V4 ou existe uma arquitetura melhor de roteamento/especialização?

Estrutura:

```text
arena/
baseline_mlp
v4_sparse_top1
v5_competicao
v6_top2
v7_arvore
v8_memoria
v9_low_rank
```

**Regra central:** mesmas seeds, dataset, épocas, otimizador, batch size e cálculo de FLOPs.

**Status:** especificação inicial criada em `arena/`. V4 econômica passa a ser baseline experimental; V5-V9 entram como desafiantes.
