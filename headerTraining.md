# Relatório de Treinamento: V8 Image-to-Code (Headers)

## O Experimento
Treinamos a arquitetura Multimodal (`V8_2_MultimodalArchitecture`) em 1000 imagens de Headers (divididas em 3 famílias) usando um Tokenizador Word-Level. O objetivo era observar se o Roteador do MoE (5 experts de tamanhos heterogêneos: 16, 32, 64, 128, 256) distribuiria as categorias sintáticas (HTML, CSS, Texto) naturalmente entre os experts.

## O Resultado Estatístico (10 Épocas)
A *Loss* caiu rapidamente de 7.22 para 0.67, indicando que o encanamento imagem-para-texto está funcionando perfeitamente e o modelo está aprendendo a sintaxe. 

No entanto, a Análise Estatística revelou um fenômeno clássico:

```text
==================================================
RELATÓRIO ESTATÍSTICO DE ESPECIALIZAÇÃO FUNCIONAL
==================================================

[Expert 0] Dominou 26.0 tokens únicos:
  -> 30.8% pertencem à categoria: HTML_ESTRUTURAL
  -> 38.5% pertencem à categoria: CSS_LAYOUT
  -> 30.8% pertencem à categoria: CONTEUDO_TEXTO

[Expert 1] Não dominou estatisticamente nenhuma categoria (Colapso)
[Expert 2] Não dominou estatisticamente nenhuma categoria (Colapso)
[Expert 3] Não dominou estatisticamente nenhuma categoria (Colapso)
[Expert 4] Não dominou estatisticamente nenhuma categoria (Colapso)
==================================================
```

## Conclusão Científica: O Colapso do Roteador (Routing Collapse)

O resultado acima é a prova empírica de um dos maiores desafios no design de MoEs: o **Colapso de Roteamento**.

1. **O que aconteceu:** O Expert 0 (provavelmente o menor, de 16 neurônios) inicializou com pesos marginalmente melhores. O Roteador mandou os primeiros tokens para ele. Como ele processou os tokens e recebeu os gradientes, ele ficou mais inteligente. O Roteador percebeu isso e começou a mandar *todos* os tokens para o Expert 0, matando de fome os Experts 1 a 4.
2. **O significado para a Teoria:** A heterogeneidade estrutural por si só não é forte o suficiente para forçar a segregação funcional *espontânea* se o roteador for ganancioso (Greedy Router).

## O Próximo Passo
Para forçar a especialização emergente, precisamos quebrar o monopólio do Expert 0. Na literatura de MoE (Switch Transformer, Mixtral), isso é feito de duas formas:
1. **Load Balancing Loss:** Uma penalidade matemática na *Loss Function* que pune o roteador se ele mandar todos os tokens para um único expert.
2. **Noisy Routing:** Adicionar ruído às probabilidades (Gate Probs) para forçar o roteador a testar outros experts nos estágios iniciais do treinamento.

A fundação do V8 está validada: o pipeline aprende. Agora o trabalho é calibrar o Roteador para induzir a especialização distribuída.

resultado Load Balancing Loss e Noisy Routing:

Iniciando Treinamento Intenso (10 Épocas para decantação estatística)...
Época 1 | Batch 0 | Loss: 7.8707
Época 1 | Batch 20 | Loss: 4.6706
Época 1 | Batch 40 | Loss: 3.9724
Época 1 | Batch 60 | Loss: 3.4957
Época 1 | Batch 80 | Loss: 3.3735
Época 1 | Batch 100 | Loss: 2.7134
Época 1 | Batch 120 | Loss: 1.9509
Época 2 | Batch 0 | Loss: 1.8288
Época 2 | Batch 20 | Loss: 1.3815
Época 2 | Batch 40 | Loss: 1.0115
Época 2 | Batch 60 | Loss: 0.9988
Época 2 | Batch 80 | Loss: 0.9892
Época 2 | Batch 100 | Loss: 0.9818
Época 2 | Batch 120 | Loss: 0.9554
Época 3 | Batch 0 | Loss: 1.0137
Época 3 | Batch 20 | Loss: 0.8246
Época 3 | Batch 40 | Loss: 0.8942
Época 3 | Batch 60 | Loss: 0.8712
Época 3 | Batch 80 | Loss: 0.9137
Época 3 | Batch 100 | Loss: 0.8172
Época 3 | Batch 120 | Loss: 0.8209
Época 4 | Batch 0 | Loss: 0.8112
Época 4 | Batch 20 | Loss: 0.8215
Época 4 | Batch 40 | Loss: 0.8383
Época 4 | Batch 60 | Loss: 0.8051
Época 4 | Batch 80 | Loss: 0.8181
Época 4 | Batch 100 | Loss: 0.7805
Época 4 | Batch 120 | Loss: 0.8030
Época 5 | Batch 0 | Loss: 0.7761
Época 5 | Batch 20 | Loss: 0.8360
Época 5 | Batch 40 | Loss: 0.7675
Época 5 | Batch 60 | Loss: 0.7638
Época 5 | Batch 80 | Loss: 0.7679
Época 5 | Batch 100 | Loss: 0.8401
Época 5 | Batch 120 | Loss: 0.7842
Época 6 | Batch 0 | Loss: 0.7351
Época 6 | Batch 20 | Loss: 0.7429
Época 6 | Batch 40 | Loss: 0.7204
Época 6 | Batch 60 | Loss: 0.7547
Época 6 | Batch 80 | Loss: 0.7692
Época 6 | Batch 100 | Loss: 0.7208
Época 6 | Batch 120 | Loss: 0.7973
Época 7 | Batch 0 | Loss: 0.7943
Época 7 | Batch 20 | Loss: 0.7044
Época 7 | Batch 40 | Loss: 0.7078
Época 7 | Batch 60 | Loss: 0.7901
Época 7 | Batch 80 | Loss: 0.7250
Época 7 | Batch 100 | Loss: 0.7428
Época 7 | Batch 120 | Loss: 0.7256
Época 8 | Batch 0 | Loss: 0.7465
Época 8 | Batch 20 | Loss: 0.6557
Época 8 | Batch 40 | Loss: 0.7104
Época 8 | Batch 60 | Loss: 0.7172
Época 8 | Batch 80 | Loss: 0.6802
Época 8 | Batch 100 | Loss: 0.7612
Época 8 | Batch 120 | Loss: 0.7247
Época 9 | Batch 0 | Loss: 0.6502
Época 9 | Batch 20 | Loss: 0.6694
Época 9 | Batch 40 | Loss: 0.6908
Época 9 | Batch 60 | Loss: 0.6868
Época 9 | Batch 80 | Loss: 0.6915
Época 9 | Batch 100 | Loss: 0.6754
Época 9 | Batch 120 | Loss: 0.7123
Época 10 | Batch 0 | Loss: 0.6717
Época 10 | Batch 20 | Loss: 0.6255
Época 10 | Batch 40 | Loss: 0.6122
Época 10 | Batch 60 | Loss: 0.7122
Época 10 | Batch 80 | Loss: 0.6476
Época 10 | Batch 100 | Loss: 0.7838
Época 10 | Batch 120 | Loss: 0.6379

Treinamento concluído. Extraindo Mapa de Especialização Funcional...

==================================================
RELATÓRIO ESTATÍSTICO DE ESPECIALIZAÇÃO FUNCIONAL
==================================================

[Expert 0] Dominou 5.0 tokens únicos:
  -> 100.0% pertencem à categoria: HTML_ESTRUTURAL

[Expert 1] Dominou 12.0 tokens únicos:
  -> 8.3% pertencem à categoria: HTML_ESTRUTURAL
  -> 91.7% pertencem à categoria: CSS_LAYOUT

[Expert 2] Dominou 5.0 tokens únicos:
  -> 20.0% pertencem à categoria: HTML_ESTRUTURAL
  -> 80.0% pertencem à categoria: CONTEUDO_TEXTO

[Expert 3] Não dominou estatisticamente nenhuma categoria principal (Possível Colapso ou Memória Passiva).

[Expert 4] Dominou 7.0 tokens únicos:
  -> 14.3% pertencem à categoria: HTML_ESTRUTURAL
  -> 85.7% pertencem à categoria: CONTEUDO_TEXTO
==================================================

[Artefato Crítico Gerado] Heatmap salvo em: graficos/expert_heatmap_v8.png

filipe@eufilip MINGW64 /f/neuronios quanticos (main)
$