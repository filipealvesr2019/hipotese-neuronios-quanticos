# E1/E2 — MNIST Accuracy/FLOPs Matrix

Data: 2026-06-08

Configuração:

```text
Dataset: MNIST completo
Treino: 60.000 imagens
Teste: 10.000 imagens
Seed: 1
Épocas: 5
Batch size: 128
Estados V4: 4
Gate hidden: 32
```

## Resultados

| Modelo | Hidden | Accuracy | Loss Final | FLOPs/amostra | Params | Treino | Inferência | Acc/MFLOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 64 | 93.71% | 0.2134 | 109.824 | 55.050 | 12.09s | 0.078s | 8.5327 |
| MLP | 128 | 93.80% | 0.1961 | 236.032 | 118.282 | 16.20s | 0.110s | 3.9740 |
| MLP | 256 | 94.13% | 0.1867 | 537.600 | 269.322 | 28.57s | 0.195s | 1.7509 |
| V4 | 64 | 92.70% | 0.2106 | 273.152 | 300.114 | 66.86s | 0.305s | 3.3937 |
| V4 | 128 | 93.30% | 0.1883 | 528.384 | 615.762 | 89.85s | 0.453s | 1.7658 |
| V4 | 256 | 93.31% | 0.1744 | 1.137.152 | 1.369.938 | 174.78s | 0.687s | 0.8206 |

## Leitura

Nesta matriz, a V4 não vence a curva de eficiência do MLP.

O ponto mais eficiente foi:

```text
MLP 64
93.71% accuracy
109.824 FLOPs/amostra
8.5327 accuracy/MFLOP
```

A melhor V4 por accuracy foi:

```text
V4 256
93.31% accuracy
1.137.152 FLOPs/amostra
0.8206 accuracy/MFLOP
```

A melhor V4 por eficiência foi:

```text
V4 64
92.70% accuracy
273.152 FLOPs/amostra
3.3937 accuracy/MFLOP
```

## Conclusão

MNIST seed 1 confirma que a V4 consegue aprender o dataset e ficar relativamente próxima do MLP, mas refuta a hipótese forte nesta configuração:

```text
V4 não entrega mesma accuracy com menos FLOPs que MLP no MNIST.
```

O gargalo provável é que, em entradas de alta dimensionalidade, o custo do gate, do skip e da estrutura de especialistas supera a economia do Top-1 routing. Além disso, a implementação NumPy não executa sparse routing com kernels otimizados.

## Próximo Passo

Antes de rodar 10 seeds desta configuração, a prioridade deve ser testar variantes mais econômicas:

```text
gate menor
sem skip em camadas onde ele domina FLOPs
V4 auto-hidden por teto de FLOPs
2 estados
especialistas low-rank
```

Rodar 10 seeds da matriz atual provavelmente confirmaria melhor o resultado negativo, mas não atacaria o gargalo arquitetural.
