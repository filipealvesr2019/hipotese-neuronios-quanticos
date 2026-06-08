# E1/E2 — MNIST V4 Econômica

Data: 2026-06-08

Configuração:

```text
Dataset: MNIST completo
Treino: 60.000 imagens
Teste: 10.000 imagens
Seed: 1
Épocas: 5
Batch size: 128
Variações: hidden 64/128, estados 2/4, gate 8/16, skip on/off
```

## Baselines

| Modelo | Hidden | Accuracy | FLOPs/amostra | Params | Acc/MFLOP |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 64 | 93.71% | 109.824 | 55.050 | 8.5327 |
| MLP | 128 | 93.80% | 236.032 | 118.282 | 3.9740 |

## Melhores V4 Econômicas

| Modelo | Hidden | Estados | Gate | Skip | Accuracy | FLOPs/amostra | Params | Acc/MFLOP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| V4 | 128 | 2 | 8 | não | 93.92% | 250.688 | 242.622 | 3.7465 |
| V4 | 96 | 2 | 8 | não | 93.59% | 185.024 | 177.406 | 5.0583 |
| V4 | 64 | 2 | 8 | não | 93.31% | 123.456 | 116.286 | 7.5582 |
| V4 | 64 | 2 | 8 | sim | 93.49% | 232.000 | 170.686 | 4.0297 |
| V4 | 64 | 4 | 8 | não | 93.22% | 123.520 | 225.122 | 7.5470 |

## Leitura

A matriz econômica recuperou parte da hipótese de eficiência:

```text
V4 128 / 2 estados / gate 8 / sem skip
93.92% accuracy
250.688 FLOPs/amostra
```

Comparação direta com MLP 128:

```text
MLP 128: 93.80%, 236.032 FLOPs
V4 econ: 93.92%, 250.688 FLOPs
```

A V4 econômica passou o MLP 128 em accuracy por +0.12pp, mas ainda usa ~6.2% mais FLOPs e mais parâmetros.

Comparação com MLP 64:

```text
MLP 64: 93.71%, 109.824 FLOPs
V4 64/2/g8/no-skip: 93.31%, 123.456 FLOPs
```

Aqui a V4 ficou -0.40pp abaixo e ~12.4% acima em FLOPs. Portanto, ainda não vence o ponto mais eficiente da curva.

## Conclusões

1. Remover skip reduziu bastante FLOPs e tempo, sem destruir o treinamento.
2. Dois estados foram melhores que quatro estados nesta matriz.
3. Gate 8 foi melhor que gate 16 em custo-benefício.
4. A V4 econômica ficou competitiva contra MLP 128, mas ainda não superou MLP 64 em eficiência.

## Próximo Experimento

O alvo intermediário foi testado:

```text
V4 hidden 96
2 estados
gate 4/8
sem skip
```

Resultado:

```text
V4 96 / 2 estados / gate 8 / sem skip
93.59% accuracy
185.024 FLOPs/amostra
```

Esse ponto ficou abaixo do MLP 128 em FLOPs, mas também abaixo em accuracy (-0.21pp). Em seguida, `hidden=112` também foi testado e não melhorou:

```text
V4 112 / 2 estados / gate 8 / sem skip
93.08% accuracy
217.344 FLOPs/amostra
```

## Próximo Experimento

O comportamento não é monotônico com a largura. A próxima direção deve investigar otimização e roteamento, não apenas aumentar hidden:

```text
mais épocas para V4 96/128
learning rate menor
temperatura do gate
entropia/uso dos especialistas por época
```
