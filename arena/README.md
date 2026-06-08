# Arena de Arquiteturas

Este diretório define a "rinha de arquiteturas": um protocolo comum para comparar MLP, V4 econômica e novas variantes V5+ sem mudar as regras do jogo no meio do caminho.

## Objetivo

Trocar a pergunta:

```text
A V4 é boa?
```

por:

```text
Qual arquitetura entrega mais accuracy por FLOP sob o mesmo protocolo?
```

A V4 econômica passa a ser o baseline experimental. Novas arquiteturas entram como desafiantes.

## Regras da Arena

Todos os competidores devem usar:

```text
mesmo dataset
mesmas seeds
mesmo número de épocas
mesmo batch size
mesmo otimizador
mesma normalização
mesmo cálculo de FLOPs
mesma máquina
```

Para MNIST, o protocolo inicial é:

```text
dataset: MNIST completo
seeds: 1-3 inicialmente
epochs: 5 inicialmente
batch_size: 128
lr: 0.01
l2: 1e-4
```

## Métricas

Cada run deve registrar:

```text
accuracy final
loss final
FLOPs estimados por amostra
accuracy por MFLOP
parâmetros
tempo de treino
tempo de inferência
entropia Layer 1
entropia Layer 2
distribuição de uso dos especialistas
```

## Ranking

Ranking preliminar:

| Métrica | Peso |
| --- | ---: |
| Accuracy | 40 |
| FLOPs | 40 |
| Parâmetros | 10 |
| Tempo de inferência | 10 |

O score final deve ser normalizado de 0 a 100 dentro de cada bateria.

## Critério de Resultado Forte

Um desafiante começa a ficar sério quando atinge algo como:

```text
MLP 128:      93.8%, 236k FLOPs
desafiante:   93.8%, 150k FLOPs
```

ou:

```text
MLP 128:      93.8%
desafiante:   94.3%
```

com 10 seeds ou mais.

## Baselines Atuais

| Modelo | Accuracy | FLOPs/amostra | Observação |
| --- | ---: | ---: | --- |
| MLP 64 | 93.71% | 109.824 | Melhor ponto Accuracy/FLOP atual |
| MLP 128 | 93.80% | 236.032 | Baseline médio |
| V4 Econ h96/s2/g8/no-skip | 93.59% | 185.024 | Menos FLOPs que MLP 128, -0.21pp |
| V4 Econ h128/s2/g8/no-skip | 93.92% | 250.688 | +0.12pp vs MLP 128, +6.2% FLOPs |

## Backlog de Desafiantes

Ver [CHALLENGERS.md](./CHALLENGERS.md).
