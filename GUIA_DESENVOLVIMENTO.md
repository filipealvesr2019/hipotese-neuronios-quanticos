# Guia de Desenvolvimento da Arquitetura MoE Heterogênea

## Objetivo

Este documento serve como referência para pesquisadores e desenvolvedores que desejam reproduzir, validar ou expandir a arquitetura de Mixture of Experts (MoE) Heterogênea desenvolvida neste projeto.

O objetivo principal não é apenas maximizar acurácia, mas compreender os mecanismos que geram especialização emergente em sistemas MoE.

---

# Filosofia do Projeto

Durante a evolução das versões V1 até V7, observamos que o comportamento do sistema não pode ser explicado apenas pela qualidade individual dos experts.

Os experimentos indicam três princípios fundamentais:

## Lei 1 — Primazia do Router

O Router define a geometria da especialização.

Sem um mecanismo de roteamento eficiente, os experts tendem a aprender funções semelhantes.

---

## Lei 2 — Experts são Condição Secundária

Experts não criam especialização por conta própria.

Eles se adaptam às regiões da manifold definidas pelo Router.

---

## Lei 3 — Especialização é um Fenômeno de Fronteira

A especialização emerge da divisão do espaço de entrada.

Ela não reside dentro de um expert isolado.

---

# Arquitetura Recomendada

## Experts Heterogêneos

Evite experts idênticos.

Exemplo:

```python
hidden_sizes = [64, 128, 128, 256, 512]
```

A heterogeneidade foi um dos fatores mais importantes para aumentar a diversidade funcional.

---

## Contextual Router

Utilize um MLP para o roteamento.

Evite gates lineares simples.

Exemplo:

```text
Input
↓
Linear
↓
ReLU
↓
Linear
↓
ReLU
↓
Linear
↓
Softmax
```

---

## Residual Routing

Priorize um expert principal.

Exemplo:

```python
Top1 = 1.0
Top2 = 0.5
Top3 = 0.5
```

Após normalização:

```python
[0.5, 0.25, 0.25]
```

Isso reduz diluição de responsabilidade.

---

# Métricas Obrigatórias

Não avalie apenas Accuracy.

Sempre medir:

## Accuracy

Qualidade final da predição.

---

## ERI (Expert Redundancy Index)

Mede redundância entre experts.

Desejável:

```text
ERI baixo
```

---

## Gini Index

Mede concentração de uso.

Interpretação:

* Baixo = uso uniforme
* Alto = pruning emergente

---

## Routing Stability (RS)

Mede estabilidade do Router ao longo do treinamento.

Desejável:

```text
RS → 0
```

---

# Experimentos Recomendados

## Ablation Study

Remover componentes individualmente:

* Sem Router Contextual
* Sem Adam
* Sem Heterogeneidade
* Sem Residual Routing

Objetivo:

Identificar contribuição causal.

---

## Freeze Study

Testar:

* Router congelado
* Experts congelados

Esse experimento foi responsável pela principal descoberta do projeto.

---

## Sweet Spot Test

Avaliar:

```python
N_experts = [5, 10, 20, 40, 80]
```

Medir:

* Accuracy
* FLOPs
* ERI
* Gini

Objetivo:

Encontrar o ponto onde mais experts deixam de trazer benefício.

---

# Resultados Negativos e Lições Aprendidas

Essas conclusões ajudam a evitar a repetição de meses de experimentação desnecessária:

* **Experts idênticos convergem**: Sem heterogeneidade ou Soft Cosine Diversity, redes com a mesma arquitetura tendem a convergir para funções muito semelhantes, invalidando a premissa do MoE.
* **Escala cega falha**: Aumentar o número de experts indefinidamente não garante ganho de accuracy se o roteador não isolar funções. Existe uma *U-Curve of Sparse Scaling* a ser respeitada.
* **Router congelado destrói a esparsidade**: Deixar os experts aprenderem sem a guia do roteador (Freeze Study) faz com que todos se tornem generalistas redundantes, inviabilizando qualquer divisão eficiente posterior da manifold.
* **Top-K mal ajustado**: Top-K excessivamente grande (ex: todos os experts ativos) força maiores custos de FLOPs e dilui a responsabilidade, o que pode reduzir a accuracy final.
* **O Sweet Spot é empírico**: A capacidade do modelo sempre deve ser medida analisando simultaneamente Accuracy, FLOPs e Gini Index para determinar onde adicionar experts se torna peso morto.

---

# Próximos Desafios

* CIFAR-10
* CIFAR-100
* SVHN
* Tiny ImageNet
* Dados tabulares reais
* Small Transformer MoE

---

# Conclusão

Os resultados atuais sugerem que sistemas MoE devem ser entendidos como arquiteturas de particionamento de manifold.

O Router não atua apenas como um mecanismo de seleção.

Ele define a geometria do aprendizado.
