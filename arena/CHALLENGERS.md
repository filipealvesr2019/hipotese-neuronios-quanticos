# Desafiantes da Arena

## Baselines

### Baseline MLP

Rede densa tradicional usada como referência:

```text
MLP hidden 64
MLP hidden 128
MLP hidden 256
```

### V4 Sparse Top-1

Baseline experimental atual:

```text
Top-1 sparse routing
2 estados na versão econômica
gate pequeno
skip desligado
```

## V5 — Competição Direta

**Ideia:** remover o gate externo.

Cada especialista produz:

```text
saída
score de confiança
```

O especialista com maior score vence e sua saída é usada.

Pergunta:

```text
O gate separado é overhead desnecessário?
```

## V6 — Top-2 Sparse Routing

**Ideia:** usar dois especialistas ativos em vez de um.

Pergunta:

```text
Dois especialistas complementares melhoram accuracy o suficiente para justificar FLOPs extras?
```

## V7 — Árvore Hierárquica

**Ideia:** roteamento em árvore:

```text
gate raiz -> grupo
gate local -> especialista
```

Pergunta:

```text
Roteamento hierárquico reduz custo do gate e melhora separação dos especialistas?
```

## V8 — Gate com Memória de Uso

**Ideia:** penalizar especialistas muito usados durante treino/inferência.

Pergunta:

```text
Memória de uso reduz colapso da Layer 2 sem prejudicar accuracy?
```

## V9 — Especialistas Low-Rank

**Ideia:** substituir matrizes densas dos especialistas por fatores:

```text
W = A x B
```

Pergunta:

```text
Especialistas compactos low-rank reduzem FLOPs/parâmetros mantendo accuracy?
```

Prioridade inicial alta, porque ataca diretamente o overhead dos especialistas em entradas 784D.

## Ordem Recomendada

```text
1. Consolidar V4 econômica com entropia e seeds 1-3
2. V9 Low-Rank
3. V5 Competição Direta
4. V6 Top-2
5. V7 Árvore
6. V8 Memória
```
