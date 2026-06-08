# E1 — MNIST

Pergunta central:

```text
A V4 Sparse mantém performance próxima ao MLP tradicional em MNIST?
```

## Dataset

Baixar MNIST:

```bash
python scripts/download_mnist.py
```

Os arquivos IDX ficam em `datasets/mnist/` e são ignorados pelo Git.

## Rodada preliminar

Single seed completo:

```bash
python experimentos/e1_mnist/codigo/e1_mnist_v4.py --epochs 5 --hidden 128 --batch-size 128 --seed 1 --out e1_mnist_seed1_result.json
```

Resultado observado:

```text
MLP Tradicional: 93.80%, 236.032 FLOPs/amostra
V4 Sparse:       92.97%, 232.584 FLOPs/amostra
```

Conclusão: V4 sobreviveu ao MNIST, mas ainda não confirmou a hipótese forte de mesma accuracy com redução substancial de FLOPs.

## Curva Accuracy/FLOPs

Smoke test:

```bash
python experimentos/e1_mnist/codigo/run_mnist_matrix.py --seeds 1 --epochs 1 --dense-hidden 64 --v4-hidden 64 --states 4 --train-limit 1000 --test-limit 500
```

Matriz planejada:

```bash
python experimentos/e1_mnist/codigo/run_mnist_matrix.py --seeds 1-10 --epochs 5 --dense-hidden 64,128,256 --v4-hidden 64,128,256 --states 4
```

Saídas:

```text
resultados_finais/e1_mnist_matrix/runs.csv
resultados_finais/e1_mnist_matrix/summary.csv
resultados_finais/e1_mnist_matrix/summary.json
```
