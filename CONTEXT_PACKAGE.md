# Pacote de Contexto — Arena de Arquiteturas Neurais

Gerado em: 2026-06-08

---

## 1. Visão Geral do Projeto

**Objetivo:** Explorar arquiteturas com especialistas internos compactos + roteamento esparso (MoE) que reproduzam desempenho de MLPs tradicionais com menos custo computacional.

**Hipótese original (refutada):** Neurônios MultiEstado são intrinsecamente superiores.
**Hipótese revisada:** Especialistas compactos com Top-1 Sparse Routing podem manter desempenho reduzindo FLOPs de inferência.
**Status:** Parcialmente confirmada em datasets pequenos; MNIST ainda não fechou.

**Evolução arquitetural:**
- **V1** — MultiEstado simples (soma ponderada): refutado, 84.27% vs 90.26%
- **V2** — Gate Softmax: especializa L1 mas vaza ruído na L2, caiu para 62.1%
- **V3** — Gate MLP + Skip Connections: recuperou para 85.0%
- **V4** — Top-1 Sparse Routing: empate técnico com MLP (87.67% ± 1.15% vs 87.84% ± 1.36%), ~50% FLOPs
- **V4.1** — Load Balancing Loss: não melhorou (87.75% ± 1.16%)

---

## 2. Protocolo da Arena (Regras Fixas)

Toda arquitetura competidora deve usar:

| Parâmetro | Valor |
|-----------|-------|
| Dataset | MNIST completo (60k treino, 10k teste) |
| Seeds | 1-3 inicialmente |
| Épocas | 5 |
| Batch size | 128 |
| Learning rate | 0.01 (decay 0.97/epoch) |
| L2 | 1e-4 |
| Normalização | Z-score (mean/std do treino) |

**Métricas obrigatórias por run:**
- Accuracy final, loss final
- FLOPs estimados por amostra
- Accuracy por MFLOP
- Parâmetros
- Tempo de treino e inferência
- Entropia normalizada Layer 1 e Layer 2
- Distribuição de uso dos especialistas

**Sistema de pontuação:**
| Métrica | Peso |
|---------|------|
| Accuracy | 40 |
| FLOPs | 40 |
| Parâmetros | 10 |
| Tempo inferência | 10 |

---

## 3. Baselines (MNIST, seed 1, 5 épocas)

| Modelo | Hidden | Accuracy | FLOPs | Params | Acc/MFLOP |
|--------|--------|----------|-------|--------|-----------|
| MLP | 64 | 93.71% | 109.824 | 55.050 | 8.5327 |
| MLP | 128 | 93.80% | 236.032 | 118.282 | 3.9740 |
| MLP | 256 | 94.13% | 537.600 | 269.322 | 1.7509 |

**Melhores V4 Econômicas (seed 1):**
| Config | Accuracy | FLOPs | Acc/MFLOP |
|--------|----------|-------|-----------|
| V4 h128/s2/g8/no-skip | **93.92%** | 250.688 | 3.7465 |
| V4 h96/s2/g8/no-skip | 93.59% | **185.024** | **5.0583** |
| V4 h64/s2/g8/no-skip | 93.31% | 123.456 | 7.5582 |
| V4 h64/s2/g8/skip | 93.49% | 232.000 | 4.0297 |
| V4 h64/s4/g8/no-skip | 93.22% | 123.520 | 7.5470 |

**Padrão estrutural observado (recorrente em todos os experimentos):**
- Layer 1: entropia alta (0.90-0.99) — especialistas bem distribuídos
- Layer 2: frequentemente colapsada (entropia 0.03-0.73) — 1 especialista domina

---

## 4. Estrutura do Repositório

```
F:\neuronios quanticos\
├── arena/
│   ├── CHALLENGERS.md              ← V5-V9 especificados
│   ├── README.md                   ← Regras e ranking
│   ├── resultados/README.md        ← Template de saída
│   └── v5_competicao/train.py      ← Esboço inicial V5 (sklearn, não MNIST)
├── datasets/
│   └── mnist/                      ← Arquivos IDX (ignorados pelo git)
├── docs/
│   ├── pt-BR/
│   │   ├── conclusoes.md           ← Conclusões validadas
│   │   ├── diario.md               ← Diário completo (15 experimentos)
│   │   ├── HIPOTESES_REFUTADAS.md  ← H001-H004 refutadas
│   │   └── ROADMAP.md              ← E1-E9 planejados
│   └── en-US/                      ← Mesmo conteúdo em inglês
├── experimentos/
│   ├── e1_mnist/codigo/
│   │   ├── e1_mnist_v4.py          ← Código principal V4 (625 linhas)
│   │   ├── run_mnist_matrix.py     ← Runner matriz completa
│   │   ├── run_mnist_economic_matrix.py  ← Runner V4 econômica
│   │   └── consolidate_mnist_economic.py  ← Consolidador JSON→CSV
│   ├── v1_multistate/
│   ├── v2_gating/
│   ├── v3_skip_connections/
│   ├── v4_1_load_balancing/
│   └── v4_sparse_routing/
├── resultados_finais/
│   ├── e1_mnist_economic/          ← 30 JSONs + summary.json/csv + REPORT.md
│   ├── e1_mnist_matrix/            ← 6 JSONs + summary.json/csv + REPORT.md
│   └── output.txt
├── scripts/
│   ├── bateria_completa.py         ← 12 testes (30 seeds)
│   ├── common.py                   ← Layers, modelos, datasets, treino
│   ├── runner.py                   ← Runner sequencial
│   ├── sumario.py                  ← Sumarizador de resultados
│   └── download_mnist.py
├── rinha de arquiteturas.md        ← 669 linhas de discussão/protocolo
├── CONTEXT_PACKAGE.md              ← Este arquivo
├── README.md / README_EN.md
└── refactor_repo.py
```

---

## 5. Arquivos de Resultados

### `resultados_finais/e1_mnist_economic/summary.json`
Lista completa de runs + sumário agrupado. Cada run tem: model, seed, hidden, states, gate_hidden, use_skip, accuracy, final_loss, train_time_sec, inference_time_sec, flops_per_sample, params, accuracy_per_mflop, final_l1_entropy, final_l2_entropy.

### `resultados_finais/e1_mnist_economic/summary.csv`
CSV com médias por configuração (model, hidden, states, gate_hidden, use_skip → accuracy_mean, flops, etc.)

### `resultados_finais/e1_mnist_economic/runs.csv`
CSV com todas as linhas individuais de run.

### `resultados_finais/e1_mnist_economic/REPORT.md`
Relatório em markdown com análise e conclusões.

---

## 6. Desafiantes Pendentes (V5-V9)

### V5 — Competição Direta
**Ideia:** Remover gate externo. Cada especialista produz saída + score de confiança. O de maior score vence.
**Pergunta:** O gate separado é overhead desnecessário?
**Status:** Esboço inicial em `arena/v5_competicao/train.py` (dataset sklearn, não MNIST, pseudo-random)

### V6 — Top-2 Sparse Routing
**Ideia:** Usar 2 especialistas ativos em vez de 1.
**Pergunta:** Dois especialistas complementares melhoram accuracy o suficiente para justificar FLOPs extras?
**Status:** Não implementado.

### V7 — Árvore Hierárquica
**Ideia:** Gate raiz → grupo, gate local → especialista (árvore de decisão neural).
**Pergunta:** Roteamento hierárquico reduz custo do gate e melhora separação?
**Status:** Não implementado.

### V8 — Gate com Memória de Uso
**Ideia:** Penalizar especialistas muito usados, forcing diversificação.
**Pergunta:** Memória de uso reduz colapso da Layer 2 sem prejudicar accuracy?
**Status:** Não implementado.

### V9 — Especialistas Low-Rank
**Ideia:** Substituir matrizes densas W = A × B com fatores menores.
**Pergunta:** Especialistas compactos low-rank reduzem FLOPs/params mantendo accuracy?
**Status:** Não implementado. **Prioridade alta** (ataca overhead 784D).

---

## 7. Pipeline de Execução

### Rodar experimento individual:
```bash
python experimentos/e1_mnist/codigo/e1_mnist_v4.py --model v4 --epochs 5 --batch-size 128 --v4-hidden 96 --states 2 --gate-hidden 8 --seed 1 --out e1_mnist_economic/v4_h96_s2_g8_noskip_seed1.json
```

### Rodar matriz econômica:
```bash
python experimentos/e1_mnist/codigo/run_mnist_economic_matrix.py --seeds 1 --epochs 5 --dense-hidden 64,128 --v4-hidden 64,96,112,128 --states 2 --gate-hidden 4,6,8 --skip false
```

### Consolidar resultados:
```bash
python experimentos/e1_mnist/codigo/consolidate_mnist_economic.py
```

### Cálculo de FLOPs (V4):
```python
# V4 sparse FLOPs:
l1_gate = 2 * (input_dim * gate_hidden + gate_hidden * n_states)
l1_active = 2 * input_dim * hidden
l2_gate = 2 * (hidden * gate_hidden + gate_hidden * n_states)
l2_active = 2 * hidden * hidden
out = 2 * hidden * output_dim

# MLP dense FLOPs:
total = 2 * (input_dim * hidden + hidden * hidden + hidden * output_dim)
```

---

## 8. Roadmap de Experimentos (E1-E9)

| ID | Experimento | Status |
|----|-------------|--------|
| E1 | MNIST (V4 sobrevive?) | Parcial — seed 1 ok, falta múltiplas seeds |
| E2 | Curva Accuracy/FLOPs | Parcial — matriz econômica rodada, falta seeds 2-3 |
| E3 | Especialização por classe | Não iniciado |
| E4 | Fashion-MNIST | Não iniciado |
| E5 | Escala de estados (2-32) | Não iniciado |
| E6 | Curva de entropia por época | Logging implementado |
| E7 | Robustez a ruído | Não iniciado |
| E8 | Ponte para Transformers | Não iniciado |
| E9 | Arena V5-V9 | V5 esboçado, V6-V9 não implementados |

---

## 9. Observações Técnicas

- **Implementação:** NumPy puro, sem PyTorch/TensorFlow (por design)
- **Hardware testado:** Windows, CPU
- **Datasets:** MNIST IDX (download via `scripts/download_mnist.py`), datasets sintéticos (circles, moons, spirals) são gerados por funções em `scripts/common.py`
- **Semente:** `np.random.RandomState(seed)` para determinismo
- **Overhead V4:** Skip connections dominam FLOPs (dobra custo). Gate pequeno (4-8) vs gate grande (16-32) faz diferença significativa. 2 estados > 4 estados em economia.
- **Colapso L2:** Recorrente em todas as variações — pode ser propriedade intrínseca da arquitetura rasa (2 camadas)
