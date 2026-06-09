# Roadmap de Pesquisa — Roteamento Esparso e Mistura de Especialistas (V4+)

*Atualizado em: 2026-06-08 — após validação MNIST 10 seeds + escala de estados*

---

## Estado Atual (resumo executivo)

**V4 s=2/g=8/no-skip vs MLP 128 — 10 seeds MNIST:**

| Métrica | MLP 128 | MLP 235 (242k params) | V4 s=2 (242k params) |
|---------|---------|-----------------------|----------------------|
| Média   | 95.05%  | **95.19%**            | 94.78%               |
| Std     | ±0.15%  | ±0.10%                | ±0.36%               |
| FLOPs   | 236k    | 483k                  | **250k**             |
| Acc/MFLOP | 4.03  | 1.97                   | **3.79**             |

**Descobertas-chave já validadas:**

1. **V4 não supera MLP em accuracy** (−0.28pp vs MLP 128, −0.41pp vs MLP 235)
2. **Mas é 48% mais eficiente em FLOPs** que MLP de mesmo tamanho
3. **Temperatura não controla colapso** — regime de entropia é definido pela seed
4. **Seeds com colapso têm accuracy MELHOR** que seeds com entropia alta
5. **Mais estados piora accuracy** (s=2: 95.23% → s=16: 93.34%)
6. **Especialização real existe mas não melhora resultado**
7. **O problema é arquitetura, não roteamento** — limitação fundamental no top-1

---

## Próximos Passos (ordenados por prioridade)

### 1️⃣ Confirmar eficiência e estabilidade

**Pergunta:** O padrão de 48% menos FLOPs com ~0.4pp de perda se mantém em mais seeds?

- Rodar seeds 11–20 para V4 s=2 e MLP 235
- Testar V4 s=2 vs s=4 em configurações econômicas
- Medir std de accuracy e entropia L2
- **Métrica-chave:** Acc/MFLOP

### 2️⃣ Explorar trade-offs de gate e estados

**Pergunta:** Existe um ponto ótimo de accuracy vs eficiência?

- Temperatura (0.5, 1.0, 2.0) em seeds com entropia alta natural
- Escala de estados (2, 4, 8, 16) vs entropia L1/L2
- Registrar distribuição de especialistas por classe
- **Métrica-chave:** curva accuracy × FLOPs

### 3️⃣ Otimização da arquitetura V4 Econômica

**Pergunta:** Qual config minimiza perda de accuracy maximizando economia de FLOPs?

- Testar hidden ajustado para FLOPs balanceados
- Gate menor (4–8)
- Estados reduzidos (2–4)
- Sem skip
- **Objetivo:** encontrar o ponto ótimo na curva accuracy/FLOPs

### 4️⃣ Testes de generalização

**Pergunta:** A eficiência da V4 se mantém em outros datasets?

- Moons, Spirals, 20 Features (datasets sintéticos)
- MNIST completo ✅ (concluído)
- Fashion-MNIST
- CIFAR-10 (futuro)
- **Métrica-chave:** Acc/MFLOP cruzando datasets

### 5️⃣ Arena de arquiteturas (V5–V9)

**Pergunta:** Existe uma arquitetura melhor de roteamento/especialização?

| Challenger | Ideia central |
|------------|---------------|
| V5 | Especialistas competem sem gate externo |
| V6 | Top-2 sparse routing |
| V7 | Árvore hierárquica de gates |
| V8 | Gate com memória de uso |
| V9 | Especialistas low-rank |

**Regra:** mesmas seeds, dataset, épocas, batch, lr, l2.

### 6️⃣ Multi-agente / integração

**Pergunta:** A economia de FLOPs se traduz em ganho real em sistemas multi-agente?

- Cada especialista como um "agente" independente
- Medir economia real em pipelines locais
- Avaliar impacto em simulações de circuitos

### 7️⃣ Documentação e visualização

- Consolidar logs de entropia, distribuição, Acc/MFLOP
- Gráficos: Accuracy vs FLOPs, entropia por camada
- Diários PT-BR/EN-US prontos para publicação

---

## Experimentos concluídos

| Experimento | Status | Resultado principal |
|------------|--------|---------------------|
| E1 — MNIST 10 seeds V4 s=2 | ✅ | 94.78% ±0.36% vs MLP 95.05% ±0.15% |
| E2 — Temperatura | ✅ | Não altera regime de entropia |
| E3 — Controle de parâmetros | ✅ | V4 empata em accuracy com MLP de mesmo tamanho, mas gasta metade dos FLOPs |
| E4 — Escala de estados | ✅ | Mais estados → pior accuracy (s=2: 95.23% → s=16: 93.34%) |
| E5 — Especialização por classe | ✅ | Existe mas não melhora accuracy |

---

## Hipóteses refutadas

1. ~~"Mais especialistas = melhor accuracy"~~ → Falso. Mais estados piora.
2. ~~"Temperatura controla colapso do gate"~~ → Falso. Regime é definido pela seed.
3. ~~"V4 ganha performance com mais parâmetros"~~ → Parcial. Igual参ados, empata em accuracy; o ganho real é eficiência.

## Hipóteses em teste

1. "V4 econômica (s=2, gate pequeno) maximiza Acc/MFLOP"
2. "A eficiência se mantém em outros datasets"
3. "V5–V9 podem superar V4 em accuracy ou eficiência"
