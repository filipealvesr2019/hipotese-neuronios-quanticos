
python experimentos/V8_9_MultiLangTrainer.py

Iniciando Treinamento (10 Épocas com Alta Penalidade de Colapso)...
Época 1 | Batch 0 | CE Loss: 6.5505 | Bal Loss: 614.9697
Época 1 | Batch 20 | CE Loss: 5.6037 | Bal Loss: 9.1774
Época 2 | Batch 0 | CE Loss: 5.0404 | Bal Loss: 0.9669
Época 2 | Batch 20 | CE Loss: 4.8853 | Bal Loss: 1.8060
Época 3 | Batch 0 | CE Loss: 4.6543 | Bal Loss: 1.4180
Época 3 | Batch 20 | CE Loss: 4.6828 | Bal Loss: 5.3832
Época 4 | Batch 0 | CE Loss: 4.6576 | Bal Loss: 1.6825
Época 4 | Batch 20 | CE Loss: 4.3705 | Bal Loss: 0.7585
Época 5 | Batch 0 | CE Loss: 4.3225 | Bal Loss: 1.4996
Época 5 | Batch 20 | CE Loss: 4.1804 | Bal Loss: 2.6774
Época 6 | Batch 0 | CE Loss: 4.0276 | Bal Loss: 1.3489
Época 6 | Batch 20 | CE Loss: 3.6854 | Bal Loss: 4.9034
Época 7 | Batch 0 | CE Loss: 3.6512 | Bal Loss: 8.3433
Época 7 | Batch 20 | CE Loss: 3.3987 | Bal Loss: 0.6155
Época 8 | Batch 0 | CE Loss: 3.2176 | Bal Loss: 0.4462
Época 8 | Batch 20 | CE Loss: 2.9828 | Bal Loss: 7.0238
Época 9 | Batch 0 | CE Loss: 2.9634 | Bal Loss: 1.6605
Época 9 | Batch 20 | CE Loss: 2.8585 | Bal Loss: 1.2528
Época 10 | Batch 0 | CE Loss: 2.7859 | Bal Loss: 0.9436
Época 10 | Batch 20 | CE Loss: 2.3899 | Bal Loss: 0.2623

Extraindo Matrizes de Especialização Visual e Estrutural...

============================================================
RELATÓRIO ESTATÍSTICO: POR COMPONENT
============================================================

[FOOTER] (22059 tokens):
  -> Expert 1 (32n): 93.5%
  -> Expert 2 (64n): 0.4%
  -> Expert 3 (128n): 4.7%
  -> Expert 4 (256n): 1.5%

[CARD] (35775 tokens):
  -> Expert 1 (32n): 93.5%
  -> Expert 2 (64n): 0.4%
  -> Expert 3 (128n): 4.7%
  -> Expert 4 (256n): 1.4%

[HEADER] (16875 tokens):
  -> Expert 1 (32n): 96.2%
  -> Expert 2 (64n): 0.1%
  -> Expert 3 (128n): 2.1%
  -> Expert 4 (256n): 1.6%

[HERO] (13635 tokens):
  -> Expert 0 (16n): 7.9%
  -> Expert 1 (32n): 38.9%
  -> Expert 2 (64n): 24.1%
  -> Expert 3 (128n): 28.5%
  -> Expert 4 (256n): 0.7%

[FORM] (18063 tokens):
  -> Expert 1 (32n): 79.8%
  -> Expert 2 (64n): 1.2%
  -> Expert 3 (128n): 16.3%
  -> Expert 4 (256n): 2.6%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_ultralux_component.png

============================================================
RELATÓRIO ESTATÍSTICO: POR THEME
============================================================

[LIGHT] (35469 tokens):
  -> Expert 0 (16n): 0.8%
  -> Expert 1 (32n): 86.1%
  -> Expert 2 (64n): 2.5%
  -> Expert 3 (128n): 9.4%
  -> Expert 4 (256n): 1.1%

[BRAND] (35469 tokens):
  -> Expert 0 (16n): 1.4%
  -> Expert 1 (32n): 86.7%
  -> Expert 2 (64n): 2.7%
  -> Expert 3 (128n): 7.9%
  -> Expert 4 (256n): 1.3%

[DARK] (35469 tokens):
  -> Expert 0 (16n): 0.8%
  -> Expert 1 (32n): 81.1%
  -> Expert 2 (64n): 5.3%
  -> Expert 3 (128n): 10.6%
  -> Expert 4 (256n): 2.2%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_ultralux_theme.png

============================================================
RELATÓRIO ESTATÍSTICO: POR STYLE
============================================================

[GLASSMORPHISM] (35469 tokens):
  -> Expert 0 (16n): 1.4%
  -> Expert 1 (32n): 83.0%
  -> Expert 2 (64n): 4.8%
  -> Expert 3 (128n): 9.1%
  -> Expert 4 (256n): 1.7%

[FLAT] (35469 tokens):
  -> Expert 0 (16n): 1.6%
  -> Expert 1 (32n): 78.8%
  -> Expert 2 (64n): 5.1%
  -> Expert 3 (128n): 13.2%
  -> Expert 4 (256n): 1.3%

[MINIMAL] (35469 tokens):
  -> Expert 1 (32n): 92.1%
  -> Expert 2 (64n): 0.6%
  -> Expert 3 (128n): 5.6%
  -> Expert 4 (256n): 1.7%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_ultralux_style.png

============================================================
RELATÓRIO ESTATÍSTICO: POR DEPTH_DOM
============================================================

[PROFUNDO (5+)] (106407 tokens):
  -> Expert 0 (16n): 1.0%
  -> Expert 1 (32n): 84.6%
  -> Expert 2 (64n): 3.5%
  -> Expert 3 (128n): 9.3%
  -> Expert 4 (256n): 1.5%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_ultralux_depth_dom.png

============================================================
RELATÓRIO ESTATÍSTICO: POR NUM_TAGS
============================================================

[GRANDE (16+)] (75897 tokens):
  -> Expert 1 (32n): 90.3%
  -> Expert 2 (64n): 0.6%
  -> Expert 3 (128n): 7.5%
  -> Expert 4 (256n): 1.7%

[MÉDIO (6-15)] (30510 tokens):
  -> Expert 0 (16n): 3.5%
  -> Expert 1 (32n): 70.6%
  -> Expert 2 (64n): 10.8%
  -> Expert 3 (128n): 13.9%
  -> Expert 4 (256n): 1.2%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_ultralux_num_tags.png

filipe@eufilip MINGW64 /f/neuronios quanticos (main)
$

Ah, agora ficou cristalino. 😅

O que está acontecendo aqui é o **colapso de roteamento do Expert 1**. Ele está literalmente “engolindo” quase todo o dataset. Resultado:

* Expert 1 domina **90%+ de todos os tokens** nos componentes complexos, themes, styles e DOM profundo.
* Os Experts “betinhas” (0, 2, 3, 4) mal recebem amostras suficientes para se especializar.
* Bal Loss baixo no final indica que o roteador está satisfeito com essa alocação, mesmo sendo desequilibrada.

### Por que isso acontece

1. **Expert 1 tem 32 neurônios**, uma capacidade “intermediária”, perfeita para capturar quase todo o padrão sem dificuldade.
2. **Dataset sintético ultra-luxo** tem muita repetição estrutural e visual, mesmo com variantes, então o roteador percebe que Expert 1 consegue “resolver” a maior parte do trabalho sozinho.
3. **Penalidade de colapso insuficiente**: mesmo com alta penalidade, o roteador prefere concentrar tokens em um expert consolidado para minimizar CE Loss, sacrificando diversidade.

### Estratégias para balancear

1. **Forçar balanceamento do roteamento**:

   * Adicionar `entropy_regularization` maior no roteador.
   * Usar `load_balance_loss` mais pesado durante o treinamento.

2. **Fragmentar o dataset por complexidade**:

   * Separar por depth DOM, número de tags, ou estilo visual.
   * Forçar que diferentes Experts recebam diferentes “clusters” de complexidade.

3. **Especialização deliberada**:

   * Pré-atribuir grupos de tokens para cada Expert (“Expert 0 = Headers simples”, “Expert 2 = Footers”, “Expert 3 = Cards grandes”), para quebrar o monopólio.

4. **Diversificação estrutural**:

   * Criar mais variantes de DOM profundo, mixando layouts e estilos, para que **nenhum Expert consiga capturar 90% sozinho**.
   * Ex.: combinando horizontal + vertical + grid + glassmorphism em cada componente.



Perfeito. Vamos montar o **Ultra-Luxo 2.0**, desenhado para obrigar os Experts a se especializar de forma equilibrada, sem perder fidelidade visual ou complexidade de DOM. Aqui está o plano:

---

## 1️⃣ Estratégia de Roteamento Balanceado

1. **Clusterização por complexidade**

   * Dividimos os 10 esqueletos (Headers, Footers, Cards, Forms, Navbars) em **subgrupos**:

     * Simples: Depth 3-4, tags ≤ 8 → Expert 0
     * Médio: Depth 5-6, tags 9-12 → Expert 1
     * Complexo: Depth 6-7, tags 13-18 → Expert 2
     * Ultra: Depth 7-8, tags > 18 → Expert 3
     * Variantes visuais: Expert 4

2. **Forçar diversificação**

   * Para cada batch, limitamos a probabilidade máxima de roteamento para qualquer Expert a **40%**.
   * Loss de balanceamento (`Bal Loss`) é aumentado dinamicamente se um Expert ultrapassar 40% de tokens por batch.

3. **Especialização visual**

   * Themes/Styles/Layout são distribuídos propositalmente:

     * Expert 0 → minimal + light
     * Expert 1 → flat + brand
     * Expert 2 → glassmorphism + dark
     * Expert 3 → grid layouts complexos
     * Expert 4 → casos mistos / borda experimental

---

## 2️⃣ Dataset Ultra-Luxo 2.0

1. **Criação de Templates Sintéticos Puro Tailwind**

   * Mantemos **DOM fixo** para cada componente, mas variamos:

     * `theme_bg`, `theme_text`, `style_border`, `style_glass`, `layout_direction`
   * Cada template gera **6 variantes** (2x Theme × 3x Style × 1 Layout)
   * Total: 10 esqueletos × 6 variantes × 5 Experts → 300 samples por Expert.

2. **Fragmentação artificial**

   * Para Depths e número de tags, cada Expert recebe batches separados de acordo com cluster acima.
   * Evita que Expert 1 “engula” tudo.

3. **Aumento de diversidade**

   * Shuffle de imagens, ícones e textos.
   * Randomização de `padding`, `margin` e `grid-cols` para quebrar padrões fáceis.

---

## 3️⃣ Treinamento

* **Hyperparams**:

  ```python
  entropy_regularization = 0.5  # força roteador a explorar mais Experts
  load_balance_loss = 5.0        # alto para evitar monopolização
  max_expert_prob = 0.4          # não mais de 40% de tokens por Expert
  ```
* **Batch composition**: cada batch deve conter pelo menos 1 token de cada cluster, garantindo que todos os Experts sejam acionados.

---

## 4️⃣ Resultados Esperados

* Heatmaps de Experts devem mostrar **distribuição quase uniforme**:

  * Expert 0 → Headers simples + minimal
  * Expert 1 → Forms e Cards médios
  * Expert 2 → Footers e Cards complexos
  * Expert 3 → Heroes e Grid layouts profundos
  * Expert 4 → Misturas/variantes

* CE Loss total ainda deve diminuir, mas agora **Bal Loss será alto se qualquer Expert tentar monopolizar**, garantindo justiça na especialização.

---
