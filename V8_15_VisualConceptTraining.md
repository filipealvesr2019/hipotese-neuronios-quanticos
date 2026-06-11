
Iniciando Treinamento (10 Épocas)...
Época 1 | Batch 0 | Loss: 3530.6575
Época 2 | Batch 0 | Loss: 205.4782
Época 3 | Batch 0 | Loss: 41.9917
Época 4 | Batch 0 | Loss: 138.0528
Época 5 | Batch 0 | Loss: 154.2118
Época 6 | Batch 0 | Loss: 59.7420
Época 7 | Batch 0 | Loss: 6.5567
Época 8 | Batch 0 | Loss: 45.2187
Época 9 | Batch 0 | Loss: 59.0157
Época 10 | Batch 0 | Loss: 16.6901

Extraindo Matrizes de Especialização Visual...

============================================================
RELATÓRIO ESTATÍSTICO: POR COMPONENT
============================================================

[NAV] (22845 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 75.0%
  -> Expert 2 (64n): 17.2%
  -> Expert 3 (128n): 0.1%
  -> Expert 4 (256n): 7.6%

[FOOTER] (25097 tokens):
  -> Expert 1 (32n): 36.3%
  -> Expert 2 (64n): 60.4%
  -> Expert 3 (128n): 0.3%
  -> Expert 4 (256n): 2.9%

[HEADER] (7311 tokens):
  -> Expert 1 (32n): 86.7%
  -> Expert 3 (128n): 0.1%
  -> Expert 4 (256n): 13.2%

[FORM] (872 tokens):
  -> Expert 1 (32n): 57.5%
  -> Expert 2 (64n): 27.2%
  -> Expert 3 (128n): 0.9%
  -> Expert 4 (256n): 14.4%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_component_heatmap.png

============================================================
RELATÓRIO ESTATÍSTICO: POR THEME
============================================================

[LIGHT] (56125 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 59.0%
  -> Expert 2 (64n): 34.5%
  -> Expert 3 (128n): 0.2%
  -> Expert 4 (256n): 6.4%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_theme_heatmap.png

============================================================
RELATÓRIO ESTATÍSTICO: POR STYLE
============================================================

[FLAT] (51483 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 55.7%
  -> Expert 2 (64n): 37.6%
  -> Expert 3 (128n): 0.1%
  -> Expert 4 (256n): 6.7%

[GLASSMORPHISM] (4642 tokens):
  -> Expert 1 (32n): 95.5%
  -> Expert 3 (128n): 1.6%
  -> Expert 4 (256n): 3.0%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_style_heatmap.png

============================================================
RELATÓRIO ESTATÍSTICO: POR LAYOUT
============================================================

[GRID] (52700 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 57.1%
  -> Expert 2 (64n): 36.3%
  -> Expert 3 (128n): 0.2%
  -> Expert 4 (256n): 6.5%

[HORIZONTAL] (2553 tokens):
  -> Expert 1 (32n): 98.7%
  -> Expert 4 (256n): 1.3%

[VERTICAL] (872 tokens):
  -> Expert 1 (32n): 57.5%
  -> Expert 2 (64n): 27.2%
  -> Expert 3 (128n): 0.9%
  -> Expert 4 (256n): 14.4%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_layout_heatmap.png

============================================================
RELATÓRIO ESTATÍSTICO: POR FRAMEWORK
============================================================

[NEXTJS] (14353 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 58.8%
  -> Expert 2 (64n): 34.3%
  -> Expert 3 (128n): 0.2%
  -> Expert 4 (256n): 6.7%

[HTML] (13792 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 59.0%
  -> Expert 2 (64n): 34.6%
  -> Expert 3 (128n): 0.2%
  -> Expert 4 (256n): 6.3%

[REACT] (14188 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 59.2%
  -> Expert 2 (64n): 34.4%
  -> Expert 3 (128n): 0.2%
  -> Expert 4 (256n): 6.2%

[TAILWIND] (13792 tokens):
  -> Expert 0 (16n): 0.0%
  -> Expert 1 (32n): 58.9%
  -> Expert 2 (64n): 34.6%
  -> Expert 3 (128n): 0.2%
  -> Expert 4 (256n): 6.3%
============================================================
[Artefato Crítico] Heatmap gerado em: graficos/expert_framework_heatmap.png

filipe@eufilip MINGW64 /f/neuronios quanticos (main)
$




Percebo o padrão que está surgindo, e ele confirma exatamente o que eu suspeitava sobre os efeitos do seu dataset “luxo” no roteamento:

---

### Observações imediatas

1. **Expert dominante**

   * O Expert **1 (32 neurônios)** está absolutamente dominando quase todos os conceitos:

     * **Componentes:** Header 86%, Nav 75%, Footer 36%, Form 57%
     * **Theme / Style / Layout / Framework:** 55–59%
   * Isso significa que o roteador está **encostando no Expert 1 como default**, mesmo com variações visuais ou de layout.

2. **Experts secundários**

   * Expert 2 (64n) atua como “backup” em alguns casos de maior complexidade (Footer 60%, Grid 36%).
   * Expert 4 (256n) só aparece em porcentagens baixíssimas (1–13%), praticamente ignorado.

3. **Ausência quase total do Expert 0 (16n)**

   * Ele não foi acionado em quase nada. Isso é esperado: neurônios muito pequenos não têm capacidade de representar padrões complexos de layout + tema + style.

4. **Especialização visual fraca**

   * Apesar de você ter cruzado **Theme/Style/Layout**, não há sinais claros de que o roteamento aprendeu conceitos visuais independentes.
   * Exemplo: Grid, Horizontal e Vertical são acionados pelo mesmo Expert 1 na maior parte das vezes, indicando que a rede ainda está priorizando **estrutura e tokens dominantes**, não “conceito visual”.

---

### Interpretação

* O modelo está **memorizing patterns grandes** (Header grande, Form médio, Footer com grids) em vez de realmente separar **conceitos visuais e layout**.
* O dataset “luxo” aumentou a complexidade, mas não criou ainda **contextos ricos o suficiente** para forçar os experts a se especializarem visualmente.
* Ou seja: o roteador está **fazendo compressão estatística**, usando o Expert 1 como “curinga”, porque ele consegue lidar com a maior parte do dataset sem precisar diversificar.

---

### Próximos passos estratégicos

1. **Forçar diversidade de contexto**

   * Criar amostras onde a mesma hierarquia de DOM aparece em **vários estilos visuais** **exatamente iguais**, mudando apenas `Theme/Style/Layout`.
   * Isso força a rede a aprender que **apenas o visual muda**, não a estrutura.

2. **Balanceamento de tokens por Expert**

   * Pode ser necessário **reduzir o número de amostras do Expert 1** no treinamento ou usar **Noisy Routing / Load Balancing** mais agressivo, para que Experts 2–4 realmente aprendam padrões diferentes.

3. **Aumentar granularidade**

   * Adicionar **metadados estruturais**: número de `<div>`, `<button>`, profundidade de árvore DOM, grid/colunas.
   * Isso permitirá analisar se o Expert 2 ou 4 está aprendendo **complexidade** ao invés de **conceito visual**.

4. **Foco visual**

   * Criar **datasets apenas de tema**: pegar o mesmo Header, renderizar Dark/Light/Brand, Flat/Glass/Minimal, Horizontal/Grid.
   * Avaliar se o Expert muda quando o **conceito visual** muda sem mudar estrutura.

---









