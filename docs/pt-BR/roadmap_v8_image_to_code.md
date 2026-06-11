# Roadmap V8: Image-to-Code MoE

## O Novo Paradigma
Após esgotar os limites da arquitetura em datasets sintéticos e de classificação de imagens (V1 ao V7), o laboratório entra em sua fase mais ambiciosa: testar a hipótese da **Heterogeneidade Funcional** em problemas generativos e multimodais do mundo real.

A pergunta científica agora não é mais "se a heterogeneidade classifica melhor", mas sim:
> "A heterogeneidade estrutural induz especialização funcional em tarefas complexas de Image-to-Code?"

## A Hipótese Central
**"Experts heterogêneos são superiores em tarefas multimodais de Image-to-Code sob orçamento computacional equivalente."**

A premissa é que, ao desenhar uma interface de usuário (UI) a partir de uma imagem, o modelo necessita de diferentes níveis de abstração cognitiva. Em um MoE Heterogêneo, espera-se que o Roteador crie uma alocação semântica pura:
* **Expert Pequeno (ex: 16 neurônios):** Cores, fontes e estilo.
* **Expert Médio (ex: 64 neurônios):** Layout e grid.
* **Expert Grande (ex: 256 neurônios):** Componentes estruturais (botões, inputs).
* **Expert Massivo (ex: 1024 neurônios):** Relações globais e hierarquia do DOM.

## O Benchmark de Validação

### Controle: MoE Homogêneo (O Padrão da Indústria)
* 8 Experts idênticos.
* Hidden sizes: `[128, 128, 128, 128, 128, 128, 128, 128]`

### Variável: MoE Heterogêneo (A Teoria do Laboratório)
* 8 Experts estruturalmente diversos.
* Hidden sizes: `[16, 32, 64, 128, 256, 512, 1024, 2048]`

**(Ambos devem possuir FLOPs totais equivalentes e utilizar o mesmo roteamento Top-K)**.

## O Dataset
A transição exigirá a construção de pipelines para datasets multimodais públicos (ex: WebSight, Pix2Code, ou scrapings controlados de Tailwind/React). O input é uma imagem (Screenshot) e o target é a sequência de tokens estruturados (HTML/React).

## As Novas Métricas de Avaliação
Loss não é mais suficiente. A validação será feita através de:
1. **Similaridade Visual:** Renderização do output vs Screenshot original.
2. **Similaridade Estrutural:** AST do DOM HTML (ex: `<section>` vs `<div>`).
3. **Taxa de Compilação:** O código sintático quebra ou passa no build (ex: sintaxe React/Vue válida).
4. **Métricas de Dinâmica de Roteamento (As métricas originais do lab):**
    * Gini Index
    * Expert Redundancy Index (ERI)
    * Routing Stability (RS)
    * **Mapeamento Semântico de Ativação:** Qual expert foi ativado ao gerar um token de cor vs um token de hierarquia de tag.

## O Grande Objetivo
Construir dois legados simultâneos:
1. Um produto real de engenharia: um gerador de código visual state-of-the-art baseado em arquitetura esparsa.
2. Evidência empírica incontestável sobre como a especialização brota a partir da heterogeneidade física de um sistema de rede neural.
