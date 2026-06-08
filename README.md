# Neurônios MultiEstado – Pesquisa em Roteamento Esparso

*Read this in [English](README_EN.md).*

> **Disclaimer:** Não estou fazendo nenhuma afirmação acadêmica ou descoberta definitiva. Sou apenas um cientista da computação criando e explorando novas ideias por curiosidade. Tudo aqui é puramente experimental.

**Descrição:**  
Este projeto explora a hipótese de que arquiteturas com especialistas internos compactos, combinadas com roteamento inteligente (Sparse Routing) e skip connections, conseguem reproduzir o desempenho de redes neurais tradicionais (MLPs) exigindo um custo computacional significativamente menor (FLOPs reduzidos). O projeto evoluiu da ideia exploratória de "neurônios com múltiplos estados" para um micro-framework de Mistura de Especialistas (MoE).

**Estrutura do repositório:**
- [`experimentos/`](./experimentos/) → Códigos e resultados isolados por versão (V1, V2, V3, V4, V4.1)
- [`resultados_finais/`](./resultados_finais/) → Logs consolidados, gráficos e tabelas de performance
- [`docs/`](./docs/) → Relatórios, conclusões validadas e diário de pesquisa (PT-BR / EN-US)
- [`datasets/`](./datasets/) → Datasets utilizados nos benchmarks (Moons, Spirals, Circles)
- [`scripts/`](./scripts/) → Scripts para executar baterias de testes e validação cruzada

**Status da pesquisa:**  
⚠️ **Experimental**  
Testes de robustez em 30 seeds demonstraram um **empate técnico** em acurácia entre a V4 (Sparse Routing Top-1) e o MLP tradicional em múltiplos domínios complexos. No entanto, a arquitetura V4 alcança esse limite consumindo **aproximadamente 50% dos FLOPs** na inferência, pois o roteamento elimina com sucesso a ativação e o vazamento de estados não utilizados (ruído).

**Como executar:**  
- Rodar `python scripts/bateria_completa.py` ou os scripts específicos dentro de `experimentos/`.
- Ver [`docs/pt-BR/diario.md`](./docs/pt-BR/diario.md) e [`docs/pt-BR/HIPOTESES_REFUTADAS.md`](./docs/pt-BR/HIPOTESES_REFUTADAS.md) para relatórios detalhados, evolução das hipóteses e dados de ablação.
