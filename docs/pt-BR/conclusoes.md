# Conclusões Atuais

*Somente fatos que sobreviveram a múltiplos experimentos e validação com 30 seeds.*

---

## Status da Pesquisa

**Hipótese original:** neurônios MultiEstado são intrinsecamente superiores a neurônios tradicionais.  
**Status:** refutada.

**Hipótese revisada:** especialistas compactos com roteamento esparso Top-1 podem manter o desempenho de redes densas reduzindo computação na inferência.  
**Status:** parcialmente confirmada em datasets pequenos.

**Formulação mais honesta:** a hipótese de superioridade do neurônio MultiEstado foi refutada; a hipótese mais restrita de computação condicional eficiente via especialistas esparsos recebeu evidência experimental positiva em problemas pequenos.

---

✅ **Estados aprendem representações diferentes.**
Correlação entre estados ≈ 0.01. Eles não se tornam cópias uns dos outros. Confirmado no Teste 12.

✅ **O Gate consegue especializar o roteamento.**
Na Layer 1, o gate distribui o tráfego de forma saudável entre os especialistas. Confirmado na V2, V3 e V4.

✅ **Skip Connections resolvem o colapso de treinamento.**
Sem skip connections (V2), a performance caiu para 62.1%. Com elas (V3+), retornou a ~85%. Confirmado experimentalmente.

✅ **Sparse Routing (Top-1) elimina o vazamento de ruído.**
No V3, remover um estado *melhorava* a performance (+2.75pp), indicando vazamento de ruído. No V4, o mesmo estado tem impacto de 0.00pp — foi perfeitamente silenciado pelo roteamento esparso.

✅ **A arquitetura V4 empata com o MLP Tradicional em múltiplos domínios.**
Validado com 30 seeds em Círculos, Moons, Spirals e dataset de 20 features.

✅ **V4 consome ~50% dos FLOPs na inferência.**
MLP Tradicional: ~4.350 operações. V4 Sparse: ~2.080 operações. O roteamento Top-1 ativa apenas 1 especialista por camada, evitando cálculos desnecessários.

---

⚠️ **Não há evidência de ganho absoluto de acurácia.**
A arquitetura iguala, mas não supera consistentemente o MLP nos datasets testados.

⚠️ **Mode Collapse nas camadas mais profundas é persistente.**
A Layer 2 tende a concentrar 80-100% do tráfego em um único especialista, independente de Softmax (V3) ou Top-1 (V4).

⚠️ **Load Balancing Loss não resolveu o colapso em redes rasas.**
A penalidade auxiliar (V4.1) não melhorou a accuracy nem diversificou o uso dos especialistas de forma sustentada.

⚠️ **Sem evidência de escalabilidade para LLMs.**
Todos os testes foram em redes rasas (2 camadas) e datasets pequenos. A hipótese de que a arquitetura funciona em Transformers ainda é um experimento futuro.

⚠️ **MNIST preliminar ainda não confirma a hipótese forte.**
Em single seed, a V4 ficou próxima do MLP (92.97% vs 93.80%), mas com redução pequena de FLOPs estimados (~1.5%). O resultado é promissor como sobrevivência ao MNIST, mas ainda exige curva Accuracy/FLOPs e múltiplas seeds.

---

## Avaliação de Maturidade

**Hipótese original:** 0% viva.  
Foi refutada pela V1 e pelos testes subsequentes.

**Arquitetura V4:** promissora.  
O padrão V1 → V2 → V3 → V4 forma uma narrativa técnica coerente: cada versão resolveu um gargalo observado na anterior.

**Evidência de vantagem prática:** ainda limitada.  
Há redução forte de FLOPs em problemas pequenos, mas o MNIST preliminar ainda não mostrou economia substancial.

**Evidência de fenômeno real:** forte.  
Especialização de especialistas, roteamento efetivo e colapso recorrente da Layer 2 aparecem em múltiplos domínios. A questão aberta é se isso escala.

**Prioridade atual:** fechar MNIST com curva Accuracy/FLOPs antes de avançar para CIFAR.

**Atualização E2 MNIST:** a primeira matriz Accuracy/FLOPs em seed 1 não favoreceu a V4. O MLP 64 foi mais preciso e mais barato que V4 64/128/256. Isso sugere que a versão atual da V4 é eficiente em problemas pequenos, mas paga overhead demais em MNIST.
