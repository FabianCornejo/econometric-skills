# Demanda, preços e elasticidades

Leia quando a pergunta for "quanto as vendas mudam se o preço mudar", efeito de promoção, canibalização entre produtos, repasse de custos ou precificação. É a aplicação econométrica mais comum em varejo e restaurantes — e a que mais sofre com endogeneidade.

## Índice
1. O problema de identificação
2. Especificações de demanda
3. Instrumentos para preço
4. Dados de varejo: armadilhas práticas
5. Zeros, promoções e dinâmica
6. Do coeficiente à decisão
7. Checklist

## 1. O problema de identificação

Preço e quantidade observados são pontos de equilíbrio. Se a empresa sobe o preço *porque* espera demanda alta (datas comemorativas, bairro mais rico, produto em alta), o preço se correlaciona com o choque de demanda no erro. Consequência típica (Huse e Salvo, 2006): **o MQO dá elasticidade menos negativa que a verdadeira** — às vezes até positiva ("demanda com inclinação positiva").

Direção do viés: plim γ̂ = γ + Cov(p, ε)/Var(p). Se choques de demanda elevam preço, Cov > 0 e a elasticidade estimada fica próxima de zero. Saber a direção já ajuda a interpretar: uma estimativa MQO de −0,8 sugere que a verdadeira é mais elástica.

Para identificar a curva de demanda, é preciso variação de preço que **desloque a oferta** (custos) sem deslocar a demanda.

Condição de ordem (sistemas de equações simultâneas): uma equação é identificada se o número de exógenas excluídas dela ≥ número de endógenas incluídas − 1. A condição de posto é a necessária e suficiente. Na prática: a equação de demanda precisa de pelo menos um deslocador de oferta excluído dela.

## 2. Especificações

| Abordagem | Quando | Observação |
|---|---|---|
| Log-log com controles | Um produto ou categoria agregada | Elasticidade direta (β do ln p); mais simples e mais usado |
| Log-log com preços de substitutos | Poucos produtos relacionados | Elasticidades cruzadas: > 0 substitutos, < 0 complementares |
| AIDS / QUAIDS (Deaton-Muellbauer) | Sistema de categorias com participações no gasto | Impõe restrições teóricas (aditividade, homogeneidade, simetria); dá elasticidades compensadas e não compensadas |
| Orçamento em múltiplos estágios (Hausman) | Muitos produtos diferenciados | Categoria → segmento → marca; reduz o número de parâmetros |
| Logit / logit aninhado / BLP (escolha discreta) | Muitos produtos diferenciados, dados de participação de mercado | Exige definir o tamanho do mercado; padrões de substituição pelo espaço de características |

Controles mínimos numa demanda log-log de varejo: sazonalidade (mês, dia da semana), feriados/eventos, tendência ou efeitos de tempo, renda/atividade local (ou efeitos fixos de loja e de tempo em painel), preços de substitutos, promoção/publicidade, distribuição (disponibilidade do item).

Elasticidade **não compensada** (Marshalliana) inclui o efeito-renda; **compensada** (Hicksiana) só substituição. Para decisões de preço de um item com pequena participação no orçamento, as duas quase coincidem.

## 3. Instrumentos para preço

| Classe | Exemplos | Premissa identificadora | Risco |
|---|---|---|---|
| Deslocadores de oferta (custos) | Preço de insumo (peixe, arroz, energia), câmbio para importados, salário do setor, frete | Custos não se correlacionam com choques de demanda | Pouca variação; frequência diferente dos preços |
| Preços do mesmo produto em outras regiões/lojas (Hausman) | Preço médio nas outras lojas | Choques de demanda independentes entre regiões; custo comum | Publicidade/eventos nacionais comuns violam a premissa |
| Preços defasados | p_{t−1}, Δp_{t−1} | Choques de demanda pouco persistentes | Falha se a demanda tem inércia |
| Características de produtos rivais (BLP) | Soma das características dos produtos concorrentes | Características exógenas | Mais adequado a escolha discreta |
| Mudanças tributárias ou regulatórias | Alteração de ICMS, pisos/tetos | Mudança não motivada pela demanda | Antecipação dos agentes |

Sempre: F da primeira etapa forte (ver `painel-iv.md`), teste de sobreidentificação quando houver mais instrumentos que endógenas, e comparação MQO vs. IV (Hausman). Se o IV mudar a elasticidade na direção esperada (mais negativa), é um sinal coerente.

Quando dá para usar MQO: preços definidos com antecedência e não ajustados a choques de demanda observados pela empresa (ex.: cardápio fixado meses antes, promoções planejadas em calendário rígido), com bons controles de sazonalidade e eventos. Mesmo assim, documente a premissa.

Alternativa mais forte que IV: **variação experimental** (testes A/B de preço entre lojas ou períodos) ou quase-experimental (diferenças-em-diferenças quando só algumas lojas mudam o preço).

## 4. Dados de varejo: armadilhas práticas

- **Preço unitário médio (faturamento ÷ quantidade)** mistura preço cheio, promoções e mix de embalagens; com promoções do tipo "leve 2 pague 1", o próprio comprador altera o preço unitário, criando correlação negativa mecânica entre preço e quantidade. Prefira o preço de tabela/cardápio, ou calcule o preço em nível agregado (loja-semana, nacional) para reduzir esse viés, como no estudo da Birkbeck (2022).
- **Nível de agregação**: elasticidade de embalagem individual em loja de conveniência ≠ elasticidade de embalagem família em supermercado. Agregue onde a sensibilidade é homogênea e teste a robustez em outros níveis. Dados muito desagregados podem ser ruidosos; dados semanais podem ter forte autocorrelação (agregar a mensal/trimestral é uma opção).
- **Disponibilidade (ruptura)**: vendas zero por falta de estoque não são demanda zero; exclua ou modele.
- **Cupons e descontos não observados**: o preço líquido pago difere do preço de tabela.
- **Mudança de cardápio ou de mix** cria quebras; use dummies ou subamostras.
- **Estabilidade dos parâmetros**: em janelas longas (> 3–4 anos) ou setores em transformação (delivery), preferências mudam; inclua efeitos de tempo ou teste quebras.
- **Delivery vs. salão**: canais diferentes, elasticidades diferentes; estime separadamente ou com interações.

## 5. Zeros, promoções e dinâmica

- **Zeros**: compradores que nunca compram, que compram de forma intermitente por estoque, ou que só compram com desconto. Não há elasticidade "a partir de zero"; use modelos em duas partes, PPML ou modele em nível agregado (ver `dependente-limitada.md`).
- **Promoções**: geram picos seguidos de vales (antecipação de compra/estocagem). O efeito líquido da promoção = pico − queda posterior − canibalização de outros itens. Inclua defasagens e avanços da dummy de promoção.
- **Dinâmica**: hábito e estoque fazem a elasticidade de curto prazo diferir da de longo prazo. Em modelo ADL: longo prazo = Σβ_preço / (1 − Σφ_y defasado).
- **Repasse de custos (pass-through)**: regressão de Δln p em Δln custo (com defasagens) mede quanto e quão rápido o custo vira preço; relevante para planejar reajustes.

## 6. Do coeficiente à decisão

- Receita: com elasticidade ε, ΔR/R ≈ (1 + ε)·Δp/p. Se |ε| < 1 (inelástica), aumentar preço aumenta receita.
- Margem: o preço que maximiza lucro (Lerner) satisfaz (p − c)/p = 1/|ε| para firma com poder de mercado. Se a margem atual for maior que 1/|ε|, há indício de preço alto demais, e vice-versa.
- Reporte **faixas**, não ponto: "elasticidade entre −1,1 e −1,6 (IC 95%)" e simule o impacto em receita e margem nos extremos.
- Elasticidades valem para variações **pequenas** e dentro do intervalo de preços observado; extrapolar para aumentos de 30% é especulativo.

## 7. Checklist

1. Unidade de análise e nível de agregação definidos e justificados.
2. Preço medido de forma que o comprador não o altere (tabela/cardápio, ou agregado).
3. Sazonalidade, eventos, promoções e disponibilidade controlados.
4. Fonte de variação de preço discutida explicitamente: custos, experimento, mudança de política — ou assumida exógena com justificativa.
5. MQO e IV (ou quase-experimento) comparados; F da primeira etapa reportado.
6. Elasticidades próprias e cruzadas com intervalos; curto e longo prazo se houver dinâmica.
7. Robustez: outro nível de agregação, outra janela, com/sem período atípico.
8. Tradução para receita e margem com a incerteza explicitada.
