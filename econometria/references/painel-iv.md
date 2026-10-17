# Endogeneidade, variáveis instrumentais e dados em painel

## Índice
1. Fontes de endogeneidade
2. Variáveis instrumentais e 2SLS
3. Testes: relevância, endogeneidade, sobreidentificação
4. Instrumentos fracos
5. Equações simultâneas e identificação
6. Estrutura e preparação do painel
7. Estimadores de painel (pooled, FE, RE, FD)
8. Erros-padrão em painel e número de clusters
9. Hausman, Mundlak e LM de Breusch-Pagan
10. Diferenças-em-diferenças
11. Painel dinâmico
12. Armadilhas

## 1. Fontes de endogeneidade

E(ε\|X) ≠ 0 por: (i) variável relevante omitida correlacionada com um regressor; (ii) forma funcional errada; (iii) erro de medida em X (viés de atenuação); (iv) simultaneidade — y e x determinados juntos (preço e quantidade; renda e educação; vendas e publicidade). MQO fica viesado **e inconsistente**, e a endogeneidade de um regressor contamina os demais coeficientes.

Direção do viés: plim β̂ = β + Cov(x, ε)/Var(x). Pensar no sinal provável da covariância frequentemente basta para concluir se o efeito verdadeiro é maior ou menor que o estimado.

## 2. Variáveis instrumentais e 2SLS

Instrumento z: (a) **exógeno** — Cov(z, ε) = 0 (não testável com um instrumento só; é argumento econômico — "mais arte que ciência"); (b) **relevante** — Cov(z, x) ≠ 0 (testável na primeira etapa). Regressores exógenos são instrumentos de si mesmos; MQO é o caso particular em que todos os X são exógenos.

- Exatamente identificado (nº de instrumentos = nº de endógenas): β̂_IV = (Z'X)⁻¹Z'y.
- Sobreidentificado: **2SLS** — 1ª etapa: cada endógena em todos os instrumentos (formas reduzidas); 2ª etapa: y nos valores ajustados. β̂ = (X'P_Z X)⁻¹X'P_Z y.
- Não faça a 2ª etapa "à mão" com MQO para inferência: os erros-padrão saem errados. Use o estimador pronto.

```python
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
f = "log_q ~ 1 + log_renda + C(mes) + [log_preco ~ log_custo_insumo + log_frete]"
iv = IV2SLS.from_formula(f, data=df).fit(cov_type="robust")      # ou "clustered", clusters=df.loja
print(iv.summary)
print(iv.first_stage)       # F parcial, R² parcial dos instrumentos excluídos
iv.wu_hausman(); iv.durbin()  # endogeneidade (H0: regressor exógeno)
iv.sargan; iv.basmann        # sobreidentificação (homocedasticidade)
gmm = IVGMM.from_formula(f, data=df).fit(); gmm.j_stat           # Hansen J (robusto)
liml = IVLIML.from_formula(f, data=df).fit(cov_type="robust")    # preferível com instrumentos fracos
```

Custos do IV: menos eficiente que MQO (variância maior); viesado em amostra finita na direção do MQO; distribuição em amostra pequena pode diferir muito da assintótica.

## 3. Testes

**Relevância**: F dos instrumentos excluídos na 1ª etapa (robusto, se for o caso). Reporte sempre.

**Endogeneidade (Durbin-Wu-Hausman, versão por regressão — a do curso)**:
1. Regrida a suspeita endógena em todas as exógenas e instrumentos; guarde o resíduo v̂.
2. Inclua v̂ na equação estrutural estimada por MQO (com erros robustos, se necessário).
3. Se o coeficiente de v̂ for significativo, a variável é endógena → use IV; se não, MQO é consistente e mais eficiente.
`scripts/diagnosticos.py` tem `teste_endogeneidade` com esse procedimento.

**Sobreidentificação** (só com mais instrumentos que endógenas; q = nº instrumentos − nº endógenas):
- Regrida os resíduos do 2SLS em todas as exógenas e instrumentos; nR² ~ χ²(q) (Wooldridge) ou Sargan (n − k)R². Rejeitar → ao menos um instrumento é inválido.
- Com heterocedasticidade, use o J de Hansen (`IVGMM ... .j_stat`).
- Não rejeitar não prova validade: se todos os instrumentos forem inválidos na mesma direção, o teste não detecta.
- `scripts/diagnosticos.py` tem `sobreidentificacao`.

## 4. Instrumentos fracos

- Regra de bolso de Staiger-Stock: F da 1ª etapa > 10. Critério mais exigente de Lee et al. (2022) para o teste t com tamanho de 5% correto: F > 104,7. Com F entre 10 e 100, reporte também intervalos de Anderson-Rubin.
- Instrumentos fracos: 2SLS viesado em direção ao MQO e intervalos com cobertura errada; muitos instrumentos fracos pioram o viés.
- **LIML** é aproximadamente não viesado na mediana em modelos sobreidentificados e quase não sofre com instrumentos adicionais inúteis; 2SLS exatamente identificado também é aproximadamente não viesado na mediana (Angrist-Pischke). Com vários instrumentos, compare 2SLS e LIML: se divergirem, desconfie.

## 5. Equações simultâneas e identificação

- **Forma estrutural** (comportamento: demanda, oferta) vs. **forma reduzida** (endógenas em função só de exógenas). A forma reduzida pode ser estimada por MQO; os parâmetros estruturais só são recuperáveis se a equação for identificada.
- **Condição de ordem** (necessária): número de exógenas excluídas da equação ≥ número de endógenas incluídas − 1. Igual → exatamente identificada (mínimos quadrados indiretos funcionam); maior → sobreidentificada (use 2SLS; MQI daria várias soluções).
- **Condição de posto** (necessária e suficiente): dos coeficientes das variáveis excluídas da equação mas presentes nas demais, é possível formar ao menos um determinante não nulo de ordem (M−1).
- Exemplo clássico: demanda Q = a₁ + b₁P + c₁Renda + u₁ e oferta Q = a₂ + b₂P + c₂Chuva + u₂ — renda identifica a oferta, chuva identifica a demanda. Aplicações de varejo em `demanda-elasticidades.md`.
- Modelos recursivos (B triangular, erros não correlacionados) podem ser estimados equação a equação por MQO.

## 6. Estrutura do painel

`linearmodels` exige MultiIndex (entidade, tempo) em formato **long** (uma linha por entidade-período). Para converter de wide: `pd.wide_to_long` ou `melt`.

```python
df = df.set_index(["loja", "mes"])
df.groupby(level=0).size().describe()     # balanceamento (equivalente a xtdescribe)
# Variação total, entre e dentro (equivalente a xtsum):
x = df["preco"]
entre = x.groupby(level=0).mean().std()
dentro = (x - x.groupby(level=0).transform("mean")).std()
```

- Efeitos fixos só usam a variação **dentro** da unidade; variáveis quase constantes no tempo ficam mal estimadas e as constantes (região, formato da loja) são absorvidas.
- Painel desbalanceado é aceitável, mas investigue por que unidades saem (lojas fechadas = viés de sobrevivência).

## 7. Estimadores

| Estimador | Pressuposto-chave | Quando |
|---|---|---|
| Pooled OLS | Heterogeneidade não correlacionada com X | Referência; exige erros clusterizados |
| Efeitos fixos (`EntityEffects`) | Heterogeneidade constante no tempo pode se correlacionar com X | Padrão para efeitos em dados de lojas/empresas |
| Efeitos fixos duplos (+ `TimeEffects`) | Choques comuns a todas as unidades em cada período | Quase sempre recomendável (absorve sazonalidade e macro) |
| Efeitos aleatórios | Heterogeneidade não correlacionada com X | Amostra aleatória de unidades; estima variáveis fixas no tempo |
| Primeiras diferenças | Como FE; melhor se o erro é um passeio aleatório | T pequeno com erro persistente |
| Between | Médias por unidade | Relações de longo prazo/corte transversal |

```python
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, FirstDifferenceOLS, BetweenOLS
fe = PanelOLS.from_formula("log_vendas ~ 1 + log_preco + promocao + EntityEffects + TimeEffects", df)\
         .fit(cov_type="clustered", cluster_entity=True)
fe.f_pooled               # F: efeitos fixos conjuntamente significativos (FE vs pooled)
re = RandomEffects.from_formula("log_vendas ~ 1 + log_preco + promocao", df).fit()
re.theta, re.variance_decomposition   # peso do componente individual
```

## 8. Erros-padrão e clusters

- Observações da mesma unidade são correlacionadas: cluster por entidade é o padrão (`cluster_entity=True`); choques comuns no tempo → cluster duplo (`cluster_time=True` também) ou Driscoll-Kraay (`cov_type="kernel"`).
- **Número de clusters é o tamanho amostral efetivo** (Hansen): G = 20 é amostra muito pequena.
- **Poucos clusters** (Angrist-Pischke: menos de ~42): erros clusterizados subestimam a incerteza. Opções: usar t com G − k graus de liberdade; correção de viés BRL/CR2; wild cluster bootstrap; reportar o maior entre o erro robusto e o convencional.
- **Em que nível clusterizar** (Hansen): nível fino demais → viés (erros pequenos demais); agregado demais → erros imprecisos. Não há regra definitiva; clusterize no nível em que os choques são comuns e em que o tratamento varia, verificando se ainda há clusters suficientes.

## 9. Hausman, Mundlak e LM de Breusch-Pagan

- **Pooled vs. RE**: LM de Breusch-Pagan (H0: variância do efeito individual = 0) — `scripts/diagnosticos.py` tem `lm_breusch_pagan_painel`.
- **FE vs. pooled**: `fe.f_pooled`.
- **FE vs. RE (Hausman)**: H0 = RE consistente (e eficiente). H = (b_FE − b_RE)'[V_FE − V_RE]⁻¹(b_FE − b_RE) ~ χ²(k), usando coeficientes comuns e covariâncias **não robustas**. `scripts/diagnosticos.py hausman`. Se V_FE − V_RE não for positiva semidefinida, a estatística pode sair negativa (o Stata contorna com `sigmamore`); nesse caso use Mundlak.
- **Mundlak (robusto, preferível com cluster)**: inclua as médias por unidade dos X no modelo RE; se forem conjuntamente significativas (Wald com cluster), use FE.
- Em dados de negócio, FE quase sempre vence: localização e gestão se correlacionam com preço e promoção.

## 10. Diferenças-em-diferenças

Efeito de uma intervenção que atinge um grupo e não outro (reforma de lojas, nova política de preço, mudança tributária):

```python
df["tratado_pos"] = df["tratado"] * df["pos"]
did = PanelOLS.from_formula("y ~ tratado_pos + EntityEffects + TimeEffects", df)\
          .fit(cov_type="clustered", cluster_entity=True)
```

- **Tendências paralelas**: sem tratamento, os grupos evoluiriam igual. Verifique com *event study* (dummies de tempo relativo ao tratamento): coeficientes pré-tratamento ≈ 0. Faça placebos (data falsa, grupo falso).
- **Adoção escalonada** (unidades tratadas em datas diferentes) com efeitos heterogêneos: o TWFE simples pode ser seriamente viesado (Goodman-Bacon). Use Callaway-Sant'Anna, Sun-Abraham ou de Chaisemartin-D'Haultfœuille, ou ao menos declare a limitação.
- Poucos grupos tratados → problema de poucos clusters (seção 8).

## 11. Painel dinâmico

- y defasado com efeitos fixos gera **viés de Nickell** (ordem 1/T), relevante com T < ~20.
- T grande (≥ 30): FE comum aceitável, com ressalva.
- T pequeno, N grande: GMM de Arellano-Bond (diferenças) ou Blundell-Bond (sistema), pacote `pydynpd`. Equivalências com o `xtabond` do Stata: variáveis estritamente exógenas como instrumentos de si mesmas; **predeterminadas** (`pre()`) instrumentadas por defasagens a partir de t−1; **endógenas** (`endogenous()`) a partir de t−2. Use two-step com correção de Windmeijer (erros robustos), reporte AR(2) nas diferenças (deve não rejeitar) e Hansen J, e limite o número de instrumentos (< N).

## 12. Armadilhas

- Pooled OLS com erros não clusterizados → significância inflada.
- Interpretar coeficiente de FE como diferença entre unidades (é efeito dentro da unidade ao longo do tempo).
- *Bad controls*: controlar por variáveis afetadas pelo tratamento.
- Linguagem causal com FE sem discutir variáveis que mudam no tempo e não estão no modelo.
- IV com instrumento fraco ou exclusão implausível — pior que MQO honesto com ressalvas sobre a direção do viés.
