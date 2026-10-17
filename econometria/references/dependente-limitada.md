# Variável dependente qualitativa e limitada

Use quando y é binária (comprou/não, inadimplente/não), contagem (nº de pedidos), censurada (gasto que é zero para muitos), fracionária (participação de 0 a 1) ou ordenada/multinomial.

## Índice
1. Modelo de probabilidade linear (MPL)
2. Logit e probit
3. Efeitos marginais e razões de chance
4. Testes e qualidade do ajuste
5. Tobit (censura)
6. Seleção amostral (Heckman) e modelos em duas partes
7. Contagem, fracionária, ordenada e multinomial
8. Armadilhas

## 1. Modelo de probabilidade linear

y binária estimada por MQO: P(y=1\|x) = xβ, então β_j = variação na probabilidade.
- Vantagens: interpretação direta; bom para efeitos médios perto da média dos X; base para IV com regressor endógeno.
- Problemas: previsões fora de [0,1]; heterocedasticidade por construção, Var(ε\|x) = p(x)(1 − p(x)) → **sempre use erros robustos (HC3)**; efeitos constantes, irreais nos extremos.
- Use como verificação de robustez ao lado de logit/probit: os sinais e a significância devem concordar.

## 2. Logit e probit

Modelo de variável latente: y* = xβ + ε, y = 1 se y* > 0. Probit: ε normal; logit: ε logística. Estimados por máxima verossimilhança.

```python
import statsmodels.formula.api as smf
logit  = smf.logit("inadimplente ~ renda + idade + C(regiao)", data=df).fit(disp=False)
probit = smf.probit("inadimplente ~ renda + idade + C(regiao)", data=df).fit(disp=False)
# erros robustos (QMV): .fit(cov_type="HC1") ; cluster: .fit(cov_type="cluster", cov_kwds={"groups": df["loja"]})
```

- Logit e probit dão probabilidades quase iguais, exceto nas caudas; os coeficientes diferem por escala (β_logit ≈ 1,6·β_probit). Compare efeitos marginais, nunca coeficientes brutos.
- Logit: ln[p/(1−p)] = xβ (log-chance linear), o que dá a interpretação por razão de chances.
- Separação perfeita (uma variável prevê y exatamente) faz o MV divergir: o statsmodels avisa; agrupe categorias ou use regularização/Firth.

## 3. Efeitos marginais e razões de chance

Os coeficientes **não** são efeitos sobre a probabilidade. O efeito de x_j é f(xβ)·β_j, que depende de x.

```python
me = logit.get_margeff(at="overall", method="dydx")   # efeito marginal médio (AME) — preferível
me_media = logit.get_margeff(at="mean")                # efeito na média dos X (MEM)
me.summary()
# Dummies: dummy=True calcula a variação discreta P(y=1|d=1) − P(y=1|d=0) em vez da derivada
logit.get_margeff(at="overall", dummy=True)
# Contagens: count=True
```

- **AME** (média dos efeitos individuais) é mais robusto que o efeito avaliado na média (MEM), sobretudo com dummies — o "indivíduo médio" pode não existir.
- Para um perfil específico (como o `margins, at()` do Stata): calcule `logit.predict(perfil_com_x)` e `perfil_com_x+Δ` e tire a diferença; para intervalo, bootstrap ou método delta.
- **Razão de chances (logit)**: exp(β_j) = multiplicador da chance quando x_j aumenta 1 unidade. Ex.: exp(0,11) = 1,116 → a chance aumenta 11,6% por unidade. Não confunda chance com probabilidade.

## 4. Testes e qualidade do ajuste

- Wald: `res.wald_test("x1 = 0, x2 = 0")`; hipóteses lineares como `"x1 = x2"`.
- Razão de verossimilhança (restrito vs. irrestrito): `LR = 2*(irr.llf - restr.llf)`, χ² com g graus de liberdade. O statsmodels reporta `llr` e `llr_pvalue` contra o modelo só com constante.
- Pseudo-R² de McFadden: 1 − llf/llnull (`res.prsquared`). Valores de 0,2 a 0,4 já indicam bom ajuste; não compare com R² de MQO.
- Tabela de classificação: `res.pred_table(threshold=0.5)`. O corte de 0,5 é arbitrário; com classes desbalanceadas, escolha o corte pelo custo dos erros (falso positivo vs. falso negativo) ou pela proporção amostral.
- Curva ROC e AUC (`sklearn.metrics.roc_auc_score`) medem discriminação independentemente do corte.
- Calibração: agrupe por decis de probabilidade prevista e compare com a frequência observada (teste de Hosmer-Lemeshow).
- Para previsão, avalie **fora da amostra** (validação cruzada).

## 5. Tobit (censura)

Use quando y é contínua e positiva mas acumula massa em zero (ou em um teto): horas trabalhadas, gasto com um item, volume comprado.

y* = xβ + ε, ε ~ N(0, σ²); y = max(0, y*).

- MQO só com y > 0 ou com todos os dados é viesado e inconsistente.
- O statsmodels não tem Tobit pronto: `scripts/diagnosticos.py` tem `tobit(y, X, limite_inferior=0)` (MV com função de verossimilhança censurada).
- Três efeitos marginais, escolha conforme a pergunta:
  - sobre a variável latente: β_j;
  - sobre E(y\|x) (incondicional): Φ(xβ/σ)·β_j — o mais usado em negócio;
  - sobre E(y\|y>0, x): β_j·[1 − λ(c)(c + λ(c))], com c = xβ/σ e λ(c) = φ(c)/Φ(c) (inversa da razão de Mills).
- O Tobit supõe que **o mesmo processo** decide se compra e quanto compra. Se forem decisões distintas (muito comum), use modelo em duas partes (seção 6). Também é sensível a heterocedasticidade e não normalidade.

## 6. Seleção amostral (Heckman) e duas partes

- **Heckman (dois passos)**: quando y só é observado para uma amostra selecionada não aleatoriamente (salário só de quem trabalha). Passo 1: probit da seleção com todas as variáveis, incluindo ao menos uma que afeta a seleção e não o resultado (restrição de exclusão — sem ela, a identificação depende só da não linearidade, frágil). Passo 2: MQO de y na subamostra incluindo λ̂ = φ(ẑγ)/Φ(ẑγ). Coeficiente de λ̂ significativo indica viés de seleção. Os erros-padrão do passo 2 precisam de correção (bootstrap de todo o procedimento).
- **Modelo em duas partes (hurdle)**: parte 1 = logit/probit de comprar ou não (margem extensiva); parte 2 = MQO/GLM do valor entre quem compra (margem intensiva). E(y\|x) = P(y>0\|x)·E(y\|y>0, x). Permite que preço afete diferente a decisão de comprar e a quantidade.
- **Zeros e logs**: ln(y+1) não é uma solução neutra (o resultado depende da escala). Prefira PPML — Poisson por quase-máxima verossimilhança, `smf.glm("y ~ x", data, family=sm.families.Poisson()).fit(cov_type="HC1")` — que aceita zeros e tem coeficientes interpretáveis como semielasticidades.

## 7. Outros tipos de y

| y | Modelo | statsmodels |
|---|---|---|
| Contagem (0, 1, 2…) | Poisson; binomial negativa se sobredispersão (Var > média) | `smf.poisson`, `smf.negativebinomial`; ou GLM |
| Excesso de zeros | Zero-inflated Poisson/NB | `statsmodels.discrete.count_model.ZeroInflatedPoisson` |
| Fração/participação (0–1) | Resposta fracionária de Papke-Wooldridge | `smf.glm(..., family=sm.families.Binomial()).fit(cov_type="HC1")` |
| Ordenada (nota 1–5) | Probit/logit ordenado | `statsmodels.miscmodels.ordinal_model.OrderedModel` |
| Categorias sem ordem | Logit multinomial | `smf.mnlogit` (cuidado com a hipótese de independência de alternativas irrelevantes) |
| Duração até o evento | Sobrevivência (Cox, Weibull) | `lifelines` ou `statsmodels.duration.hazard_regression.PHReg` |

## 8. Armadilhas

- Interpretar coeficiente de logit/probit como variação de probabilidade.
- Comparar coeficientes entre logit e probit ou entre amostras diferentes (a escala σ muda).
- Usar R² de MQO ou acurácia com corte 0,5 como medida principal em amostra desbalanceada (1% de inadimplentes → 99% de acerto prevendo sempre "não").
- Esquecer que em modelos não lineares os efeitos de interação não são o coeficiente da interação; calcule a dupla diferença das probabilidades.
- Endogeneidade em modelo binário: 2SLS sobre MPL é a solução mais simples e transparente; probit com função de controle (incluir o resíduo do primeiro estágio) é a alternativa não linear.
