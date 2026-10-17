# Séries temporais

## Índice
1. Árvore de decisão
2. Decomposição e filtros
3. Estacionariedade e raiz unitária
4. Quebras estruturais
5. Modelos univariados (ARIMA/SARIMA, ETS, ARFIMA)
6. Regressão com séries temporais (ADL, ARIMAX) e regressão espúria
7. Cointegração e modelos de correção de erros (ECM/VECM)
8. VAR: estimação, Granger, impulso-resposta, decomposição de variância
9. VAR estrutural (SVAR) e identificação
10. VAR bayesiano e VAR com muitos dados
11. Previsão: avaliação, intervalos, combinação
12. Armadilhas comuns

## 1. Árvore de decisão

```
Uma série só, objetivo = prever?
 └─ SARIMA / ETS (seção 5) → avaliação fora da amostra (seção 11)
Y explicado por X, efeito de interesse?
 ├─ Todas I(0)            → MQO/ADL com erros HAC (seção 6)
 ├─ Todas I(1)
 │   ├─ Cointegradas      → ECM/VECM (seção 7)
 │   └─ Não cointegradas  → modelo em diferenças (seção 6)
 └─ Ordens misturadas     → diferenciar as I(1) ou ARDL/bounds test de Pesaran
Variáveis que se afetam mutuamente (juros, câmbio, inflação, vendas)?
 └─ VAR (I(0) ou diferenças) / VECM (cointegradas) → seções 8–10
Retornos financeiros, risco → volatilidade.md
```

Protocolo geral (ciclo de modelagem, Heij et al./curso Erasmus): tratar não estacionariedade → modelo inicial pelo ACF/PACF e teoria → estimar → diagnosticar resíduos → reespecificar até os diagnósticos passarem → só então prever.

## 2. Decomposição e filtros

Série = componente determinístico (tendência, sazonalidade — sem necessidade de estatística) + componente estocástico (o que se modela).

- Sazonalidade: dummies sazonais na regressão; `statsmodels.tsa.seasonal.STL` para decompor; X-13 (`x13_arima_analysis`, requer o binário do Census) para ajuste oficial.
- **Filtro HP** (`statsmodels.tsa.filters.hp_filter.hpfilter(y, lamb=...)`): λ = 1600 trimestral, 129600 mensal, 6,25 anual (Ravn-Uhlig). É usado pelo BCB para hiato do produto. Ressalvas importantes: distorce o fim da amostra (o último ponto depende de dados futuros não observados — péssimo para tempo real), pode gerar ciclos espúrios e falha em remover tendências estocásticas adequadamente (Phillips e Jin, 2015; Hamilton, 2018). Alternativa: filtro de Hamilton — regredir y_{t+h} em y_t, …, y_{t−3} (h = 8 trimestres ou 24 meses) e usar o resíduo como ciclo.
- Variação anual (YoY: ln y_t − ln y_{t−12}) elimina sazonalidade estável e é intuitiva para negócio, mas introduz autocorrelação MA(11) nas variações.

## 3. Estacionariedade e raiz unitária

Por que importa: com raiz unitária, choques têm efeito permanente (impulso-resposta não decai), a variância cresce com o tempo e a teoria assintótica usual (t, F) não vale — regressões entre séries I(1) independentes parecem significativas (regressão espúria).

Use **ADF e KPSS juntos**, porque têm nulas opostas:

| ADF (H0: raiz unitária) | KPSS (H0: estacionária) | Conclusão |
|---|---|---|
| rejeita | não rejeita | Estacionária |
| não rejeita | rejeita | Raiz unitária → diferenciar |
| rejeita | rejeita | Possível tendência determinística ou quebra; teste com tendência e Zivot-Andrews |
| não rejeita | não rejeita | Inconclusivo: baixo poder, alta persistência ou **memória longa** (ARFIMA, seção 5) |

Escolha de testes (Stock, Handbook cap. 46):
- **DF-GLS (Elliott-Rothenberg-Stock)** tem poder bem maior que o ADF (ex.: 60% vs. 22% contra ρ = 0,9) mantendo bom controle de tamanho → use como teste principal de raiz unitária (`arch.unitroot.DFGLS`).
- ADF: tamanho correto, pouco poder. Phillips-Perron: grandes distorções de tamanho quando há componente MA negativo (Schwert) — evite como teste único.
- Defasagens do ADF: por BIC/AIC modificado (`autolag="AIC"`/`"BIC"`) ou regra do curso (começar com p grande e reduzir até a estatística t da última defasagem ser ≥ 1,6). Poucas defasagens distorcem o tamanho; muitas reduzem o poder.
- Determinísticos: constante se a série oscila em torno de um nível; constante + tendência se tem tendência visível. Errar aqui muda o resultado.
- Bootstrap para valores críticos em amostras pequenas: estimar sob H0 (Δy_t AR(p)), reamostrar resíduos centrados, reconstruir y*, recalcular a estatística B vezes (5.000+).
- Quebra estrutural não modelada faz uma série estacionária parecer I(1) → Zivot-Andrews (`arch.unitroot.ZivotAndrews`).

Visão prática do curso: para as séries macro mais comuns, a ordem de integração já é conhecida na literatura (juros e inflação muitas vezes tratados como I(0) persistentes no Brasil pós-1999; PIB, preços e câmbio em log como I(1)); "não reinvente a roda". A escolha é mais metodológica que estatística — reporte a robustez nas duas versões.

`scripts/diagnosticos.py estacionariedade` roda ADF, KPSS e DF-GLS em nível e em diferença, com constante e com tendência, e sugere a ordem de integração.

## 4. Quebras estruturais

Quebras (pandemia, mudança de regime cambial, mudança de metodologia, novo cardápio) violam a hipótese de parâmetros constantes. Ligadas à crítica de Lucas: parâmetros dependem das "regras do jogo".

| Situação | Teste | Python |
|---|---|---|
| Data conhecida | Chow (F comparando subamostras); Wald com covariâncias HAC (robusto) | `scripts/diagnosticos.py chow` |
| Data desconhecida | sup-Wald de Andrews (máximo do Wald em todas as datas no intervalo [0,15T; 0,85T]; valores críticos não padrão) | laço manual + valores de Andrews (1993) ou bootstrap |
| Várias quebras | Bai-Perron | pacote `ruptures` (detecção) |
| Instabilidade gradual | CUSUM / CUSUM-Q | `statsmodels.stats.diagnostic.breaks_cusumolsresid`, `recursive_olsresiduals` |
| Quebra perto do fim da amostra | Teste de Andrews (2003) por subamostragem | implementação manual |
| Raiz unitária com quebra | Zivot-Andrews | `arch.unitroot.ZivotAndrews` |

Com resíduos autocorrelacionados, use **sieve bootstrap**: estime a tendência com quebra, ajuste AR(p) por BIC aos resíduos, gere resíduos bootstrap pelo AR, reconstrua a série e recalcule a estatística.

Tratamento: dummies de nível/inclinação, estimar só no regime atual, ou modelos de mudança de regime (Markov switching: `statsmodels.tsa.regime_switching.MarkovRegression`).

## 5. Modelos univariados

**Box-Jenkins (protocolo do curso)**:
1. Transformar (log se a variância cresce com o nível; Box-Cox).
2. Definir d (seção 3) e D sazonal.
3. ACF/PACF da série estacionária:

| Modelo | ACF | PACF |
|---|---|---|
| AR(p) | decai exponencialmente | corta após lag p |
| MA(q) | corta após lag q | decai exponencialmente |
| ARMA(p,q) | decai a partir de q | decai a partir de p |

4. Começar com um ARMA grande, estimar as combinações menores e escolher por **BIC** (mais parcimonioso e consistente; AIC tende a escolher modelos maiores). Não descarte uma defasagem teoricamente relevante só por não ser significativa.
5. Verificar estacionariedade (raízes do polinômio AR fora do círculo unitário) e invertibilidade (raízes MA fora do círculo unitário).
6. Resíduos devem ser ruído branco: Ljung-Box (`acorr_ljungbox(res.resid, lags=[12, 24], model_df=p+q)`), sem efeito ARCH remanescente.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
m = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, 12)).fit(disp=False)
```

- AR(p) puro pode ser estimado por MQO; com termos MA, é preciso máxima verossimilhança. Um AR longo aproxima qualquer ARMA invertível — é um atalho prático.
- **ETS / suavização exponencial** (`statsmodels.tsa.exponential_smoothing.ets.ETSModel` ou `holtwinters.ExponentialSmoothing`): nas competições M (Makridakis e Hibon, 2000), métodos simples como suavização exponencial com tendência amortecida e o método Theta igualaram ou superaram modelos sofisticados. Sempre compare.
- **ARFIMA (memória longa)**: quando ADF e KPSS não rejeitam (caso clássico do IPCA mensal) e o ACF decai hiperbolicamente. (1 − L)^d y_t com −0,5 < d < 0,5 é estacionário e invertível. O curso estimou d ≈ 0,19 para o IPCA com AR(1). Em Python: estimar d pelo método de Geweke-Porter-Hudak (regressão do log-periodograma nas frequências baixas), filtrar (1 − L)^d com os pesos b_0 = 1, b_j = b_{j−1}(j − 1 − d)/j, e ajustar o ARMA na série filtrada.

## 6. Regressão com séries temporais

**Regressão espúria** — o que acontece com MQO de y em x (prova do curso):

| y e x | MQO / teste t | Solução |
|---|---|---|
| Estacionárias em tendência | R² e t inflados pela tendência comum | Incluir tendência ou eliminar a tendência das duas |
| I(1) com drift, não cointegradas | t não converge, R² → 1, correlação espúria | Modelar em diferenças |
| I(1) sem drift, não cointegradas | Inferência inválida | Diferenças |
| I(1) cointegradas | Coeficiente superconsistente; t não padrão | ECM (seção 7) |

Sinal de alerta: R² > Durbin-Watson.

**ADL** — efeitos que levam tempo:

```python
df["d_vendas"] = df["log_vendas"].diff()
for k in range(0, 7):
    df[f"d_selic_l{k}"] = df["selic"].diff().shift(k)
termos = " + ".join(f"d_selic_l{k}" for k in range(7))
m = smf.ols(f"d_vendas ~ d_vendas.shift(1) + {termos} + C(mes)", data=df)\
       .fit(cov_type="HAC", cov_kwds={"maxlags": 6})
```

- Multiplicador de longo prazo = Σβ / (1 − Σφ); erro-padrão pelo método delta (`m.t_test` em combinação linear, ou bootstrap).
- Defasagens por BIC ou teoria (política monetária costuma levar de 6 a 12 meses para afetar a atividade).
- Subestimar p (ordem do AR) torna os estimadores inconsistentes; superestimar só custa eficiência.
- **ARIMAX** (`SARIMAX(y, exog=X, order=...)`) quando o resíduo tem estrutura ARMA.

**Causalidade de Granger**: x Granger-causa y se defasagens de x melhoram a previsão de y. É **precedência preditiva**, não causalidade: um banco central que antecipa a inflação faz os juros "causarem" inflação no sentido de Granger. Escreva "x ajuda a prever y".

## 7. Cointegração e ECM/VECM

Duas séries I(1) são cointegradas se uma combinação linear é I(0) — existe um equilíbrio de longo prazo. Exemplos que o curso cita: consumo e renda, preço e dividendos, PPC (câmbio e preços), preços de substitutos (etanol e gasolina, PETR3 e PETR4 — base de arbitragem estatística long-short).

**Engle-Granger (três passos)**:
1. Confirmar que todas são I(1) (cointegração exige mesma ordem).
2. Estimar o vetor por MQO em nível (superconsistente); com drifts diferentes, incluir tendência.
3. Testar raiz unitária nos resíduos com os valores críticos de MacKinnon para cointegração — **não os do ADF comum**. `statsmodels.tsa.stattools.coint(y, x)` já faz isso.

**Johansen** (várias séries, várias relações): reescreve o VAR como ΔY_t = ΠY_{t−1} + ΣΓ_iΔY_{t−i} + ε_t e testa o posto de Π = αβ' (teorema de representação de Granger).

```python
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank, VECM
p = select_order(dados, maxlags=12, deterministic="ci").bic       # defasagens em diferenças
r = select_coint_rank(dados, det_order=0, k_ar_diff=p, method="trace", signif=0.05).rank
vecm = VECM(dados, k_ar_diff=p, coint_rank=r, deterministic="ci").fit()
vecm.beta, vecm.alpha
```

- Casos determinísticos mais comuns: (a) séries sem drift, constante só no vetor de cointegração (`"ci"`); (b) séries com drift e constante (`"co"` ou `"ci"` + `"lo"`); (c) constante e tendência no vetor (`"li"`). A escolha muda o resultado — justifique pelo gráfico.
- Variáveis exógenas no Johansen: use-as **centradas e sem tendência**, inclusive dummies sazonais (subtraia a média).
- Normalize β pela variável de interesse e imponha restrições da teoria quando existirem.
- **α** = velocidade de ajuste; −0,2 → 20% do desvio corrigido por período; meia-vida = ln 0,5 / ln(1 − 0,2) ≈ 3,1 períodos. O sinal precisa corrigir o desvio.
- O ECM permite separar curto e longo prazo e revela previsibilidade de longo prazo que regressões simples não captam (Campbell-Shiller: dividend-yield prevendo retornos).
- `scripts/diagnosticos.py cointegracao` roda Engle-Granger e Johansen (traço e máximo autovalor).

## 8. VAR

Y_t = c + Φ₁Y_{t−1} + … + Φ_pY_{t−p} + ε_t, com Σ = Var(ε_t) não diagonal. Cada equação por MQO equivale ao MQG do sistema (mesmos regressores).

```python
from statsmodels.tsa.api import VAR
var = VAR(dados_estacionarios, exog=exogenas)       # exógenas: commodities, dummies
print(var.select_order(12).summary())               # AIC, BIC, HQ, FPE
res = var.fit(p, trend="c")
res.is_stable()                                      # autovalores < 1 em módulo
res.test_whiteness(nlags=12, adjusted=True).summary()
res.test_normality().summary()
res.test_causality("vendas", ["selic"], kind="f").summary()
irf = res.irf(24); irf.plot(orth=True); irf.plot_cum_effects(orth=True)
res.fevd(24).summary()
```

- **Defasagens**: BIC é consistente, AIC tende a escolher mais. Mas se os resíduos ficarem autocorrelacionados, acrescente defasagens — o próprio BCB, num VAR de inflação setorial (Relatório de Inflação, jun/2015), usou 2 defasagens apesar de o BIC indicar 1, por causa da correlação serial.
- **Parâmetros crescem rápido**: k variáveis × p defasagens × k equações. Um VAR com 9 variáveis e 4 defasagens tem 333 parâmetros. Com 120 meses, mantenha k ≤ 4–5.
- **Impulso-resposta ortogonalizada (Cholesky)**: a ordem importa — a primeira variável não responde contemporaneamente às demais. Coloque as mais "exógenas" primeiro e justifique (BCB: administrados → alimentação → industriais → serviços). Mostre a robustez a outras ordens.
- **IRF generalizada (Pesaran-Shin)**: não depende da ordem, mas não permite ler "choque de 1% em x causa β% em y". Fórmula: ψ_j(h) = Φ_h Σ e_j / √σ_jj (não está pronta no statsmodels; calcule com `res.ma_rep(h)` e `res.sigma_u`).
- **Intervalos das IRF**: `res.irf(h).plot(orth=True, stderr_type="mc", repl=1000)` (Monte Carlo) ou bootstrap. Intervalos bootstrap simples de IRFs podem ter cobertura ruim em amostras pequenas; prefira o bootstrap com correção de viés (Kilian, 1998) quando a conclusão depender disso.
- **Decomposição de variância**: quais choques explicam a variância do erro de previsão de cada variável; também depende da ortogonalização.
- **Price puzzle**: juros sobem e preços sobem depois, porque o banco central antecipa inflação; incluir commodities (ou expectativas) no VAR mitiga.
- VAR com séries I(1) em nível estima consistentemente, mas testes de Granger e IRF longos ficam problemáticos; com cointegração, use VECM; sem ela, use diferenças.

## 9. SVAR e identificação

Forma estrutural: B Y_t = A₀ + A₁Y_{t−1} + … + u_t, com choques estruturais u_t não correlacionados (Ω diagonal). Estimamos a forma reduzida e recuperamos B impondo restrições.

| Estratégia | Ideia | Python | Cuidados |
|---|---|---|---|
| Recursiva (Cholesky) | B triangular | `res.irf(...).orth_irfs` | Ordem precisa de justificativa econômica |
| Restrições de curto prazo não recursivas | Zeros em B pela teoria (ex.: BC só observa o produto com defasagem) | `statsmodels.tsa.vector_ar.svar_model.SVAR(dados, svar_type="AB", A=..., B=...)` | Contar parâmetros: precisa de ≥ k(k−1)/2 restrições |
| Longo prazo (Blanchard-Quah) | Choque não tem efeito permanente (neutralidade da moeda): C₁₂(1) = 0 | Manual: C(1) = (I − ΣΦ_i)⁻¹; B₀⁻¹ = C(1)⁻¹·chol(C(1) Σ C(1)') | Exige saber a raiz unitária exata; não fixa o sinal da resposta; frágil com variável I(0) muito persistente; sensível a nível vs. diferença (Kilian, 2011) |
| Restrições de sinal (Faust, Uhlig) | Choque de demanda: p↑ e q↑; oferta: p↑ e q↓ | Manual: rotações Q ortogonais (decomposição QR de matrizes normais), guardar as que respeitam os sinais | Só identifica um conjunto; a mediana ponto a ponto não corresponde a nenhum modelo (Kilian); reporte o conjunto ou o modelo mais provável |
| Regressores exógenos | Variáveis que entram em uma equação e não em outra | VAR com `exog` | Exogeneidade em macro é controversa (Sims, 1980) |
| Heterocedasticidade, instrumentos externos (proxy-SVAR) | Mudança de volatilidade ou série de choques externa | Manual | Literatura avançada |

Exemplo de identificação recursiva justificada (prova do curso, mercado de etanol): preço da gasolina (mais exógeno, definido por política) → preço do açúcar → quantidade de etanol → preço do etanol (mais endógeno).

Limitações gerais (curso e Kilian): choques de VAR podem refletir variáveis omitidas; crítica de Lucas — IRFs valem dentro do mesmo regime de política; quebras e parâmetros que mudam no tempo exigem subamostras ou TVP-VAR.

## 10. VAR bayesiano e muitos dados

- **BVAR com prior de Minnesota (Litterman)**: média zero para os coeficientes, exceto 1 na primeira defasagem própria (passeio aleatório); variância menor para defasagens longas e para outras variáveis. Contrai o excesso de parâmetros; historicamente muito bom para previsão. Normal-Wishart permite solução analítica. Python: implementação manual simples (MQO com observações artificiais "dummy observations") ou pacotes externos.
- **FAVAR**: resume dezenas de séries em poucos fatores (componentes principais) e os inclui no VAR (Bernanke, Boivin e Eliasz).

## 11. Previsão

**Avaliação**
- Ajuste dentro da amostra não mede capacidade preditiva. Use **origem móvel** (janela expansiva ou deslizante), reestimando a cada passo; regra do curso para uma divisão simples: 80% estimação, 20% avaliação.

```python
import numpy as np
erros = []
for fim in range(n_inicial, len(y) - h + 1):
    m = SARIMAX(y.iloc[:fim], order=ordem, seasonal_order=sazonal).fit(disp=False)
    erros.append(y.iloc[fim:fim + h].values - m.forecast(h).values)
rmse_por_horizonte = np.sqrt(np.mean(np.square(erros), axis=0))
```

- Compare sempre com: ingênuo (último valor), sazonal ingênuo (mesmo mês do ano anterior), média, ETS. Se o modelo não superar o sazonal ingênuo, diga.
- **A métrica deve casar com o que se quer prever** (IJF, 2022): MSE/RMSE ↔ média condicional; MAE ↔ mediana; MAPE indefinido com zeros e assimétrico; MASE para comparar séries diferentes. Avaliar a mesma previsão pontual com várias métricas pouco informa.
- Intervalos: avalie cobertura (o % de realizações dentro do intervalo de 95% deve ser ≈ 95%) e o *interval score*.
- **Teste de Diebold-Mariano** (com correção de Harvey-Leybourne-Newbold para amostras pequenas) para dizer se a diferença entre dois modelos é significativa: `scripts/diagnosticos.py` tem `diebold_mariano`. Para modelos aninhados, o DM é conservador (use Clark-West).
- Dados revisados (PIB, IBC-Br) e divulgação com defasagem: use as séries como estavam disponíveis na data (vintages) para não superestimar a precisão.

**Intervalos de previsão**
- Fórmula analítica: Var(e_{T+h}) = σ²(1 + ψ₁² + … + ψ²_{h−1}) — converge para a variância incondicional; supõe normalidade e ignora a incerteza dos parâmetros (tende a ser estreito demais).
- **Bootstrap** (curso): estime o AR(p), reamostre os resíduos, gere séries artificiais, reestime e preveja B vezes; os percentis α/2 e 1 − α/2 formam o intervalo. Captura não normalidade e incerteza de parâmetros.

**Combinação de previsões**
- Combinar previsões de métodos diferentes costuma superar o melhor método individual (M3; IJF). Pesos iguais são difíceis de bater na prática ("forecast combination puzzle").
- Alternativas: pesos que minimizam a variância do erro combinado (Bates-Granger), regressão de Granger-Ramanathan (y em previsões, com soma dos pesos = 1), pesos por BIC/Akaike (aproximação bayesiana), inclusive com previsões de consenso (Focus).

## 12. Armadilhas comuns

- Regressão espúria em níveis com tendência.
- Look-ahead: usar dado revisado ou ainda não divulgado na data da previsão.
- Sobreajuste com muitas defasagens/variáveis em amostra curta.
- Ignorar sazonalidade → autocorrelação nos lags 12/24.
- Quebra não tratada (2020) distorcendo coeficientes e testes de raiz unitária.
- Diferenciar demais: série I(0) diferenciada ganha MA não invertível (ACF com −0,5 no lag 1).
- Filtro HP no fim da amostra para decisões em tempo real.
- Interpretar Granger como causalidade e IRF de Cholesky sem justificar a ordem.
