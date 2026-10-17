# Regressão linear (MQO) — especificação, diagnóstico e correções

## Índice
1. Pressupostos e o que cada violação causa
2. Formas funcionais e interpretação
3. Variáveis dummy e interações
4. Testes de restrições (F, Wald, LR)
5. Ordem recomendada dos diagnósticos
6. Heterocedasticidade: testes e correções (HC, MQP, MQGF)
7. Autocorrelação: testes e correções (HAC, MQG)
8. Multicolinearidade
9. Observações influentes
10. Erro de medida
11. Seleção de modelo

## 1. Pressupostos e consequências

| Pressuposto | Se falhar | Consequência para MQO |
|---|---|---|
| Linear nos parâmetros, forma funcional correta | Má especificação | Viés e inconsistência |
| E(ε\|X) = 0 (exogeneidade) | Variável omitida, simultaneidade, erro de medida em X | Viés e inconsistência — e contamina **todos** os coeficientes, não só o da variável endógena |
| Posto completo de X | Multicolinearidade perfeita | Não estima |
| Homocedasticidade | Heterocedasticidade | Coeficientes não viesados e consistentes, mas erros-padrão errados e MQO deixa de ser BLUE |
| Ausência de autocorrelação | Correlação serial | Idem; erros-padrão geralmente subestimados |
| Normalidade | Não normalidade | Só importa para inferência exata em amostras pequenas |

Ponto que os slides do curso enfatizam: **heterocedasticidade e autocorrelação são frequentemente sintomas de forma funcional errada ou variável omitida**. Por isso o diagnóstico começa pela especificação (RESET), não pela correção dos erros-padrão.

Diferença importante (aula de séries temporais): exogeneidade **estrita** E(ε_t\|X) = 0 para todos os períodos garante ausência de viés; exogeneidade **fraca/contemporânea** E(x_t ε_t) = 0 garante só consistência. Modelos com y defasado violam a estrita, então MQO é viesado em amostra pequena, mas consistente se não houver autocorrelação residual.

## 2. Formas funcionais

| Modelo | Equação | Interpretação de β | Elasticidade |
|---|---|---|---|
| Linear | y = α + βx | Δy por unidade de x | β·x/y (avaliar na média) |
| Log-log | ln y = α + β ln x | Elasticidade constante: %Δy por 1% em x | β |
| Log-lin (semilog) | ln y = α + βx | 100·β ≈ %Δy por unidade de x (exato: 100·(e^β − 1)) | β·x |
| Lin-log | y = α + β ln x | β/100 = Δy por 1% em x | β/y |
| Recíproco | y = α + β(1/x) | Efeito decrescente; assíntota em α | −β/(x·y) |
| Log-inverso | ln y = α − β/x | Curva em S | β/x |
| Quadrático | y = α + β₁x + β₂x² | Efeito marginal β₁ + 2β₂x; ponto de inflexão em −β₁/(2β₂) | (β₁ + 2β₂x)·x/y |

Cuidados:
- Para comparar R² entre y e ln y, os modelos não são comparáveis diretamente; compare por critério de informação calculado na mesma escala ou pela previsão fora da amostra.
- Ao prever y a partir de um modelo em ln y, exp(ln ŷ) subestima E(y); corrija com o fator de Duan (média de exp(resíduos)) ou exp(σ²/2) sob normalidade.
- Dummies em modelo log: efeito percentual exato = 100·(exp(β) − 1).

## 3. Dummies e interações

- **Armadilha da dummy**: com intercepto, inclua g − 1 dummies para g categorias. Em fórmulas, `C(categoria)` faz isso automaticamente; escolha a categoria de referência com `C(x, Treatment(reference="..."))`.
- **Dummy de intercepto**: desloca o nível. **Dummy de inclinação** (interação `x:D`): muda o efeito de x entre grupos.
- Codificar categorias como 1, 2, 3, 4 numa variável só impõe efeitos igualmente espaçados — quase sempre errado.
- Interação entre contínuas: o efeito de x₁ passa a depender de x₂ (β₁ + β₁₂·x₂); reporte o efeito em valores relevantes de x₂ (média, quartis).
- Dummies sazonais (`C(mes)`) para dados mensais e dummies de quebra/evento (pandemia, mudança de metodologia).
- Teste de Chow equivalente: modelo com todas as interações com a dummy de grupo; F conjunto nas interações e na dummy.

## 4. Testes de restrições

- **F parcial (modelo restrito vs. irrestrito)**: `res.f_test("x1 = 0, x2 = 0")` ou `res.compare_f_test(restrito)`.
- Hipóteses lineares gerais: `res.f_test("b1 = b2")`, `res.t_test("x1 = 0.25")`, `res.wald_test(...)`.
- Com erros robustos, `f_test`/`wald_test` usam a matriz robusta automaticamente (o teste F clássico de SQR não vale com heterocedasticidade).
- Modelos de máxima verossimilhança (logit/probit/tobit): razão de verossimilhança `LR = 2(llf_irrestrito − llf_restrito) ~ χ²(g)`.

## 5. Ordem recomendada dos diagnósticos

1. Gráficos: y vs. cada x; resíduos vs. ajustados; resíduos vs. ordem/tempo; resíduos² vs. cada x.
2. **RESET** (forma funcional). Se rejeitar, reespecifique antes de qualquer outra coisa.
3. Heterocedasticidade (Breusch-Pagan; White completo ou simplificado com ŷ e ŷ²).
4. Autocorrelação (Durbin-Watson só como indicador; Breusch-Godfrey com p lags).
5. Normalidade (Jarque-Bera) — relevante só para amostras pequenas.
6. Multicolinearidade (VIF, número de condição) e observações influentes.
7. Estabilidade (Chow com data conhecida, CUSUM) quando há tempo ou grupos.
8. Exogeneidade (teoria + teste de Hausman quando houver instrumento) → `painel-iv.md`.

`scripts/diagnosticos.py` cobre 2 a 6 numa chamada (`diagnostico_mqo`), mais `influencia` e `chow`.

## 6. Heterocedasticidade

Testes:
- **Breusch-Pagan**: regride ê² nos X; LM = n·R² ~ χ²(k−1).
- **White**: regride ê² nos X, quadrados e produtos cruzados; capta forma desconhecida, mas consome muitos graus de liberdade. Versão econômica: ê² = δ₀ + δ₁ŷ + δ₂ŷ² + v, H0: δ₁ = δ₂ = 0 (`white_simplificado` no script).

Correções, da mais à menos usada:
1. **Erros-padrão robustos (HC)** — padrão. Hansen (2020) recomenda **HC2 ou HC3** sobre HC1 (o default do Stata): HC2 é não viesado sob homocedasticidade e HC3 é conservador. Só diferem muito quando há observações de alta alavancagem (h_ii perto de 1). Use `cov_type="HC3"`.
2. **MQP (WLS)** quando a forma é conhecida, Var(ε\|x) = σ²h(x): divida tudo por √h(x). `sm.WLS(y, X, weights=1/h)`. Interprete os coeficientes no modelo original.
3. **MQGF (FGLS/FWLS)** com forma desconhecida (Wooldridge): estime MQO; regrida ln(ê²) nos X; ĥ = exp(ĝ); estime WLS com pesos 1/ĥ. Reporte também erros robustos no WLS, pois ĥ pode estar mal especificado.
4. Transformar a variável (log) muitas vezes elimina a heterocedasticidade e melhora a forma funcional ao mesmo tempo.

O R² não é afetado pela heterocedasticidade.

## 7. Autocorrelação

- Em corte transversal, autocorrelação nos resíduos (ordenados por alguma variável) sinaliza **forma funcional errada ou variável omitida**.
- Testes: Durbin-Watson (≈2 sem autocorrelação; inválido com y defasado), **Breusch-Godfrey** (LM = n·R² ~ χ²(p); vale com y defasado), Ljung-Box nos resíduos.
- Correções: **HAC/Newey-West** (`cov_type="HAC", cov_kwds={"maxlags": L}`, L ≈ 4(T/100)^(2/9)); modelar a dinâmica (incluir defasagens — ADL); MQG com erro AR(p) (`sm.GLSAR(y, X, rho=1).iterative_fit()`), equivalente a Cochrane-Orcutt/Prais-Winsten.
- Prefira modelar a dinâmica a "corrigir" o erro: autocorrelação residual quase sempre esconde defasagens omitidas.

## 8. Multicolinearidade

- Não viesa nada; aumenta variâncias. Sintoma clássico: R² alto e F significativo com t individuais fracos, sinais "trocados", coeficientes instáveis ao retirar observações.
- VIF_j = 1/(1 − R²_j): > 5 atenção, > 10 preocupante. Número de condição de X > 30 sugere problema.
- Remédios: mais dados; combinar variáveis (índice, razão); retirar uma das variáveis quase redundantes (cuidado com viés de variável omitida); restrições teóricas; aceitar quando a colinearidade está só entre controles e não afeta o coeficiente de interesse.

## 9. Observações influentes

- **Alavancagem** h_ii: X distante da média. Alavancagem alta dá *potencial* de influência.
- Influência efetiva: mudança no coeficiente ao retirar a observação (**DFBETA** = mudança / erro-padrão) e no ajustado, |h_ii·ẽ_i| (Hansen: estatística "Influence" = max |h_ii·ẽ_i|).
- Distância de Cook e DFFITS existem, mas Hansen alerta que não têm base estatística firme; foque em "a observação muda materialmente o coeficiente de interesse?".
- O que fazer: verificar erro de digitação/registro primeiro; se for erro, corrigir ou excluir **documentando no script** (preserve os dados brutos); se for legítima, reporte resultados com e sem ela. Não exclua só porque é inconveniente.
- `scripts/diagnosticos.py influencia` lista as observações mais influentes.

## 10. Erro de medida

- Em **y**: coeficientes não viesados, só menos precisos.
- Em **x** (clássico): **viés de atenuação** — plim β̂ = β·σ²_x*/(σ²_x* + σ²_erro), sempre em direção a zero. Solução: variável instrumental (outra medida da mesma variável é um bom instrumento).

## 11. Seleção de modelo

- Omitir variável relevante → viés; incluir irrelevante → ineficiência. Na dúvida sobre um controle teoricamente relevante, mantenha-o.
- Abordagem geral-para-específico: comece pelo modelo mais amplo justificável e simplifique com testes F conjuntos, verificando os diagnósticos a cada passo.
- R² ajustado, AIC, BIC (BIC penaliza mais e escolhe modelos menores; preferível quando o objetivo é identificar o modelo; AIC tende a ser melhor para previsão).
- Nunca use significância isolada como único critério de inclusão; reporte quantas especificações foram testadas.
- **Bad controls** (Angrist-Pischke): não controle por variáveis que são elas mesmas resultado do tratamento (ex.: ao medir o efeito de uma promoção sobre o faturamento, não controle pelo número de clientes, que é afetado pela promoção).
