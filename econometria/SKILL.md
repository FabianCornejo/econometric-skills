---
name: econometria
description: Fluxo rigoroso de econometria aplicada em Python (statsmodels, linearmodels, arch) — regressão com diagnóstico completo (RESET, heterocedasticidade, autocorrelação, influência), endogeneidade e variáveis instrumentais, dados em painel, logit/probit/tobit, séries temporais (raiz unitária, quebras, ARIMA, VAR/SVAR, cointegração/VECM), volatilidade (GARCH), previsão com avaliação fora da amostra, e estimação de demanda e elasticidades, com dados do BCB, IBGE e Ipeadata. Use sempre que o usuário pedir regressão, elasticidade-preço, efeito de uma variável sobre outra, impacto de Selic, IPCA ou câmbio sobre vendas, previsão de série temporal, teste de estacionariedade ou cointegração, modelo de painel, probabilidade de inadimplência, efeito de promoção, ou qualquer pergunta do tipo "X explica/causa Y?", mesmo sem a palavra "econometria".
---

# Econometria aplicada

Esta skill transforma uma pergunta econômica ("a Selic afeta minhas vendas?", "qual a elasticidade-preço do combinado?") num modelo estimado, diagnosticado e defensável. O valor está no **rigor do processo**: qualquer um roda `sm.OLS(...).fit()`; o que separa uma análise confiável de uma enganosa são a especificação, os testes de pressupostos, a estratégia de identificação e a honestidade sobre o que o modelo não prova.

A metodologia segue a literatura de referência (Wooldridge, Greene, Hansen, Angrist-Pischke, Cameron-Trivedi, Hamilton, Enders, Stock, Kilian, Huse-Salvo) e os protocolos ensinados em cursos de pós-graduação em econometria e séries temporais. `references/bibliografia.md` indica onde aprofundar cada tópico.

## Recursos desta skill

| Arquivo | Quando ler |
|---|---|
| `scripts/diagnosticos.py` | Sempre que estimar um modelo: baterias de testes com conclusões em português (MQO, influência, Chow, estacionariedade com ADF/KPSS/DF-GLS, cointegração, endogeneidade, sobreidentificação, Hausman, LM de painel, Tobit, Diebold-Mariano) |
| `references/regressao-mqo.md` | Toda regressão: formas funcionais, dummies, ordem dos diagnósticos, HC/HAC/MQG, multicolinearidade, influência, erro de medida |
| `references/series-temporais.md` | Dados com dimensão de tempo: raiz unitária, quebras, ARIMA/ETS/ARFIMA, ADL, cointegração/VECM, VAR/SVAR/BVAR, avaliação e combinação de previsões |
| `references/painel-iv.md` | Endogeneidade, instrumentos, equações simultâneas, painel (FE/RE), diferenças-em-diferenças, painel dinâmico |
| `references/dependente-limitada.md` | y binária, contagem, censurada, fracionária ou ordenada (MPL, logit, probit, tobit, Heckman, PPML) |
| `references/demanda-elasticidades.md` | Preço, promoção, canibalização, repasse de custos, elasticidades |
| `references/volatilidade.md` | Retornos financeiros, câmbio, VaR |
| `references/fontes-de-dados.md` | Buscar dados (BCB/SGS, IBGE/SIDRA, Ipeadata, internacionais) |
| `references/equivalencias-software.md` | Traduzir procedimentos de Stata/EViews para Python |
| `references/bibliografia.md` | Aprofundar um tópico na literatura de referência |

Leia apenas as referências que o caso exige. O script é ponto de partida: leia o código e adapte quando necessário.

## Combinação com outras skills

- `data:statistical-analysis` para a exploração inicial (distribuições, outliers, cuidados com correlação ≠ causalidade); esta skill assume a partir da modelagem.
- A skill de visualização de dados disponível para os gráficos (observado vs. ajustado, resíduos, impulso-resposta, previsão com intervalo).
- A skill do formato de entrega (planilha, documento, apresentação) quando o resultado virar arquivo.
- `avoid-hallucinated-numbers-in-finance`, se disponível, como reforço: nenhum número sem cálculo ou fonte.

## Ambiente

```bash
python3 -c "import statsmodels, linearmodels, arch" 2>/dev/null || \
  pip install --break-system-packages -q statsmodels linearmodels arch
```
Opcionais conforme o caso: `python-bcb`, `sidrapy` (dados), `ruptures` (quebras múltiplas), `pydynpd` (painel dinâmico), `scikit-learn` (ROC/AUC, validação cruzada).

## Fluxo de trabalho

Siga as etapas na ordem; cada uma existe porque pulá-la produz um tipo específico de erro. Explique ao usuário o que está sendo feito em cada etapa e por quê.

### 1. Formular a pergunta e a hipótese

Antes de tocar nos dados, escreva: variável dependente, explicativas, sinal esperado de cada coeficiente e por quê (teoria ou lógica do negócio). Isso evita "pescar" especificações até achar significância.

Classifique o objetivo, porque ele muda o modelo certo:
- **Estimar um efeito** (elasticidade, impacto de política): o foco é coeficiente não viesado → identificação, endogeneidade, variáveis omitidas.
- **Prever**: o foco é erro fora da amostra → validação temporal e comparação com benchmarks ingênuos. Um modelo pode prever bem sem que seus coeficientes tenham interpretação causal.

### 2. Identificar a estrutura dos dados

| Estrutura / tipo de y | Exemplo | Referência |
|---|---|---|
| Corte transversal, y contínua | 500 lojas num mês | `regressao-mqo.md` |
| Série temporal | Vendas mensais 2015–2026 | `series-temporais.md` |
| Painel | 80 lojas × 48 meses | `painel-iv.md` |
| y binária, contagem, censurada | Inadimplência, nº de pedidos, gasto com zeros | `dependente-limitada.md` |
| Preço e quantidade | Elasticidade, promoção | `demanda-elasticidades.md` |
| Retornos | Ibovespa diário | `volatilidade.md` |

### 3. Obter e auditar os dados

- Fontes em `fontes-de-dados.md` (as APIs do BCB, SIDRA e Ipeadata costumam responder a partir do ambiente de trabalho). Registre fonte, código da série, data de extração e unidade.
- **Frequência e alinhamento**: agregue igual (Selic média do mês ≠ Selic no fim do mês).
- **Nominal vs. real**: deflacione valores monetários em horizontes longos (IPCA para consumo).
- **Sazonalidade, feriados, eventos e quebras** (pandemia, mudança de metodologia, novo cardápio).
- **Tamanho amostral**: com menos de ~30 observações por parâmetro, avise que a inferência é frágil.
- **Transformações**: log para elasticidades e variância; diferenças para séries não estacionárias.
- Preserve os dados brutos e faça toda limpeza em código (exclusões documentadas).

### 4. Especificar e estimar

Comece pelo modelo mais simples que responde à pergunta; adicione complexidade só quando um diagnóstico ou a teoria justificar.

```python
import numpy as np, statsmodels.formula.api as smf
m = smf.ols("np.log(vendas) ~ np.log(preco) + np.log(renda) + C(mes)", data=df).fit(cov_type="HC3")
# séries temporais: cov_type="HAC", cov_kwds={"maxlags": int(4*(len(df)/100)**(2/9))}
# painel/grupos: erros clusterizados (ver painel-iv.md)
```

Erros robustos por padrão: heterocedasticidade e autocorrelação não viesam coeficientes, mas invalidam erros-padrão e p-valores. HC3 é preferível ao HC1 (Hansen, 2020).

### 5. Diagnosticar

```bash
python3 scripts/diagnosticos.py mqo dados.csv --y log_vendas --x log_preco,log_renda [--hac 6]
python3 scripts/diagnosticos.py influencia dados.csv --y log_vendas --x log_preco,log_renda
python3 scripts/diagnosticos.py estacionariedade dados.csv --cols selic,ipca,log_vendas
python3 scripts/diagnosticos.py cointegracao dados.csv --cols log_p,log_cambio --lags 2
```
ou `from diagnosticos import ...` em Python.

Ordem (o curso enfatiza: heterocedasticidade e autocorrelação são frequentemente sintomas de especificação errada, então a forma funcional vem primeiro):

| Pressuposto | Teste | Se falhar |
|---|---|---|
| Forma funcional | RESET de Ramsey | Logs, quadráticos, interações; reestimar antes de seguir |
| Homocedasticidade | Breusch-Pagan, White (completo ou com ŷ e ŷ²) | HC3; MQP/MQGF se a forma for conhecida |
| Ausência de autocorrelação | Breusch-Godfrey, Durbin-Watson | HAC; modelar a dinâmica (defasagens) |
| Normalidade | Jarque-Bera | Só importa em amostra pequena |
| Multicolinearidade | VIF, número de condição | Combinar/retirar variáveis ou aceitar se só afeta controles |
| Influência | Alavancagem, DFBETA, max\|h·ẽ\| | Verificar erro de dado; reportar com e sem |
| Estacionariedade (séries) | ADF + KPSS + DF-GLS | Diferenciar ou testar cointegração |
| Estabilidade | Chow, CUSUM, sup-Wald | Dummies de quebra ou subamostras |
| Exogeneidade | Teoria + Durbin-Wu-Hausman com instrumento | IV/2SLS, painel FE, quase-experimento |

**Regressão espúria**: séries com tendência (faturamento e PIB em nível) quase sempre dão R² alto e t significativo sem relação real. Alerta: R² > Durbin-Watson.

### 6. Testar robustez

Uma conclusão só é confiável se sobrevive a variações razoáveis. Faça pelo menos duas:
- especificação alternativa (controles, forma funcional, defasagens);
- subamostra (sem pandemia; períodos separados);
- outro estimador (MQO vs. IV; FE vs. RE; 2SLS vs. LIML; ordem de Cholesky alternativa);
- outra definição de variável ou nível de agregação;
- para previsão: avaliação com origem móvel contra ingênuo, sazonal ingênuo e ETS, com teste de Diebold-Mariano.

Apresente lado a lado (`tabela_comparativa` no script, estilo artigo).

### 7. Interpretar e reportar

Estrutura:
1. **Pergunta e resposta curta**, com magnitude e intervalo: "Um aumento de 1 p.p. na Selic está associado a queda de 0,8% a 1,4% nas vendas após 6 meses (IC 95%)."
2. **Dados**: fontes, período, frequência, transformações, exclusões e por quê.
3. **Modelo**: equação, estimador, tipo de erro-padrão, estratégia de identificação.
4. **Resultados**: coeficientes, erros-padrão, IC, N, R² ajustado (ou pseudo-R²), critérios de informação ao comparar modelos.
5. **Diagnósticos**: teste, estatística, p-valor, conclusão.
6. **Robustez**: o que mudou e o que não mudou.
7. **Limitações**: o que o modelo não prova.

Regras de interpretação:
- **Pela forma funcional** (tabela em `regressao-mqo.md`): log-log → elasticidade; log-lin → 100·β% (exato: 100·(e^β − 1)); lin-log → β/100 por 1%; quadrático → efeito β₁ + 2β₂x.
- **Em modelos não lineares** (logit, probit, tobit), reporte efeitos marginais médios, não coeficientes.
- **Significância estatística ≠ econômica**: reporte magnitude e intervalo, não só p-valor.
- **Associação ≠ causalidade**: linguagem causal só com estratégia de identificação (IV, DiD, experimento, descontinuidade). Granger é precedência preditiva. Sem isso, "está associado a".
- **Direção do viés**: quando a identificação é imperfeita, diga para que lado o viés provavelmente vai.
- **Precisão honesta**: 2–3 algarismos significativos; previsões sempre com intervalo; elasticidades valem para variações pequenas dentro da faixa observada.
- **Nenhum número inventado**: todo coeficiente, estatística ou dado vem de cálculo executado na sessão ou de fonte citada. Se faltar dado, diga.

## Entregáveis

- Análise rápida: resposta na conversa com a tabela principal e os diagnósticos resumidos.
- Análise para guardar ou compartilhar: relatório com as seções acima e o script `.py` reprodutível.
- Para estudo ou reaproveitamento: notebook `.ipynb` comentado passo a passo.
- Gráficos que comunicam o resultado: observado vs. ajustado, resíduos no tempo, impulso-resposta com bandas, previsão com intervalo, efeitos marginais.
