# Volatilidade (família GARCH)

Use para retornos de ativos, câmbio, juros de mercado ou qualquer série cuja variância muda no tempo em *clusters*.

## 1. Fatos estilizados de retornos (verifique antes de modelar)

1. E(r_t\|passado) ≈ 0: retornos quase imprevisíveis na média (martingal em diferença; hipótese fraca de mercado eficiente).
2. Mas E(r_t) > 0 no longo prazo (prêmio de risco).
3. **Caudas pesadas**: curtose > 3 (ações brasileiras tipicamente 7 a 15).
4. **Agrupamento de volatilidade**: retornos ao quadrado autocorrelacionados.
5. Variância maior após fins de semana/feriados.
6. Volatilidade intradiária maior na abertura e no fechamento.
7. **Efeito alavancagem**: quedas aumentam a volatilidade mais que altas.

Procedimento: preço em nível parece ter tendência → ACF de Δln p sem estrutura (ARIMA(0,1,0)) → ACF de r² com estrutura → modelar a variância.

## 2. Preparação
- Retornos log em %: `r = 100 * np.log(p).diff().dropna()` (a escala ajuda o otimizador do `arch`). Use preços ajustados por proventos.
- Teste de efeito ARCH: `het_arch(r - r.mean(), nlags=10)` (H0: sem ARCH) e Ljung-Box em r². Sem efeito ARCH, variância constante basta.

## 3. Especificar e estimar

```python
from arch import arch_model
m = arch_model(r, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="t")
res = m.fit(disp="off")
print(res.summary())
```

| Modelo | `arch_model` | Quando |
|---|---|---|
| GARCH(1,1) | `vol="GARCH", p=1, q=1` | Ponto de partida; costuma bastar (ARCH puro exigiria muitas defasagens) |
| GJR/TARCH | `vol="GARCH", p=1, o=1, q=1` | Efeito alavancagem (dummy para choque negativo) |
| EGARCH | `vol="EGARCH", p=1, o=1, q=1` | Assimetria em log-variância, sem restrição de positividade; γ significativo = alavancagem |
| Média AR / com regressores | `mean="AR", lags=1` ou `mean="ARX", x=...` | Autocorrelação ou efeito dia da semana na média |
| Distribuição | `dist="normal" / "t" / "skewt" / "ged"` | Compare por AIC/BIC; `t` costuma vencer a normal |

O `arch` não aceita regressores na equação da variância; para testar efeito de dia da semana na variância (fato 5), estime com o pacote via modelo customizado ou compare a variância condicional média por dia.

Diagnóstico:
- Persistência α + β (+ γ/2 no GJR) < 1; perto de 1 = choques muito persistentes. Meia-vida = ln 0,5 / ln(α + β).
- Resíduos padronizados (`res.std_resid`): sem autocorrelação neles nem nos quadrados; distribuição compatível com a escolhida.
- Teste de viés de sinal de Engle-Ng para verificar assimetria omitida.

## 4. Previsão e risco

```python
f = res.forecast(horizon=10, reindex=False)
vol = np.sqrt(f.variance.values[-1])           # % por período
vol_anual = vol * np.sqrt(252)                  # diário → anual
nu = res.params["nu"]
q01 = res.model.distribution.ppf(0.01, [nu])    # quantil da t padronizada
var_99 = -(res.params["mu"] + vol[0] * q01)     # VaR 1 dia, 99%, em %
```

Backtesting do VaR: proporção de violações ≈ 1% (Kupiec) e violações não agrupadas (Christoffersen). Avalie previsões de volatilidade contra volatilidade realizada (perda QLIKE).

## 5. Limitações (visão do curso)
- Modelos ARCH são **backward looking**: reagem depois que o choque já ocorreu. Em gestão de risco profissional, a volatilidade implícita de derivativos (forward looking), simulação de Monte Carlo e bootstrap histórico substituíram boa parte do uso de GARCH.
- Muito sensíveis à distribuição assumida para os erros.
- Precisam de muitas observações (> 500 diárias, idealmente).
- Crises e quebras inflam a persistência estimada; teste subamostras.
- Usar preços em vez de retornos é erro conceitual.
