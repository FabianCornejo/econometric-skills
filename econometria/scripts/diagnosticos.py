"""Baterias de diagnóstico econométrico com conclusões em português.

Uso como módulo:
    import sys; sys.path.insert(0, "<pasta da skill>/scripts")
    from diagnosticos import *

Uso pela linha de comando (CSV ou XLSX; coluna de data opcional via --data):
    python3 diagnosticos.py mqo dados.csv --y log_vendas --x log_preco,log_renda [--hac 6]
    python3 diagnosticos.py influencia dados.csv --y log_vendas --x log_preco,log_renda
    python3 diagnosticos.py chow dados.csv --y log_vendas --x log_preco --quebra 2020-03-01
    python3 diagnosticos.py estacionariedade dados.csv --cols selic,ipca,log_vendas
    python3 diagnosticos.py cointegracao dados.csv --cols log_p,log_cambio,log_pext [--lags 2]

Funções principais
    diagnostico_mqo(res)                 RESET, BP, White, White simplificado, DW, BG, JB, VIF
    influencia(res)                      alavancagem, DFBETA, max|h*e~| (Hansen)
    chow(y, X, ponto)                    teste de Chow para quebra em data conhecida
    comparar_erros(formula, dados)       clássico vs HC3 vs HAC
    tabela_comparativa({...})            várias especificações lado a lado
    estacionariedade(serie)              ADF + KPSS + DF-GLS, nível e diferença
    cointegracao(df)                     Engle-Granger + Johansen
    teste_endogeneidade(...)             Durbin-Wu-Hausman por regressão (função de controle)
    sobreidentificacao(...)              nR² de Wooldridge e Sargan
    hausman(fe, re)                      FE vs RE (linearmodels)
    lm_breusch_pagan_painel(...)         pooled vs efeitos aleatórios
    tobit(y, X)                          Tobit censurado em um limite (MV)
    diebold_mariano(e1, e2, h)           igualdade de acurácia de previsões (correção HLN)

Nível de significância padrão: 5%.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, coint, kpss

ALFA = 0.05


def _linha(teste, h0, estat, p, conclusao):
    return {"Teste": teste, "H0": h0, "Estatística": estat, "p-valor": p, "Conclusão": conclusao}


def _fmt(v):
    return f"{v:.4f}"


def lags_newey_west(n: int) -> int:
    """Regra de bolso de Newey-West para o nº de defasagens HAC."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def _ols_simples(res):
    """Reestima o mesmo modelo com covariância clássica (os testes usam resíduos e X)."""
    return sm.OLS(res.model.endog, res.model.exog).fit()


# =============================================================================
# MQO
# =============================================================================
def white_simplificado(res):
    """White com ŷ e ŷ² (economiza graus de liberdade). Retorna (LM, p)."""
    base = _ols_simples(res)
    e2 = base.resid ** 2
    yhat = base.fittedvalues
    Z = sm.add_constant(np.column_stack([yhat, yhat ** 2]))
    aux = sm.OLS(e2, Z).fit()
    lm = len(e2) * aux.rsquared
    return lm, 1 - stats.chi2.cdf(lm, 2)


def diagnostico_mqo(res, nlags_bg: int | None = None, serie_temporal: bool = False,
                    imprimir: bool = True) -> pd.DataFrame:
    """Bateria completa para um resultado OLS do statsmodels (qualquer cov_type).

    Ordem: forma funcional → heterocedasticidade → autocorrelação → normalidade.
    O tipo de erro-padrão escolhido não altera os diagnósticos.
    """
    base = _ols_simples(res)
    resid = np.asarray(base.resid)
    exog = np.asarray(base.model.exog)
    n = len(resid)
    nlags_bg = nlags_bg or (12 if serie_temporal else lags_newey_west(n))
    linhas = []

    try:
        reset = linear_reset(base, power=3, use_f=True)
        p = float(reset.pvalue)
        linhas.append(_linha("RESET de Ramsey (ŷ², ŷ³)", "forma funcional correta", float(reset.fvalue), p,
                             "Má especificação provável: reespecificar ANTES de corrigir erros-padrão"
                             if p < ALFA else "Sem evidência de má especificação"))
    except Exception as e:  # noqa: BLE001
        linhas.append(_linha("RESET de Ramsey", "forma funcional correta", np.nan, np.nan, f"não calculado ({e})"))

    lm, p_bp, _, _ = het_breuschpagan(resid, exog)
    linhas.append(_linha("Breusch-Pagan", "homocedasticidade", lm, p_bp,
                         "Heterocedasticidade: usar HC3 (ou MQP/MQGF)" if p_bp < ALFA else "Sem evidência"))
    try:
        lm_w, p_w, _, _ = het_white(resid, exog)
        linhas.append(_linha("White (completo)", "homocedasticidade", lm_w, p_w,
                             "Heterocedasticidade" if p_w < ALFA else "Sem evidência"))
    except Exception:  # noqa: BLE001  (falha com muitos regressores/dummies)
        pass
    lm_ws, p_ws = white_simplificado(res)
    linhas.append(_linha("White simplificado (ŷ, ŷ²)", "homocedasticidade", lm_ws, p_ws,
                         "Heterocedasticidade" if p_ws < ALFA else "Sem evidência"))

    dw = durbin_watson(resid)
    linhas.append(_linha("Durbin-Watson", "sem autocorrelação de 1ª ordem (≈2)", dw, np.nan,
                         "Autocorrelação positiva provável" if dw < 1.5 else
                         "Autocorrelação negativa provável" if dw > 2.5 else "Próximo de 2"))
    try:
        try:
            bg = acorr_breusch_godfrey(base, nlags=nlags_bg, result_object=False)
        except TypeError:  # versões antigas do statsmodels
            bg = acorr_breusch_godfrey(base, nlags=nlags_bg)
        lm_bg, p_bg = bg[0], bg[1]
        linhas.append(_linha(f"Breusch-Godfrey ({nlags_bg} lags)", "sem autocorrelação", lm_bg, p_bg,
                             "Autocorrelação: HAC e/ou modelar a dinâmica (ou variável omitida/forma funcional)"
                             if p_bg < ALFA else "Sem evidência"))
    except Exception:  # noqa: BLE001
        pass

    jb, p_jb, assim, curt = jarque_bera(resid)
    linhas.append(_linha("Jarque-Bera", "resíduos normais", jb, p_jb,
                         (f"Não normais (assimetria {assim:.2f}, curtose {curt:.2f}); "
                          + ("relevante: amostra pequena" if n < 100 else "pouco relevante com N grande"))
                         if p_jb < ALFA else "Sem evidência contra normalidade"))

    if serie_temporal and base.rsquared > dw:
        linhas.append(_linha("R² > DW", "—", base.rsquared, np.nan,
                             "ALERTA de regressão espúria: testar estacionariedade/cointegração"))

    tabela = pd.DataFrame(linhas)
    vif = tabela_vif(res)
    if imprimir:
        print(f"\nDiagnóstico MQO (N={n}, R²={base.rsquared:.3f}, R² aj.={base.rsquared_adj:.3f}, "
              f"nº de condição={np.linalg.cond(exog):.1f})")
        print(tabela.to_string(index=False, float_format=_fmt))
        if not vif.empty:
            print("\nVIF (>10 preocupante; >5 atenção):")
            print(vif.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return tabela


def tabela_vif(res) -> pd.DataFrame:
    exog = np.asarray(res.model.exog)
    nomes = list(res.model.exog_names)
    linhas = []
    for i, nome in enumerate(nomes):
        if nome.lower() in ("const", "intercept") or np.ptp(exog[:, i]) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linhas.append({"Variável": nome, "VIF": variance_inflation_factor(exog, i)})
    return pd.DataFrame(linhas)


def influencia(res, top: int = 10, imprimir: bool = True) -> pd.DataFrame:
    """Observações mais influentes: alavancagem h_ii, |h_ii·ẽ_i| (Hansen) e o maior |DFBETA|.

    ẽ_i = resíduo de previsão leave-one-out = ê_i/(1−h_ii). Não há limiar mágico:
    a pergunta é se a observação muda materialmente o coeficiente de interesse.
    """
    base = _ols_simples(res)
    inf = base.get_influence()
    h = inf.hat_matrix_diag
    e_loo = base.resid / (1 - h)
    dfb = pd.DataFrame(inf.dfbetas, columns=list(res.model.exog_names),
                       index=getattr(res.model.data, "row_labels", None))
    out = pd.DataFrame({
        "alavancagem_h": h,
        "influencia_|h·ẽ|": np.abs(h * e_loo),
        "maior_|DFBETA|": dfb.abs().max(axis=1).values,
        "coef_mais_afetado": dfb.abs().idxmax(axis=1).values,
        "cook": inf.cooks_distance[0],
    }, index=dfb.index).sort_values("influencia_|h·ẽ|", ascending=False)
    if imprimir:
        k, n = base.model.exog.shape[1], len(h)
        print(f"\nInfluência (N={n}). Alavancagem média = k/n = {k/n:.3f}; "
              f"atenção para h > {2*k/n:.3f} (2k/n) e |DFBETA| > {2/np.sqrt(n):.3f} (2/√n).")
        print(f"Estatística de influência de Hansen (max |h·ẽ|) = {out['influencia_|h·ẽ|'].iloc[0]:.4f} "
              f"— compare com o desvio-padrão de y ({np.std(base.model.endog):.4f}).")
        print(out.head(top).to_string(float_format=_fmt))
        print("Verifique se são erros de registro; se forem legítimas, reporte resultados com e sem elas.")
    return out


def chow(y, X, ponto, imprimir: bool = True) -> dict:
    """Teste de Chow para quebra em ponto conhecido.

    y: Series; X: DataFrame (sem constante; ela é adicionada). ponto: rótulo do índice
    a partir do qual começa o segundo regime. Sensível a heterocedasticidade: com variâncias
    diferentes entre regimes, prefira a versão com interações e erros robustos.
    """
    dados = pd.concat([y, X], axis=1).dropna()
    yy, XX = dados.iloc[:, 0], sm.add_constant(dados.iloc[:, 1:])
    pos = dados.index >= ponto
    k = XX.shape[1]
    n1, n2 = (~pos).sum(), pos.sum()
    if min(n1, n2) <= k:
        raise ValueError("Observações insuficientes em um dos regimes.")
    sqr = sm.OLS(yy, XX).fit().ssr
    sqr1 = sm.OLS(yy[~pos], XX[~pos]).fit().ssr
    sqr2 = sm.OLS(yy[pos], XX[pos]).fit().ssr
    F = ((sqr - sqr1 - sqr2) / k) / ((sqr1 + sqr2) / (n1 + n2 - 2 * k))
    p = 1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)
    # Versão robusta: interações com dummy de regime e Wald HC3
    D = pos.astype(float)
    XI = pd.concat([XX, XX.mul(D, axis=0).add_suffix("_x_pos")], axis=1)
    rob = sm.OLS(yy, XI).fit(cov_type="HC3")
    R = np.zeros((k, XI.shape[1])); R[:, k:] = np.eye(k)
    w = rob.wald_test(R, scalar=True)
    out = {"F": F, "p": p, "wald_robusto": float(w.statistic), "p_robusto": float(w.pvalue),
           "n1": int(n1), "n2": int(n2)}
    if imprimir:
        print(f"\nChow em {ponto}: F({k},{n1+n2-2*k}) = {F:.3f}, p = {p:.4f}; "
              f"Wald robusto (HC3) = {out['wald_robusto']:.3f}, p = {out['p_robusto']:.4f} → "
              + ("quebra estrutural" if out["p_robusto"] < ALFA else "sem evidência de quebra"))
    return out


def comparar_erros(modelo_formula: str, dados: pd.DataFrame, hac_lags: int | None = None) -> pd.DataFrame:
    """Mesmo modelo com erros clássicos, HC3 e (opcional) HAC: mostra quanto a inferência muda."""
    import statsmodels.formula.api as smf
    versoes = {"Clássico": smf.ols(modelo_formula, dados).fit(),
               "HC3": smf.ols(modelo_formula, dados).fit(cov_type="HC3")}
    if hac_lags:
        versoes[f"HAC({hac_lags})"] = smf.ols(modelo_formula, dados).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    partes = [pd.DataFrame({"coef": r.params, f"ep_{n}": r.bse, f"p_{n}": r.pvalues}) for n, r in versoes.items()]
    out = pd.concat(partes, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def tabela_comparativa(resultados: dict, estrelas: bool = True):
    """Tabela estilo artigo com várias especificações (resultados OLS do statsmodels)."""
    from statsmodels.iolib.summary2 import summary_col
    return summary_col(list(resultados.values()), model_names=list(resultados.keys()), stars=estrelas,
                       info_dict={"N": lambda r: f"{int(r.nobs)}",
                                  "R² aj.": lambda r: f"{getattr(r, 'rsquared_adj', np.nan):.3f}"})


# =============================================================================
# Estacionariedade
# =============================================================================
def _testes(x: pd.Series, reg: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, regression=reg, autolag="AIC")
        kp = kpss(x, regression=reg, nlags="auto")
        try:
            from arch.unitroot import DFGLS
            dg = DFGLS(x, trend=reg, method="bic")
            dg_stat, dg_p = dg.stat, dg.pvalue
        except Exception:  # noqa: BLE001
            dg_stat = dg_p = np.nan
    return adf[0], adf[1], adf[2], kp[0], kp[1], dg_stat, dg_p


def _veredito(p_adf, p_kpss, p_dfgls):
    p_ur = p_dfgls if not np.isnan(p_dfgls) else p_adf   # DF-GLS tem mais poder (Stock)
    rej_ur, rej_kpss = p_ur < ALFA, p_kpss < ALFA
    if rej_ur and not rej_kpss:
        return "Estacionária"
    if not rej_ur and rej_kpss:
        return "Raiz unitária"
    if rej_ur and rej_kpss:
        return "Conflito (tendência/quebra?)"
    return "Inconclusivo (persistência alta ou memória longa?)"


def estacionariedade(serie: pd.Series, nome: str | None = None, imprimir: bool = True) -> pd.DataFrame:
    """ADF, KPSS e DF-GLS com constante e com tendência, em nível e na 1ª diferença.

    O veredito usa DF-GLS (mais poder) + KPSS (nula oposta). p-valor do KPSS é interpolado
    numa tabela limitada a [0,01; 0,10]: valores nos extremos significam '≤0,01' ou '≥0,10'.
    """
    nome = nome or (serie.name if serie.name is not None else "série")
    x = pd.Series(serie).dropna()
    linhas = []
    for rotulo, dados in (("nível", x), ("1ª diferença", x.diff().dropna())):
        for reg, desc in (("c", "constante"), ("ct", "constante+tendência")):
            a, pa, lags, k, pk, g, pg = _testes(dados, reg)
            linhas.append({"Série": nome, "Transformação": rotulo, "Determinísticos": desc,
                           "ADF": a, "p ADF": pa, "lags": lags, "DF-GLS": g, "p DF-GLS": pg,
                           "KPSS": k, "p KPSS": pk, "Veredito": _veredito(pa, pk, pg)})
    tab = pd.DataFrame(linhas)

    def ver(transf, det):
        return tab.loc[(tab["Transformação"] == transf) & (tab["Determinísticos"] == det), "Veredito"].iloc[0]
    vc, vt = ver("nível", "constante"), ver("nível", "constante+tendência")
    if vc == "Estacionária":
        ordem = "I(0)"
    elif vt == "Estacionária":
        ordem = ("I(0) em torno de tendência determinística (ou com quebra) — com só constante os testes "
                 "divergem/rejeitam; confira se a tendência é significativa e plausível")
    elif ver("1ª diferença", "constante") == "Estacionária":
        ordem = "I(1)"
    else:
        ordem = "indefinida — inspecionar gráfico; testar quebra (Zivot-Andrews) ou memória longa (ARFIMA)"
    tab.attrs["ordem_sugerida"] = ordem
    if imprimir:
        print(f"\nEstacionariedade — {nome} (N={len(x)}): {ordem}")
        print(tab.drop(columns="Série").to_string(index=False, float_format=_fmt))
    return tab


def estacionariedade_varias(df: pd.DataFrame, imprimir: bool = True) -> pd.DataFrame:
    resumo = []
    for c in df.columns:
        t = estacionariedade(df[c], c, imprimir=imprimir)
        resumo.append({"Série": c, "Ordem sugerida": t.attrs["ordem_sugerida"]})
    out = pd.DataFrame(resumo)
    if imprimir:
        print("\nResumo:\n" + out.to_string(index=False))
    return out


# =============================================================================
# Cointegração
# =============================================================================
def cointegracao(df: pd.DataFrame, k_ar_diff: int = 1, det_order: int = 0, imprimir: bool = True) -> dict:
    """Engle-Granger (1ª coluna dependente) e Johansen (traço e máx. autovalor, 5%).

    det_order: -1 sem determinísticos, 0 constante, 1 tendência linear.
    k_ar_diff: defasagens em diferenças (escolha com vecm.select_order).
    Pré-requisito: todas as séries I(1).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    dados = df.dropna()
    y, xs = dados.iloc[:, 0], dados.iloc[:, 1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg_t, eg_p, _ = coint(y, xs, trend="c" if det_order <= 0 else "ct", autolag="aic")
    joh = coint_johansen(dados.values, det_order, k_ar_diff)
    k = dados.shape[1]
    linhas, r_tr, r_mx = [], None, None
    for r in range(k):
        tr, tr_cv, mx, mx_cv = joh.lr1[r], joh.cvt[r, 1], joh.lr2[r], joh.cvm[r, 1]
        linhas.append({"H0: posto ≤": r, "Traço": tr, "VC 5% traço": tr_cv, "Rejeita (traço)": tr > tr_cv,
                       "Máx. autovalor": mx, "VC 5% máx.": mx_cv, "Rejeita (máx.)": mx > mx_cv})
        if r_tr is None and tr <= tr_cv:
            r_tr = r
        if r_mx is None and mx <= mx_cv:
            r_mx = r
    r_tr = k if r_tr is None else r_tr
    r_mx = k if r_mx is None else r_mx
    tab = pd.DataFrame(linhas)
    out = {"engle_granger": {"estatistica": eg_t, "p": eg_p,
                             "conclusao": "Cointegradas" if eg_p < ALFA else "Sem evidência de cointegração"},
           "johansen": tab, "posto_traco": r_tr, "posto_max_autovalor": r_mx}
    if imprimir:
        print(f"\nEngle-Granger ({y.name} ~ {', '.join(map(str, xs.columns))}): t={eg_t:.3f}, p={eg_p:.4f} "
              f"→ {out['engle_granger']['conclusao']} (valores críticos de MacKinnon)")
        print(f"Johansen (det_order={det_order}, k_ar_diff={k_ar_diff}):")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"Posto de cointegração: traço = {r_tr}, máx. autovalor = {r_mx}"
              + ("" if r_tr == r_mx else " (divergem: reportar ambos e decidir pela teoria)"))
        if r_tr == k:
            print("Posto pleno: as séries provavelmente são I(0) — reveja a estacionariedade.")
    return out


# =============================================================================
# Endogeneidade e IV
# =============================================================================
def teste_endogeneidade(y, X_exog, x_endog, Z, cov_type: str = "HC3", imprimir: bool = True) -> dict:
    """Durbin-Wu-Hausman por regressão (função de controle), como no curso e no Wooldridge.

    1) x_endog ~ X_exog + Z (forma reduzida) → resíduos v̂
    2) y ~ x_endog + X_exog + v̂ por MQO; v̂ significativo ⇒ x_endog é endógena.
    y: Series; X_exog: DataFrame (sem constante); x_endog: DataFrame com 1+ colunas; Z: DataFrame de instrumentos.
    Também reporta o F dos instrumentos excluídos na 1ª etapa (relevância).
    """
    d = pd.concat([y, X_exog, x_endog, Z], axis=1).dropna()
    yy = d[y.name]
    Xe = sm.add_constant(d[X_exog.columns]) if X_exog is not None and len(X_exog.columns) else sm.add_constant(
        pd.DataFrame(index=d.index))
    residuos, f_primeira = {}, {}
    for c in x_endog.columns:
        Zfull = pd.concat([Xe, d[Z.columns]], axis=1)
        fe = sm.OLS(d[c], Zfull).fit(cov_type=cov_type)
        R = np.zeros((len(Z.columns), Zfull.shape[1])); R[:, -len(Z.columns):] = np.eye(len(Z.columns))
        f_primeira[c] = float(fe.wald_test(R, scalar=True, use_f=True).statistic)
        residuos[f"vhat_{c}"] = fe.resid
    V = pd.DataFrame(residuos)
    est = sm.OLS(yy, pd.concat([Xe, d[x_endog.columns], V], axis=1)).fit(cov_type=cov_type)
    Rv = np.zeros((V.shape[1], est.params.shape[0])); Rv[:, -V.shape[1]:] = np.eye(V.shape[1])
    w = est.wald_test(Rv, scalar=True, use_f=True)
    out = {"F_primeira_etapa": f_primeira, "estatistica": float(w.statistic), "p": float(w.pvalue)}
    if imprimir:
        for c, f in f_primeira.items():
            aviso = "forte" if f > 104.7 else ("aceitável (>10), reporte também Anderson-Rubin" if f > 10 else "FRACO")
            print(f"\n1ª etapa para {c}: F dos instrumentos excluídos = {f:.2f} → {aviso}")
        print(f"Durbin-Wu-Hausman (função de controle, {cov_type}): F = {out['estatistica']:.3f}, p = {out['p']:.4f} → "
              + ("endógena: usar IV/2SLS" if out["p"] < ALFA else "sem evidência de endogeneidade: MQO é consistente e mais eficiente"))
    return out


def sobreidentificacao(resid_2sls, exogenas_incluidas: pd.DataFrame | None, instrumentos: pd.DataFrame,
                       n_endogenas: int, imprimir: bool = True) -> dict:
    """Teste de restrições sobreidentificadoras (Wooldridge nR²; Sargan (n−k)R²).

    Regride os resíduos do 2SLS em todas as exógenas incluídas e nos instrumentos excluídos.
    q = nº de instrumentos excluídos − nº de endógenas (precisa ser ≥ 1).
    Supõe homocedasticidade; com heterocedasticidade use o J de Hansen (IVGMM ... .j_stat).
    """
    partes = [instrumentos] if exogenas_incluidas is None else [exogenas_incluidas, instrumentos]
    Z = sm.add_constant(pd.concat(partes, axis=1))
    aux = sm.OLS(np.asarray(resid_2sls), Z).fit()
    n, k = len(resid_2sls), Z.shape[1]
    q = instrumentos.shape[1] - n_endogenas
    if q < 1:
        raise ValueError("Modelo exatamente identificado: não há restrições sobreidentificadoras para testar.")
    nr2, sargan = n * aux.rsquared, (n - k) * aux.rsquared
    p = 1 - stats.chi2.cdf(nr2, q)
    if imprimir:
        print(f"\nSobreidentificação: nR² = {nr2:.3f} (Sargan (n−k)R² = {sargan:.3f}), χ²({q}), p = {p:.4f} → "
              + ("ao menos um instrumento inválido" if p < ALFA else
                 "sem evidência contra a validade (não prova validade)"))
    return {"nR2": nr2, "sargan": sargan, "q": q, "p": p}


def hausman(fe_res, re_res, imprimir: bool = True) -> dict:
    """Hausman clássico FE vs RE (linearmodels). Estime ambos com cov_type="unadjusted"."""
    comuns = [c for c in fe_res.params.index if c in re_res.params.index and c.lower() not in ("const", "intercept")]
    b = fe_res.params[comuns] - re_res.params[comuns]
    v = fe_res.cov.loc[comuns, comuns] - re_res.cov.loc[comuns, comuns]
    autoval = np.linalg.eigvalsh(v.values)
    estat = float(b.values @ np.linalg.pinv(v.values) @ b.values)
    gl = len(comuns)
    p = float(1 - stats.chi2.cdf(estat, gl))
    valido = bool(np.all(autoval > -1e-10))
    concl = "Rejeita H0 → efeitos fixos" if p < ALFA else "Não rejeita H0 → efeitos aleatórios aceitável"
    if not valido:
        concl += " (ATENÇÃO: V_FE − V_RE não é positiva semidefinida; prefira o teste de Mundlak)"
    out = {"estatistica": estat, "gl": gl, "p": p, "valido": valido, "conclusao": concl}
    if imprimir:
        print(f"\nHausman: χ²({gl}) = {estat:.3f}, p = {p:.4f} → {concl}")
    return out


def lm_breusch_pagan_painel(resid_pooled: pd.Series, imprimir: bool = True) -> dict:
    """LM de Breusch-Pagan (xttest0): H0 = variância do efeito individual é zero (pooled ok).

    resid_pooled: resíduos do pooled OLS com MultiIndex (entidade, tempo). Painel balanceado
    na fórmula clássica; com painel desbalanceado é uma aproximação.
    """
    e = resid_pooled
    grupos = e.groupby(level=0)
    N = grupos.ngroups
    T = grupos.size().mean()
    num = (grupos.sum() ** 2).sum()
    den = (e ** 2).sum()
    lm = (N * T) / (2 * (T - 1)) * (num / den - 1) ** 2
    p = 1 - stats.chi2.cdf(lm, 1)
    if imprimir:
        print(f"\nLM de Breusch-Pagan (painel): {lm:.3f}, p = {p:.4f} → "
              + ("há efeito individual: pooled inadequado (use RE ou FE)" if p < ALFA else "pooled aceitável"))
    return {"LM": lm, "p": p}


# =============================================================================
# Tobit
# =============================================================================
def tobit(y, X, limite_inferior: float = 0.0, imprimir: bool = True) -> dict:
    """Tobit censurado à esquerda por MV. X sem constante (é adicionada).

    Retorna coeficientes (variável latente), sigma, erros-padrão (inversa da hessiana numérica)
    e efeitos marginais médios sobre E(y|x) = Φ(xβ/σ)·β (incondicional).
    """
    d = pd.concat([y, X], axis=1).dropna()
    yy = d.iloc[:, 0].values
    XX = sm.add_constant(d.iloc[:, 1:]).values
    nomes = ["const"] + list(d.columns[1:])
    cens = yy <= limite_inferior

    def nll(theta):
        b, log_s = theta[:-1], theta[-1]
        s = np.exp(log_s)
        xb = XX @ b
        ll_c = stats.norm.logcdf((limite_inferior - xb[cens]) / s)
        ll_u = stats.norm.logpdf((yy[~cens] - xb[~cens]) / s) - log_s
        return -(ll_c.sum() + ll_u.sum())

    b0 = np.linalg.lstsq(XX, yy, rcond=None)[0]
    theta0 = np.r_[b0, np.log(np.std(yy - XX @ b0))]
    opt = optimize.minimize(nll, theta0, method="BFGS")
    if not opt.success:  # perda de precisão do BFGS é comum perto do ótimo; refina com outro método
        opt2 = optimize.minimize(nll, opt.x, method="Nelder-Mead",
                                 options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
        if opt2.fun <= opt.fun:
            opt = opt2
    from statsmodels.tools.numdiff import approx_fprime, approx_hess
    grad_norm = float(np.max(np.abs(approx_fprime(opt.x, nll))))
    opt.success = bool(opt.success or grad_norm < 1e-3)
    H = approx_hess(opt.x, nll)
    cov = np.linalg.inv(H)
    ep = np.sqrt(np.diag(cov))
    b, s = opt.x[:-1], np.exp(opt.x[-1])
    escala = stats.norm.cdf((XX @ b - limite_inferior) / s).mean()
    tab = pd.DataFrame({"coef_latente": b, "ep": ep[:-1], "z": b / ep[:-1],
                        "p": 2 * (1 - stats.norm.cdf(np.abs(b / ep[:-1]))),
                        "efeito_marginal_E[y|x]": b * escala}, index=nomes)
    out = {"tabela": tab, "sigma": s, "llf": -opt.fun, "convergiu": opt.success,
           "n_censuradas": int(cens.sum()), "n": len(yy)}
    if imprimir:
        print(f"\nTobit (censura em {limite_inferior}; {cens.sum()} de {len(yy)} censuradas), σ = {s:.4f}, "
              f"log-veross. = {-opt.fun:.2f}, convergiu = {opt.success}")
        print(tab.to_string(float_format=_fmt))
        print("Efeito marginal sobre E(y|x) = Φ(xβ/σ)·β, média na amostra.")
    return out


# =============================================================================
# Previsão
# =============================================================================
def diebold_mariano(e1, e2, h: int = 1, potencia: int = 2, imprimir: bool = True) -> dict:
    """Diebold-Mariano com correção de Harvey-Leybourne-Newbold (amostras pequenas).

    e1, e2: erros de previsão (real − previsto) dos modelos 1 e 2 no mesmo período;
    h: horizonte; potencia 2 = perda quadrática, 1 = absoluta.
    Estatística negativa ⇒ modelo 1 mais preciso.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = dbar / np.sqrt(var_d)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * (1 - stats.t.cdf(abs(hln), T - 1))
    if imprimir:
        melhor = "modelo 1" if dbar < 0 else "modelo 2"
        print(f"\nDiebold-Mariano (HLN): {hln:.3f}, p = {p:.4f} → "
              + (f"diferença significativa, {melhor} mais preciso" if p < ALFA else "sem diferença significativa"))
    return {"DM_HLN": hln, "p": p, "perda_media_dif": dbar}


# =============================================================================
# CLI
# =============================================================================
def _ler(caminho, col_data):
    df = pd.read_csv(caminho) if caminho.lower().endswith(".csv") else pd.read_excel(caminho)
    if col_data and col_data in df.columns:
        bruto = df[col_data].astype(str)
        iso = pd.to_datetime(bruto, format="ISO8601", errors="coerce")       # 2020-03-01
        br = pd.to_datetime(bruto, dayfirst=True, errors="coerce")           # 01/03/2020
        df[col_data] = iso if iso.notna().mean() >= br.notna().mean() else br
        df = df.set_index(col_data).sort_index()
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acao", choices=["mqo", "influencia", "chow", "estacionariedade", "cointegracao"])
    ap.add_argument("arquivo")
    ap.add_argument("--y")
    ap.add_argument("--x", help="regressores separados por vírgula")
    ap.add_argument("--cols", help="colunas separadas por vírgula")
    ap.add_argument("--data", default="data", help="coluna de data (opcional)")
    ap.add_argument("--hac", type=int, help="defasagens HAC (séries temporais)")
    ap.add_argument("--lags", type=int, default=1, help="k_ar_diff do Johansen")
    ap.add_argument("--quebra", help="data/rótulo do início do 2º regime (Chow)")
    a = ap.parse_args()
    df = _ler(a.arquivo, a.data)

    if a.acao in ("mqo", "influencia", "chow"):
        xs = a.x.split(",")
        dados = df[[a.y] + xs].dropna()
        if a.acao == "chow":
            ponto = pd.Timestamp(a.quebra) if isinstance(dados.index, pd.DatetimeIndex) else type(dados.index[0])(a.quebra)
            chow(dados[a.y], dados[xs], ponto)
            return
        cov = {"cov_type": "HAC", "cov_kwds": {"maxlags": a.hac}} if a.hac else {"cov_type": "HC3"}
        res = sm.OLS(dados[a.y], sm.add_constant(dados[xs])).fit(**cov)
        if a.acao == "mqo":
            print(res.summary())
            diagnostico_mqo(res, serie_temporal=bool(a.hac))
        else:
            influencia(res)
    elif a.acao == "estacionariedade":
        estacionariedade_varias(df[a.cols.split(",")])
    else:
        cointegracao(df[a.cols.split(",")], k_ar_diff=a.lags)


if __name__ == "__main__":
    main()
