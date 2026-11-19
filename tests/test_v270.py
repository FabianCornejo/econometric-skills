"""v2.7.0: version, CHANGELOG, documents and the two new "find the error" cases.

The functions of this release are tested by module (tests/test_backtest.py, test_forecasting.py, test_reporting.py);
this file checks only the documents and the evals. The graders are measured on a corpus written here, 12 correct and
12 incorrect replies per case (lesson of v2.6.11: a grader is measured on both sides before it is trusted), and they
must not match the request itself.

Run from the repository root:  python3 tests/test_v270.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import reporting  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


SK = ROOT / "econometrics"
read = lambda p: (ROOT / p).read_text(encoding="utf-8")  # noqa: E731

# ============================================================ 1. version and CHANGELOG
mk = json.loads(read(".claude-plugin/marketplace.json"))
ver = mk["plugins"][0]["version"] if "plugins" in mk else mk["version"]
check("version 2.7.0 in the marketplace and in reporting.SKILL_VERSION", ver == "2.7.0" and reporting.SKILL_VERSION == "2.7.0")
cl = read("CHANGELOG.md")
m = re.search(r"^## v2\.7\.0 — (.*?)(?=^## v)", cl, re.S | re.M)
check("CHANGELOG: a v2.7.0 entry, above v2.6.13", m is not None and cl.index("## v2.7.0") < cl.index("## v2.6.13"))
body = m.group(1) if m else ""
check("CHANGELOG v2.7.0 names every new function and the files= notice",
      all(k in body for k in ("walk_forward_splits(..., horizon=h)", "purged_kfold_splits", "label_overlap",
                              "conformal_intervals", "study_card", "files=")))
check("CHANGELOG v2.7.0 says the h > 1 ACI bound is derived, not published, and gives the measured coverage",
      "not published" in body and "84–95%" in body and "93–95%" in body)

# ============================================================ 2. SKILL.md
sk = read("econometrics/SKILL.md")
sec2 = sk[sk.index("## 2. Route the question"):sk.index("## 3.")]
sec3 = sk[sk.index("## 3. Consultant workflow"):sk.index("## 4.")]
sec4 = sk[sk.index("## 4. Non-negotiable rules"):sk.index("## 5.")]
sec5 = sk[sk.index("## 5. Output contract"):sk.index("## 6.")]
check("SKILL.md §5: findings with level, message and action, and `used`",
      "`level` (\"warning\" or \"notice\"), `message` and `action`" in sec5 and "`used`" in sec5)
check("SKILL.md §3 step 0: the card is written with reporting.study_card; forecasts get a short card",
      "`reporting.study_card` before the first estimate" in sec3 and "a forecast gets a short card" in sec3)
check("SKILL.md §4: the manifest takes files= and study_card=", "`files=` and the study card in `study_card=`" in sec4)
fc_line = next(ln for ln in sec2.splitlines() if ln.startswith("| Forecasts and model-based strategies"))
check("SKILL.md §2: one new function name on an existing line (T6)", fc_line.count("conformal_intervals") == 1
      and "purged_kfold_splits" not in sec2)
desc = yaml.safe_load(sk.split("---")[1])["description"]
check("SKILL.md description unchanged in this release (no new routing words)", "conformal" not in desc and
      "forecast (ARIMA, VAR, GARCH)" in desc)

# ============================================================ 3. references
fb = read("econometrics/references/forecasting-backtesting.md")
check("forecasting-backtesting.md: §12-§14 and their entries in the contents",
      all(f"## {k}." in fb for k in (12, 13, 14)) and "12. Labels that look ahead" in fb and "14. A machine-learning" in fb)
check("forecasting-backtesting.md: the recipe names the functions and the licence of the weights",
      all(k in fb for k in ("purged_kfold_splits", "label_overlap", "conformal_intervals", "forecasting.evaluate",
                            "non-commercial")))
check("forecasting-backtesting.md: the guarantee of ACI is stated as long-run with its slack, and the measured coverage",
      "(max(α, 1 − α) + hγ)/(Tγ)" in fb and "84–95%" in fb and "93–95%" in fb)
rc = read("econometrics/references/review-checklist.md")
check("review-checklist.md: overlapping labels, base rate, leaderboard, intervals",
      "Labels that look ahead" in rc and "base rate of the same test window" in rc and "leaderboard" in rc
      and "forecasting.conformal_intervals" in rc)
wf = read("econometrics/references/workflow.md")
check("workflow.md §0: write the card with reporting.study_card, the manifest records its hash",
      "reporting.study_card(path=\"study_card.json\"" in wf and "run_manifest(..., study_card=\"study_card.json\")" in wf)
ir = read("econometrics/references/interpretation-rules.md")
check("interpretation-rules.md: the skeleton passes files= and study_card= to run_manifest",
      "files=[DATA_FILE]" in ir and 'study_card="study_card.json", path="manifest.json"' in ir)
sec = ir.split("## Rules added in v2.7.0", 1)
check("interpretation-rules.md: a v2.7.0 section with level and action, measured coverage, overlap and the card",
      len(sec) == 2 and all(k in sec[1] for k in ("level", "action", "empirical coverage", "horizon=h", "amendment")))

# ============================================================ 4. the two "find the error" cases
CORPUS = {
    "review-lstm-base-rate": {
        "good": [
            "O modelo não supera a taxa-base: 66% dos meses subiram, e 64,8% fica abaixo dela; além disso, os rótulos "
            "se sobrepõem em 20 de 21 pregões.",
            "Compare com a regra ingênua 'sempre sobe' (66%), não com 50%. A validação embaralhada com rótulos "
            "sobrepostos causa vazamento entre treino e teste.",
            "A acurácia está abaixo da taxa base do teste. Use purged K-fold com embargo, porque os rótulos de 21 dias "
            "se sobrepõem.",
            "Base rate de 66% contra 64,8%: não há habilidade. Há look-ahead: o rótulo usa informação do futuro que "
            "cai no treino.",
            "Um modelo naive que sempre prevê alta acerta 66%. A sobreposição dos rótulos diários infla o resultado.",
            "O resultado não supera a classe majoritária e a AUC de 0,52 confirma; o embaralhamento com rótulos "
            "sobrepostos gera vazamento de informação.",
            "Fica abaixo da proporção de meses de alta (66%). Rótulos com sobreposição de 20 pregões: purgue o treino.",
            "Sem purge nem embargo, o K-fold embaralhado mistura dias do mesmo mês; e a acurácia nem chega à taxa-base.",
            "O certo é comparar com a frequência de alta no teste (taxa-base). Também há overlap dos rótulos.",
            "Abaixo da taxa de meses de alta. O desenho tem vazamento temporal.",
            "Classe majoritária: 66%. O modelo erra mais que ela. E as janelas de 21 pregões sobrepostas dão leakage.",
            "Não supera o benchmark ingênuo. Os rótulos se sobrepõem, então a validação deveria ser walk-forward "
            "com purga.",
        ],
        "bad": [
            "64,8% está acima de 50%, então o modelo tem poder preditivo; recomendo usar na alocação.",
            "A AUC de 0,52 é baixa, mas a acurácia é boa. Sugiro mais dados.",
            "O modelo não supera a taxa-base de 66%.",                                   # base rate only
            "Os rótulos se sobrepõem, o que gera vazamento.",                            # leakage only
            "Sugiro testar outras arquiteturas, como GRU ou Transformer.",
            "Faltou um intervalo de confiança para a acurácia.",
            "Sete combinações de variáveis são muitas comparações; ajuste os p-valores.",
            "O período de 2010 a 2024 inclui crises; divida por regime.",
            "Recomendo validar fora da amostra antes de usar.",
            "A LSTM é adequada para séries temporais e o resultado parece promissor.",
            "Acurácia de 64,8% com AUC de 0,52: o conjunto de teste tem 66% de meses de alta.",   # numbers only
            "Use validação cruzada em 10 partes para mais estabilidade.",
        ]},
    "review-automl-leaderboard": {
        "good": [
            "Há viés de seleção: o vencedor entre 25 modelos foi escolhido e ajustado nas mesmas partes.",
            "Falta um conjunto de teste separado, usado uma única vez, para o número final.",
            "A AUC de 0,88 é otimista; use validação cruzada aninhada.",
            "Maldição do vencedor: escolher o melhor de 25 infla a métrica.",
            "Sem hold-out, a diferença de 7 pontos não é estimável sem viés.",
            "Avalie em dados nunca vistos pelo processo de seleção e ajuste.",
            "Comparar 25 modelos é um problema de múltiplas comparações.",
            "Ajustar hiperparâmetros nas mesmas dobras que reportam a AUC causa vazamento.",
            "Reserve uma amostra de teste intocada antes de rodar o AutoML.",
            "A seleção do melhor modelo pela própria métrica de validação torna a estimativa otimista.",
            "Precisa de teste final independente e de um intervalo para a diferença de AUC.",
            "Use nested cross-validation para estimar o desempenho do processo inteiro.",
        ],
        "bad": [
            "O LightGBM é claramente melhor: 0,88 contra 0,81. Pode ir para produção.",
            "Faltou um intervalo de confiança para a AUC.",                              # CI only, no selection
            "Sugiro calibrar as probabilidades antes de produção.",
            "Considere o custo de um falso negativo no cancelamento.",
            "A regressão logística é mais interpretável; talvez valha manter.",
            "Teste também XGBoost e CatBoost.",
            "A AUC não mede calibração.",
            "Recomendo monitorar o modelo em produção.",
            "Use SHAP para explicar as previsões do LightGBM.",
            "Dez partes na validação cruzada é uma boa escolha.",
            "O ganho de 7 pontos de AUC é relevante para o negócio.",
            "Verifique se há desbalanceamento de classes.",
        ]},
}
cases = json.loads(read("evals/cases.json"))["cases"]
ids = [c["id"] for c in cases]
check("evals/cases.json: ids unique; the two new cases present", len(ids) == len(set(ids))
      and {"review-lstm-base-rate", "review-automl-leaderboard"} <= set(ids))
for name, corpus in CORPUS.items():
    folder = SK / "evals" / "routing" / name
    text = (folder / "prompt.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(text.split("---")[1])
    request = text.split("---", 2)[2]
    g = yaml.safe_load((folder / "graders" / "answer.md").read_text(encoding="utf-8").split("---")[1])
    sf = yaml.safe_load((folder / "graders" / "skill-fired.md").read_text(encoding="utf-8").split("---")[1])
    rx = re.compile(g["pattern"], re.I)
    check(f"{name}: valid YAML, reviewer case with the routing tags; skill-fired grader on the Skill tool",
          "reviewer" in meta["tags"] and meta["max_turns"] == 8 and sf["type"] == "tool_used" and sf["tool"] == "Skill"
          and g["target"] == "last_message")
    check(f"{name}: the answer grader does not match the request", not rx.search(request))
    good = [s for s in corpus["good"] if not rx.search(s)]
    bad = [s for s in corpus["bad"] if rx.search(s)]
    check(f"{name}: grader passes all {len(corpus['good'])} correct replies", not good, good)
    check(f"{name}: grader fails all {len(corpus['bad'])} incorrect replies", not bad, bad)
    case = next(c for c in cases if c["id"] == name)
    check(f"{name}: cases.json holds the same prompt, with must and must_not", case["prompt"] == request.strip()
          and len(case["must"]) >= 4 and len(case["must_not"]) >= 2)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
