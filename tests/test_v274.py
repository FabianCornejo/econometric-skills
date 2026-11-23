"""v2.7.4: regressions and documentation checks (rule T4: tests of functions live in the module files).

- the grader of the new routing case review-var-omitted (written in v2.7.4, first run in the routing round of v2.8.0),
  measured on a corpus of right and wrong answers, in Python and in JavaScript (lesson of v2.7.0);
- version, CHANGELOG and README.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)

ROOT = Path(__file__).resolve().parents[1]
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


# ------------------------------------------------------------------ the grader of review-var-omitted
CASE = ROOT / "econometrics" / "evals" / "routing" / "review-var-omitted"
g_txt = (CASE / "graders" / "answer.md").read_text(encoding="utf-8")
m = re.search(r"^pattern: '(.*)'$", g_txt, re.M)
PAT = m.group(1) if m else ""
GOOD = [  # answers that name the omitted driver, or measurement and filtering (the point of the v2.7.4 rules)
    'A conclusão confunde precedência preditiva com causalidade, e uma variável omitida (petróleo, açúcar) pode mover os dois.',
    'Há risco de viés de variável omitida: a inflação ficou de fora do VAR.',
    'O VAR bivariado deixa de fora causas comuns, como o preço do petróleo; com elas o resultado pode sumir.',
    'Uma causa comum (o petróleo) explica o câmbio e o etanol; Granger aparece por omissão de variáveis.',
    'Com variáveis omitidas o VAR de duas séries é um VAR de ordem infinita e as inovações não são choques.',
    'Incluir as variáveis omitidas — inflação, petróleo, açúcar — e ver se o resultado sobrevive.',
    'Fator comum não incluído (commodities) pode gerar a causalidade de Granger observada.',
    'A inflação está fora do modelo; ela afeta os dois e cria precedência espúria.',
    'Uma terceira variável pode mover ambos.',
    'A série dessazonalizada e filtrada pode criar padrões de antecedência espúrios.',
    'Dados com erro de medição criam causalidade de Granger aparente.',
    'Confundidores como o preço internacional do açúcar não estão no VAR.',
    'Faltam variáveis no sistema: o resultado depende do conjunto escolhido; variáveis relevantes foram omitidas.',
    'O teste de Granger num VAR com variáveis omitidas não identifica o efeito.',
    'A inflação foi deixada de fora: isso basta para criar Granger sem canal causal.',
    'Viés de variável omitida: petróleo e açúcar movem os dois preços.',
    'O resultado pode ser artefato das séries filtradas e da omissão da inflação.',
    'Precisa de um VAR com as variáveis que ficaram de fora antes de qualquer leitura causal.',
    'Variáveis relevantes omitidas no VAR (petróleo) tornam a leitura causal inválida.',
    'O câmbio pode apenas antecipar um fator comum, não causar o etanol.',
]
BAD = [  # answers that miss it: generic Granger caveats, other checks, or the prompt echoed back
    'A conclusão confunde precedência preditiva com causalidade.',
    'Não confunda Granger com causa: diga que o câmbio ajuda a prever o etanol.',
    "Granger é precedência preditiva, não causalidade; use 'ajuda a prever'.",
    'Faltou testar estacionariedade das séries antes do VAR.',
    'O p-valor de 0,01 é significativo a 5%, então a conclusão está correta.',
    'Use um VECM se as séries forem cointegradas.',
    'A ordem de defasagens deve ser escolhida por BIC.',
    'Calcule as funções de impulso-resposta com bandas de bootstrap.',
    'A conclusão está certa, mas mostre o intervalo de confiança.',
    'Com p = 0,01 rejeitamos a hipótese nula de não causalidade.',
    'Cuidado com a interpretação causal: isso é associação.',
    'Verifique heterocedasticidade dos resíduos e normalidade.',
    'O câmbio move o preço do etanol, como esperado pela teoria.',
    'Ajuste o modelo em diferenças e repita o teste.',
    'Faça um teste de Toda-Yamamoto para robustez.',
    'Reporte o número de observações e a amostra usada.',
    'A decomposição da variância mostrará a importância do câmbio.',
    'Sem um choque identificado não dá para falar em causalidade estrutural.',
    'O relatório deve mostrar a tabela completa de Granger.',
    'Teste também a direção inversa, do etanol para o câmbio.',
    'Use dados diários para ter mais poder.',
    'A estabilidade do VAR precisa ser conferida pelas raízes.',
    'O VAR mensal só com câmbio e etanol mostra que o câmbio Granger-causa o etanol (p = 0,01).',
]
pat = re.compile(PAT, re.I)
check("review-var-omitted: the grader accepts each of the right answers (Python)",
      PAT and all(pat.search(g) for g in GOOD), [g for g in GOOD if not pat.search(g)])
check("review-var-omitted: the grader refuses each of the wrong answers, the prompt echoed back included (Python)",
      PAT and not any(pat.search(b) for b in BAD), [b for b in BAD if pat.search(b)])
prompt = (CASE / "prompt.md").read_text(encoding="utf-8").split("---", 2)[2]
check("review-var-omitted: the prompt itself does not satisfy the grader", PAT and not pat.search(prompt))
node = shutil.which("node")
if node:
    js = ("const d=JSON.parse(require('fs').readFileSync(0,'utf8'));const p=new RegExp(d.pat,'i');"
          "console.log(JSON.stringify([d.good.filter(g=>!p.test(g)),d.bad.filter(b=>p.test(b))]))")
    out = subprocess.run([node, "-e", js], input=json.dumps({"pat": PAT, "good": GOOD, "bad": BAD + [prompt]}),
                         capture_output=True, text=True, encoding="utf-8")
    res = json.loads(out.stdout) if out.returncode == 0 else None
    check("review-var-omitted: same verdicts in JavaScript (the eval runner's regex engine)", res == [[], []], res)
else:
    print("SKIP JavaScript check: node not found")
check("review-var-omitted: the case has the skill-fired grader of the other review cases",
      (CASE / "graders" / "skill-fired.md").read_text(encoding="utf-8")
      == (CASE.parent / "review-null-result" / "graders" / "skill-fired.md").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ version and CHANGELOG (found by title, not "the top")
import reporting  # noqa: E402
cl = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
heads = re.findall(r"^## v(\d+(?:\.\d+)+) — ", cl, re.M)
check("CHANGELOG has a v2.7.4 entry, right before v2.7.3's", "2.7.4" in heads and "2.7.3" in heads
      and heads.index("2.7.4") + 1 == heads.index("2.7.3"), heads[:4])
e274 = cl.split("## v2.7.4 — ", 1)[1].split("\n## v", 1)[0] if "## v2.7.4 — " in cl else ""
check("the v2.7.4 entry names each changed function and the two rules",
      all(w in e274 for w in ("logrank", "cox", "ph_test", "wild_cluster_bootstrap", "stationarity_many",
                              "classify_controls", "2500", "iv_report", "_has_intercept", "review-var-omitted",
                              "interpretation-rules.md")))
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
vers = {p_.get("version") for p_ in mk.get("plugins", [])}
v_ok = tuple(int(x) for x in reporting.SKILL_VERSION.split(".")) >= (2, 7, 4)   # a version that grows: >=
check("SKILL_VERSION is 2.7.4 or later and equals the marketplace's version", v_ok and vers == {reporting.SKILL_VERSION},
      (reporting.SKILL_VERSION, vers))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
