"""Generates evals/data/treino_regiao_v269.csv, the data of the blind eval `iv-interaction-numbers` (seed 26900).

Design: training hours (0-200) are endogenous (unobserved ability raises both hours and wages); instruments: distance
to the training centre (km), a municipal subsidy (0/1) and their interaction (the subsidy weakens the effect of
distance in the first stage). True effect: +0.15% of the wage per training hour (0.0015 in logs). Plausible scales:
monthly wage in R$, experience in years, distance in km. Used first in the v2.6.9 validation.

Usage: python make_treino_regiao_v269.py [output.csv]   (prints the SHA-256 and checks it against the committed file's)
"""
import numpy as np, pandas as pd, hashlib, sys
from pathlib import Path
rng = np.random.default_rng(26900)
n = 600
dist = rng.uniform(1, 60, n).round(1)
sub = rng.binomial(1, 0.4, n)
exper = rng.integers(0, 31, n)
hab = rng.normal(size=n)
horas = 70 - 0.55 * dist + 12 * sub + 0.45 * dist * sub + 12 * hab + rng.normal(0, 22, n)
horas = np.clip(horas, 0, 200).round(0)
lsal = 7.6 + 0.0015 * horas + 0.03 * exper - 0.0005 * exper**2 + 0.12 * hab + rng.normal(0, 0.25, n)
d = pd.DataFrame(dict(salario=np.exp(lsal).round(2), horas_treino=horas.astype(int), experiencia=exper,
                      distancia_km=dist, subsidio=sub))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("treino_regiao_v269.csv")
d.to_csv(out, index=False, lineterminator="\n")   # LF on every system (v2.6.11: Windows wrote CRLF)
EXPECTED = "7a2d977571858bf8fb194316bf6e5ae2bb81135f1b96075eea315644e56a7a45"
sha = hashlib.sha256(out.read_bytes()).hexdigest()
print(out, sha, "(matches the committed file)" if sha == EXPECTED else "(DIFFERS from the committed file: numpy or pandas changed the draws or the CSV format)")
print(d.describe().round(2).to_string())
