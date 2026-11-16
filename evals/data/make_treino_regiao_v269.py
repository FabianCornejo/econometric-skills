"""Gera evals/data/treino_regiao_v269.csv para a validação da v2.6.9 (semente 26900).

Desenho: horas de treinamento (0–200) endógenas (habilidade não observada eleva horas e salário); instrumentos:
distância ao centro de treinamento (km) e subsídio municipal (0/1), com efeito da distância menor onde há
subsídio (interação no 1º estágio). Efeito verdadeiro: +0,15% no salário por hora de treinamento (0,0015 em log).
Escalas plausíveis: salário mensal em R$, experiência em anos, distância em km."""
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
d.to_csv(out, index=False)
print(out, hashlib.sha256(out.read_bytes()).hexdigest())
print(d.describe().round(2).to_string())
