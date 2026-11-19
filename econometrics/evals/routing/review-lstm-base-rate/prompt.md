---
description: "Reviewer: an LSTM's direction accuracy below the base rate, overlapping 21-day labels, shuffled K-fold and seven feature sets (benchmark R2, v2.7.0)"
tags: [routing, reviewer, forecast, ml]
max_turns: 8
timeout_seconds: 300
allowed_tools: [Read, Glob, Grep, Skill]
---

Revise esta conclusão de um relatório interno: "Treinamos uma rede LSTM para prever se o Ibovespa sobe no mês seguinte. Cada pregão de 2010 a 2024 é uma observação, e o rótulo é o sinal do retorno dos 21 pregões seguintes. Usamos validação cruzada em 5 partes com embaralhamento e testamos 7 combinações de variáveis. A melhor acertou 64,8% dos meses no conjunto de teste, com AUC de 0,52. Como 64,8% está bem acima de 50%, o modelo tem poder preditivo e recomendamos usá-lo na alocação." No conjunto de teste, 66% dos meses foram de alta.
