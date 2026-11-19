---
description: "Reviewer: an AutoML leaderboard winner tuned and scored on the same folds, no test window, no interval for the difference (benchmark R12, v2.7.0)"
tags: [routing, reviewer, ml]
max_turns: 8
timeout_seconds: 300
allowed_tools: [Read, Glob, Grep, Skill]
---

Revise este trecho de uma apresentação para a diretoria: "Comparamos 25 modelos com uma ferramenta de AutoML, em validação cruzada de 10 partes, para prever o cancelamento de assinaturas. O vencedor, um LightGBM, teve AUC de 0,87, contra 0,81 da regressão logística. Ajustamos os hiperparâmetros do vencedor nas mesmas 10 partes e chegamos a AUC de 0,88. Concluímos que o LightGBM é 7 pontos melhor que a logística e vamos colocá-lo em produção."
