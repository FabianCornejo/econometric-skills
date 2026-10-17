# econometric-skills

Skills para o Claude voltadas a econometria aplicada.

## `econometria`

Fluxo rigoroso de econometria aplicada em Python (statsmodels, linearmodels, arch): regressão com diagnóstico completo, endogeneidade e variáveis instrumentais, dados em painel, logit/probit/tobit, séries temporais (raiz unitária, quebras, ARIMA, VAR/SVAR, cointegração/VECM), volatilidade (GARCH), previsão com avaliação fora da amostra e estimação de demanda e elasticidades, com dados do BCB, IBGE e Ipeadata.

```
econometria/
├── SKILL.md                         fluxo de trabalho em 7 etapas
├── scripts/diagnosticos.py          baterias de testes com conclusões em português
└── references/
    ├── regressao-mqo.md
    ├── series-temporais.md
    ├── painel-iv.md
    ├── dependente-limitada.md
    ├── demanda-elasticidades.md
    ├── volatilidade.md
    ├── fontes-de-dados.md
    ├── equivalencias-software.md    Stata/EViews → Python
    └── bibliografia.md
```

### Instalação

- **Claude.ai / app:** compacte a pasta `econometria` em `.zip` (ou use o arquivo `.skill`) e envie em Configurações → Capacidades → Skills.
- **Claude Code:** copie a pasta `econometria` para `~/.claude/skills/`.

Dependências Python: `pip install statsmodels linearmodels arch`.

### Teste rápido do script

```bash
python3 econometria/scripts/diagnosticos.py mqo seus_dados.csv --y log_vendas --x log_preco,log_renda
python3 econometria/scripts/diagnosticos.py estacionariedade seus_dados.csv --cols selic,ipca
```

## Feedback e contribuições

A skill está aberta para uso, testes e sugestões:

- **Encontrou um erro ou um resultado estranho?** Abra uma [issue](../../issues) descrevendo a pergunta feita, os dados (ou um exemplo simulado) e a resposta obtida.
- **Sugestão de método, teste ou fonte de dados?** Também via issue.
- **Quer corrigir ou ampliar?** Faça um fork e envie um pull request.

## Licença

[MIT](LICENSE) — uso livre, inclusive comercial, mantendo o aviso de copyright.
