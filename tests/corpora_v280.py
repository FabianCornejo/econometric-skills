"""v2.8.0: the corpora of right and wrong answers that measure the graders of the two new routing cases (used by
test_v280.py; not a test on its own, so the CI loop over tests/test_*.py does not run it)."""

GMM = {  # gmm-euler: the answer must point to the skill's estimation.gmm (the case measures that the agent finds it)
    "good": [
        "Use `estimation.gmm` com os momentos (β·g^(-γ)·R − 1) multiplicados pelos instrumentos.",
        "A função é estimation.gmm(moments, theta0, args=...), em duas etapas; o J tem 3 graus de liberdade.",
        "```python\nimport estimation as est\nfit = est.gmm(euler, [1.0, 0.99], args=(R, cg, Z))\n```",
        "fit = est.gmm(momentos, [2, 0.99], args=(R, c, Z), names=['gamma', 'beta'])",
        "Chame est.gmm( ... ) e leia fit['j_p'], o p-valor da cauda direita.",
        "Na skill, o GMM genérico está em `estimation.gmm` (references/estimation-theory.md, seção 7).",
        "Com `import estimation as es`, rode es.gmm(euler, theta0, args=(R, cg, Z)).",
        "O caminho é estimation.gmm: você escreve a função de momentos e ela faz o GMM em duas etapas e o J.",
        "Use a função gmm do módulo estimation (scripts/estimation.py).",
        "A função `gmm` em estimation.py estima por GMM e reporta o J com p de cauda direita.",
        "Em estimation.py há a função gmm; passe os momentos n x m.",
        "estimation.gmm(euler_moments, [1.0, 0.99], args=(R, cg, Z), steps='iterated')",
        "I would use estimation.gmm with the Euler moments times the instruments.",
        "Use est.gmm(moments, theta0, args=(R, cg, Z)) from the skill's estimation module.",
        "The skill's generic GMM is estimation.gmm; J = n·ḡ'Wḡ with 5 − 2 = 3 df.",
        "Function: `estimation.gmm` — two-step by default, Newey-West with hac_lags if needed.",
        "Rode `res = est.gmm(mom, [1, .99], args=(R, C, Z), bounds=[None, (0, 1)])`.",
        "O estimador é o estimation.gmm (GMM não linear com momentos do usuário).",
        "Passo 1: estimation.gmm em duas etapas. Passo 2: leia o J (j_p) e o teste de posto (rank_p).",
        "Para o teste de sobreidentificação, o J vem de estimation.gmm (j, j_df, j_p).",
        "Use `estimation.gmm` e, como segunda rota, o NonlinearIVGMM do statsmodels.",
        "O módulo estimation tem gmm(); o J de Hansen sai com o p da cauda direita.",
        "est.gmm(euler, theta0=[1.0, 0.99], args=(ret, cons, Z), center=False)",
        "A skill estima isso com estimation.gmm; não use o IVGMM linear, porque o modelo é não linear.",
        "gmm do estimation.py, com os momentos (β (C_{t+1}/C_t)^{-γ} R_{t+1} − 1) z_t.",
        "Estime com `estimation.gmm(...)` e confira a identificação (rank_p e a precisão de γ).",
        "Use es.gmm(moments, [1., .99], args=(R, cg, Z)).",
        "Função recomendada: estimation.gmm (steps='two-step').",
        "A estimação: estimation.gmm; o teste: o J de Hansen que ela reporta.",
        "Com o script da skill: `fit = est.gmm(euler, [1.0, 0.99], args=(R, cg, Z))` e `fit['j_p']`.",
        "Para a equação de Euler, a skill tem `estimation.gmm`: escreva g_i(θ) = (β c^-γ R − 1)·z.",  # held-out corpus (written after the grader)
        "Recomendo: `import estimation as est; res = est.gmm(...)`.",  # held-out corpus (written after the grader)
        "Use `estimation.gmm(moments, [1.0, 0.99], args=(R, cg, Z), hac_lags=None)`; J com 3 gl.",  # held-out corpus (written after the grader)
        "A função certa é a gmm do módulo `estimation`.",  # held-out corpus (written after the grader)
        "Rode o GMM genérico (estimation.gmm) e veja se o teste de posto rejeita.",  # held-out corpus (written after the grader)
        "No estimation.py há gmm(), que faz duas etapas e o J.",  # held-out corpus (written after the grader)
        "es.gmm(f, x0, args=a) dá os coeficientes e o J.",  # held-out corpus (written after the grader)
        "Estimate with estimation.gmm and read j_p, the right-tail p-value.",  # held-out corpus (written after the grader)
        "The skill's estimation module has a gmm function for user moments.",  # held-out corpus (written after the grader)
        "Use the gmm function in estimation.py.",  # held-out corpus (written after the grader)
        'Eu usaria o GMM da skill — `gmm` em `scripts/estimation.py` — e leria o J pela cauda direita.',  # review r1 (adversarial right answer)
        "Use estimation's gmm: est = estimation.gmm(...)",  # review r1 (adversarial right answer)
        "The GMM estimator lives in the skill's estimation module (`estimation.gmm`).",  # review r1 (adversarial right answer)
        'Utilize a rotina gmm (módulo estimation) com os momentos de Euler.',  # review r1 (adversarial right answer)
        'Chame `gmm()` do módulo `estimation`.',  # review r1 (adversarial right answer)
        'A skill oferece GMM genérico: veja a função gmm, no arquivo estimation.py.',  # review r1 (adversarial right answer)
        'In Python: import estimation; estimation.gmm(moments, x0)',  # review r1 (adversarial right answer)
        'Para isso a skill tem estimation.gmm; escreva os momentos como (β g^-γ R − 1)·z.',  # held-out r1 (written after the r1 grader)
        'Use `gmm` do módulo `estimation` com hac_lags se houver autocorrelação.',  # held-out r1 (written after the r1 grader)
        'res = est.gmm(euler_moms, [2.0, 0.99], args=(R, g, Z))',  # held-out r1 (written after the r1 grader)
        'O módulo estimation tem a função gmm, que reporta J, j_df e j_p.',  # held-out r1 (written after the r1 grader)
        'A rotina gmm do estimation.py faz o GMM em duas etapas.',  # held-out r1 (written after the r1 grader)
        "I'd call estimation.gmm(...) and look at j_p.",  # held-out r1 (written after the r1 grader)
        'The function gmm in the estimation module handles nonlinear moments.',  # held-out r1 (written after the r1 grader)
        'Na skill: `import estimation as es` e `es.gmm(momentos, x0, args=...)`.',  # held-out r1 (written after the r1 grader)
        "Use a função `gmm()` em `estimation.py` com steps='iterated'.",  # held-out r1 (written after the r1 grader)
        'estimation.gmm é a função; o J sai com p de cauda direita.',  # held-out r1 (written after the r1 grader)
        # review r2 (adversarial set of the reviewer, g1.py)
        'Use a função `gmm` da skill (scripts/estimation.py): `res = gmm(momentos, [2, .99], args=(R, g, Z))`.',
        'from estimation import gmm\nres = gmm(euler, [2.0, 0.99], args=(R, g, Z))',
        'Rode estimation.gmm( momentos , ... ) e leia j_p.',
        'O GMM da skill (estimation · gmm) resolve: duas etapas, J na cauda direita.',
        'Estimation.GMM with the Euler moments, then read j_p.',
        'Ferramenta: `econometrics/scripts/estimation.py::gmm`.',
        "I would run the skill's gmm (in estimation.py).",
        'Use gmm() — it is in the estimation module of this skill.',
        'Chame est.gmm (com import estimation as est).',
        'A implementação é `estimation.gmm`, não o IVGMM.',
        # review r2: held-out set written after the fix (one right answer missed, 'existe')
        'Resolvo com a gmm da própria skill: `estimation.gmm(mom, [2, .99], args=(R, cg, Z))`.',
        'import estimation\nres = estimation.gmm(momentos, theta0, args=dados)',
        "Use `gmm` from `estimation.py` (the skill's scripts) with two steps.",
        'O GMM não linear sai de est.gmm(...); depois confira j_p e rank_p.',
        "The right tool is the gmm function in the skill's estimation module.",
        'Vou usar a rotina gmm do módulo estimation, com hac_lags=4.',
        'from econometrics.scripts.estimation import gmm',
        'Recomendo es.gmm(euler, [1.5, 0.98], args=(R, C, Z), center=False).',
        'Estimate with estimation.gmm and report J with its right-tail p.',
        'No estimation.py existe a função gmm, que faz exatamente isso.',
    ],
    "bad": [
        "Use o NonlinearIVGMM do statsmodels: NonlinearIVGMM(endog, exog, instr, func).fit().",
        "Use linearmodels.iv.IVGMM.from_formula(...).fit().",
        "Escreva o critério ḡ'Wḡ e minimize com scipy.optimize.minimize.",
        "from statsmodels.sandbox.regression import gmm; gmm.GMM(...)",
        "Use o comando gmm do Stata: gmm (b*x^(-g)*R - 1), instruments(L.R L.x) twostep.",
        "No R, gmm::gmm(g, x, t0 = c(1, 0.99)).",
        "Estime por máxima verossimilhança com estimation.mle, supondo lognormalidade.",
        "Faça um 2SLS do log do retorno no log do consumo com systems.three_sls.",
        "O GMM de Arellano-Bond (systems.dynamic_panel) resolve isso.",
        "Use mínimos quadrados não lineares (estimation.nls) na equação de Euler.",
        "Não há função pronta para GMM; escreva o seu próprio código.",
        "O teste de sobreidentificação é o J de Hansen; Prob alto significa que o modelo não é rejeitado.",
        "Use IVGMM da linearmodels com a forma log-linear da equação de Euler.",
        "Você pode usar o pacote pydynpd.",
        "Use diagnostics.overid_test após um 2SLS.",
        "Use NonlinearIVGMM(...).fit(maxiter=2, weights_method='hac').",
        "Calcule o GMM com a fórmula fechada (Z'X)^-1 Z'y.",
        "The function is statsmodels' GMM class.",
        "Use the gmm package in R.",
        "Minimize the GMM objective by hand with scipy.",
        "Use linearmodels IVGMM and read j_stat.",
        "There is no GMM in the skill; use Stata.",
        "Estimate by OLS of returns on consumption growth.",
        "Use estimation.trinity_test for the overidentifying restrictions.",
        "Fit with systems.sur_system and test with the Breusch-Pagan LM.",
        "Use a VAR (multivariate.var_analysis) and test Granger causality.",
        "Use scipy.optimize.least_squares on the Euler errors.",
        "A melhor opção é o MLE do Hansen-Singleton 1983 com um VAR.",
        "Rode o GMM no EViews.",
        "Use o IVGMM com instrumentos defasados; o J aparece como j_stat.",
        # adversarial (written to share words with right answers, v2.7.5 lesson)
        "gmm.GMM do statsmodels, com estimation de pesos HAC.",
        "O GMM em estimation de equações de Euler é padrão; use o Stata.",
        "GMM in estimation of Euler equations is standard; use Stata's gmm.",
        "Use NonlinearIVGMM; o estimation.py só tem mle e nls.",
        "O estimation.mle faz o MLE; para GMM, use o statsmodels.",
        "A função de estimação gmm do R resolve.",
        "Usei a estimation strategy de GMM do artigo, com IVGMM.",
        "No Stata: gmm (eq), estimation options twostep.",
        "Use o IVGMM do linearmodels com a equação linearizada.",  # held-out corpus (written after the grader)
        "Linearize em log e rode um 2SLS com diagnostics.iv_report.",  # held-out corpus (written after the grader)
        "O pacote gmm do R é o mais indicado.",  # held-out corpus (written after the grader)
        "statsmodels.sandbox.regression.gmm.NonlinearIVGMM é a função.",  # held-out corpus (written after the grader)
        "Use mle com uma distribuição lognormal conjunta (Hansen-Singleton 1983).",  # held-out corpus (written after the grader)
        "Minimize o critério de GMM com scipy e calcule o J à mão.",  # held-out corpus (written after the grader)
        "No EViews, use a estimação por GMM com instrumentos.",  # held-out corpus (written after the grader)
        "A GMM estimation in Stata is the way: gmm with twostep.",  # held-out corpus (written after the grader)
        "Estimation by GMM requires instruments; use IVGMM.",  # held-out corpus (written after the grader)
        "Use systems.dynamic_panel com method='system'.",  # held-out corpus (written after the grader)
        "Use the gmm function in statsmodels' estimation module (statsmodels.sandbox.regression.gmm).",  # review r1 (adversarial wrong answer)
        "The gmm function of the estimation module in R's gmm package does this.",  # review r1 (adversarial wrong answer)
        'Use statsmodels.sandbox.regression.gmm.GMM com pesos HAC.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'A estimation by GMM is best done in Stata with gmm.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        "Use the gmm function from R's gmm package (estimation module there).",  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'O IVGMM do linearmodels é a função adequada.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'GMM estimation of the Euler equation: use NonlinearIVGMM.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'Minimize o critério de GMM; a skill só tem mle e nls no estimation.py.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'Use estimation.nls na forma log-linear.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'O gmm do Stata (estimation command) resolve isso.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        "Use EViews' GMM estimation.",  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        'Escreva o seu próprio GMM com numpy.',  # held-out r1 (written after the r1 grader); one passed and the grader was fixed
        # review r2 (adversarial set of the reviewer, g1.py)
        "Use statsmodels' GMM (statsmodels.sandbox.regression.gmm.GMM); the skill's estimation.py has only mle and nls.",
        'The estimation module of linearmodels has IVGMM; use it.',
        'Use a função gmm do pacote R (gmm::gmm) — module estimation of R.',
        'Use NonlinearIVGMM; avoid est.gmm because it is linear only.',
        'A função gmm do módulo estimation do statsmodels serve.',
        'Use o pacote gmm do R; a função gmm (do módulo estimation de lá) aceita momentos.',
        'Escreva sua própria função gmm em um arquivo estimation.py novo.',
        'In linearmodels, the gmm function of the estimation module (IVGMM) is right.',
        # review r2: held-out set written after the fix (one right answer missed, 'existe')
        'Use o GMM do statsmodels (NonlinearIVGMM) e reporte o J.',
        'linearmodels.IVGMM resolve; é a função gmm do pacote de estimation deles.',
        'Em R, gmm::gmm(g, x, t0) com o módulo estimation do pacote.',
        'Estime por máxima verossimilhança com estimation.mle e compare.',
        'Crie um novo arquivo estimation.py com uma função gmm sua.',
        'Use scipy.optimize.minimize no critério GMM escrito à mão.',
        "The gmm function of Stata's estimation commands handles this.",
        'Use o IVGMM do linearmodels; o modelo é linear em logs.',
        'Rode nls do estimation.py; o GMM não é necessário.',
        'Python: from statsmodels.sandbox.regression.gmm import GMM.',
    ],
}

PEEK = {  # review-optional-stopping: daily looks and stopping at p < 0.05 inflate the type I error well above 5%
    "good": [
        "Olhar o p-valor todos os dias e parar no primeiro p < 0,05 infla o erro tipo I bem acima de 5%.",
        "Isso é peeking: com olhadas diárias o erro tipo I real passa de 5% (cerca de 19% com 10 olhadas).",
        "Parada opcional (optional stopping): o p = 0,041 não é um p-valor válido.",
        "O teste foi encerrado quando ficou significativo; isso aumenta a chance de falso positivo.",
        "Ao acompanhar o teste diariamente e parar quando p < 0,05, a taxa de falsos positivos sobe muito acima do alfa nominal.",
        "Com 9 olhadas, o erro tipo I efetivo é por volta de 18%, não 5%.",
        "Esse procedimento de monitorar o p-valor todo dia e parar no primeiro resultado significativo invalida o p-valor.",
        "Deveriam ter usado um teste sequencial (SPRT) ou limites de Pocock/O'Brien-Fleming definidos antes.",
        "A regra de parada dependeu do resultado: isso infla o alfa.",
        "Múltiplas olhadas sem correção elevam o erro tipo I; a 'confiança de 95%' não vale.",
        "The test was stopped as soon as p < 0.05 after daily looks: optional stopping inflates the false positive rate.",
        "Peeking at the p-value every day inflates the type I error far above 5%.",
        "With repeated interim looks the actual alpha is not 5%; use an alpha-spending boundary.",
        "O erro tipo I não é 5%: com olhadas repetidas e parada no primeiro sucesso ele fica em torno de 19% (10 olhadas).",
        "A decisão de encerrar quando p cruzou 0,05 transforma o teste num procedimento com alfa inflado.",
        "Esse é o problema clássico de espiar os resultados (peeking).",
        "Encerrar o experimento ao ver p < 0,05 não preserva o nível de 5%: a probabilidade de falso positivo aumenta.",
        "Precisaria planejar o número de olhadas e usar um nível por olhada (Pocock: 0,0158 para 5 olhadas).",
        "Recomendo um teste sequencial de Wald ou alpha spending em vez de monitorar o p-valor diariamente.",
        "O monitoramento contínuo do p-valor com parada no primeiro valor abaixo de 0,05 aumenta o erro tipo I.",
        "Como o teste foi interrompido quando atingiu significância, o p = 0,041 está viesado para baixo e o alfa real é maior.",
        "Falso positivo provável: olhar 9 vezes e parar no primeiro p < 0,05 eleva o alfa para algo como 17-18%.",
        "O nível de significância real não é 5% por causa da parada opcional.",
        "This is optional stopping; the reported 95% confidence is wrong.",
        "Daily monitoring with stopping at significance makes the type I error much larger than 0.05.",
        "A interrupção no 9º dia por causa do p-valor infla a taxa de erro tipo I.",
        "Olhar o resultado várias vezes e parar quando der significativo é p-hacking por parada opcional; o alfa efetivo é maior que 5%.",
        "O correto seria usar um SPRT ou fixar o tamanho da amostra antes, sem olhadas intermediárias.",
        "Cada olhada é um novo teste: com 9 testes o erro tipo I acumulado passa bem de 5%.",
        "Parar o teste assim que p < 0,05 sem regra sequencial pré-definida infla o falso positivo.",
        "Parar no primeiro dia com p < 0,05 após checagens diárias aumenta a probabilidade de falso positivo.",  # held-out corpus (written after the grader)
        "Isso é um caso de optional stopping; o erro tipo I real é bem maior que 5%.",  # held-out corpus (written after the grader)
        "Olhando todos os dias, o alfa efetivo fica acima de 5% (cerca de 18% com 9 olhadas).",  # held-out corpus (written after the grader)
        "Deveriam ter definido o tamanho da amostra antes ou usado um SPRT.",  # held-out corpus (written after the grader)
        "O teste foi 'espiado' diariamente, o que infla o alfa.",  # held-out corpus (written after the grader)
        "A interrupção quando o p caiu abaixo de 0,05 eleva a taxa de erro tipo I.",  # held-out corpus (written after the grader)
        "Monitorar o p-valor e parar ao atingir significância torna o p-valor inválido.",  # held-out corpus (written after the grader)
        "With nine looks, the false positive rate is about 17%, not 5%.",  # held-out corpus (written after the grader)
        "They should have used alpha spending or Pocock boundaries.",  # held-out corpus (written after the grader)
        "Stopping at the first significant look inflates alpha.",  # held-out corpus (written after the grader)
        'Problema: múltiplas comparações ao longo do tempo — cada olhada diária é um teste, e o nível global deixa de ser 5%.',  # review r1 (adversarial right answer)
        'O p = 0,041 foi obtido com regra de parada dependente dos dados; o nível de significância real fica perto de 18%.',  # review r1 (adversarial right answer)
        'Testing every day until significance is a form of p-hacking; the nominal 5% level does not hold.',  # review r1 (adversarial right answer)
        "Isso é 'optional stopping'.",  # review r1 (adversarial right answer)
        "O desenho correto seria fixar n antes ou usar um desenho sequencial com fronteira de O'Brien-Fleming.",  # review r1 (adversarial right answer)
        'Isso é peeking: parar quando p < 0,05 após olhadas diárias infla o erro tipo I.',  # held-out r1 (written after the r1 grader)
        'O alfa efetivo, com 9 olhadas, fica em torno de 18%.',  # held-out r1 (written after the r1 grader)
        'Parar no primeiro resultado significativo eleva a taxa de falsos positivos.',  # held-out r1 (written after the r1 grader)
        'Repeated significance testing makes the false positive rate much larger than 5%.',  # held-out r1 (written after the r1 grader)
        'O nível de significância nominal de 5% não se mantém com parada opcional.',  # held-out r1 (written after the r1 grader)
        "Deveriam ter usado um SPRT ou fronteiras de O'Brien-Fleming.",  # held-out r1 (written after the r1 grader)
        'Monitorar diariamente e parar quando significativo torna o p-valor inválido.',  # held-out r1 (written after the r1 grader)
        'The type I error is about 18% with nine looks, not 5%.',  # held-out r1 (written after the r1 grader)
        'Com olhadas diárias, a probabilidade de um falso positivo aumenta muito.',  # held-out r1 (written after the r1 grader)
        'Optional stopping: the 95% confidence claim is wrong.',  # held-out r1 (written after the r1 grader)
        # review r2 (adversarial set of the reviewer, g1.py)
        'Olhar o resultado todo dia e parar no primeiro p < 0,05 eleva o alfa para algo como 19%.',
        'Daily peeking: the overall false-positive probability is roughly 0.19, not 0.05.',
        'Com 9 olhadas o erro do tipo I acumulado é cerca de 0,18, não 0,05.',
        "O problema é o 'parar quando deu significativo': o nível real não é 5%.",
        'This is the classic sequential-testing problem; with interim looks the p-value is not valid as reported.',
        'A interrupção precoce ao ver significância invalida o p-valor.',
        'O erro tipo I fica acima do nominal.',
        'Repeated looks inflate the Type-I error rate.',
        'Peeking diário: o α efetivo é bem maior que 5%.',
        'A taxa de falso positivo passa de 5% para uns 18%.',
        # review r2: held-out set written after the fix (one right answer missed, 'existe')
        'Ao parar no primeiro dia com p < 0,05 o erro tipo I real fica perto de 0,19.',
        'The team peeked daily, which inflates the false positive rate well above 5%.',
        'Isso é p-hacking por parada opcional: o nível de 5% não vale.',
        'Monitorar o p-valor e parar quando cai abaixo de 0,05 invalida o p-valor reportado.',
        'With repeated significance testing the type I error is about 19%, not 5%.',
        'Deveriam ter usado um teste sequencial (SPRT) ou fronteiras de Pocock.',
        'O alfa efetivo é muito maior que o nominal de 5%.',
        'Olhar todo dia aumenta a chance de um falso positivo para algo como 18%.',
        'Early stopping on significance means the reported p is not valid.',
        'A taxa de falsos positivos sobe para cerca de 0,2 com olhadas diárias.',
    ],
    "bad": [
        "O p = 0,041 é menor que 0,05, então a conclusão está correta.",
        "O teste de proporções está correto; o problema é que 9 dias é pouco tempo para captar sazonalidade.",
        "Falta reportar o tamanho do efeito e o intervalo de confiança.",
        "Pode haver efeito novidade: usuários clicam no botão novo só por ser novo.",
        "O teste deveria ser bicaudal.",
        "A amostra pode não ser representativa de todos os usuários.",
        "Significância estatística não é significância prática; reporte o aumento em pontos percentuais.",
        "Há múltiplas métricas; ajuste para comparações múltiplas entre métricas.",
        "O acompanhamento diário não afeta o erro tipo I, porque cada dia é um teste independente.",
        "Não há problema em olhar o p-valor todo dia; o que importa é o valor final.",
        "A conclusão está correta: 95% de confiança porque p < 0,05.",
        "O problema é a falta de randomização.",
        "O teste deveria ter usado um teste t em vez de proporções.",
        "Faltou checar o poder do teste.",
        "Com p = 0,041 a evidência é fraca; prefira p < 0,01.",
        "O resultado é causal porque houve randomização, então a conclusão vale.",
        "The p-value is below 0.05, so the conclusion holds.",
        "The test should have run for a full week to capture weekly cycles.",
        "Report the confidence interval of the difference in conversion rates.",
        "There may be a novelty effect.",
        "Use a Bayesian A/B test instead.",
        "Check for sample ratio mismatch between the arms.",
        "A dúvida é se a conversão foi medida corretamente.",
        "O alfa de 5% foi respeitado, porque o p final é 0,041.",
        "Não houve problema de erro tipo I: o teste parou no 9º dia como planejado.",
        "Seria bom repetir o teste em outro mês.",
        "O problema é o viés de seleção dos usuários que viram o botão.",
        "O intervalo de 95% não inclui zero, então o efeito é real.",
        "Faltou segmentar por dispositivo.",
        "Use um teste qui-quadrado em vez do teste de proporções.",
        # adversarial (written to share words with right answers, v2.7.5 lesson)
        "O botão aumenta a conversão e o alfa de 5% foi respeitado.",
        "O alfa de 5% acima do p = 0,041 confirma o efeito.",
        "O aumento de 18% na conversão é relevante.",
        "Um teste sequencial não seria diferente aqui; a conclusão vale.",
        "O erro tipo I é 5%, como em qualquer teste com p < 0,05.",
        "A taxa de conversão sobe 2 pontos, e o erro tipo I é controlado.",
        "With p = 0.041 the false positive rate is 4.1%.",
        "The type I error is 5%, which is larger than the p-value.",
        "A conclusão está certa: p = 0,041 < 0,05.",  # held-out corpus (written after the grader)
        "O teste deveria ter rodado por duas semanas por causa da sazonalidade.",  # held-out corpus (written after the grader)
        "Falta o intervalo de confiança da diferença de proporções.",  # held-out corpus (written after the grader)
        "O alfa é 5% e o p é 0,041; logo rejeitamos H0.",  # held-out corpus (written after the grader)
        "Houve aumento de 20% na conversão, o que é relevante.",  # held-out corpus (written after the grader)
        "A taxa de falsos positivos é controlada pelo teste de proporções.",  # held-out corpus (written after the grader)
        "O efeito é pequeno; a significância prática é questionável.",  # held-out corpus (written after the grader)
        "The conversion increased 15%, a large effect.",  # held-out corpus (written after the grader)
        "The type I error rate is 5% by design.",  # held-out corpus (written after the grader)
        "Check whether the randomization was balanced.",  # held-out corpus (written after the grader)
        'O alfa = 0,05 e p = 0,041, então o efeito não é nulo e a conclusão vale.',  # review r1 (adversarial wrong answer)
        'The type I error is 5%, which is not a problem here; the conclusion holds.',  # review r1 (adversarial wrong answer)
        'O acompanhamento diário é boa prática e o efeito não é pequeno: o botão funciona.',  # review r1 (adversarial wrong answer)
        'O novo botão aumenta a taxa de conversão sem falsos positivos, pois p < 0,05.',  # review r1 (adversarial wrong answer)
        'Alfa respeitado e conversão subiu 18%: resultado sólido.',  # review r1 (adversarial wrong answer)
        'O alfa real é maior que o p, então rejeitamos H0 com segurança.',  # review r1 (adversarial wrong answer)
        'Monitoramos diariamente; o resultado não é pequeno nem espúrio.',  # review r1 (adversarial wrong answer)
        'False positives are unlikely: the lift of 20% is large.',  # review r1 (adversarial wrong answer)
        'O p-valor de 0,041 está abaixo de 0,05; conclusão válida.',  # held-out r1 (written after the r1 grader)
        'O aumento de conversão foi de 19%, efeito grande.',  # held-out r1 (written after the r1 grader)
        'O alfa é 0,05 e não há outro problema.',  # held-out r1 (written after the r1 grader)
        'The type I error is controlled at 5% by the test.',  # held-out r1 (written after the r1 grader)
        'Faltou calcular o poder; o alfa está ok.',  # held-out r1 (written after the r1 grader)
        'A taxa de conversão aumentou e o erro padrão é pequeno.',  # held-out r1 (written after the r1 grader)
        'O erro tipo I não é relevante aqui porque o efeito é grande.',  # held-out r1 (written after the r1 grader)
        'Os falsos positivos ocorrem em 5% dos testes, como sempre.',  # held-out r1 (written after the r1 grader)
        'False positives are rare when p < 0.05.',  # held-out r1 (written after the r1 grader)
        'O teste deveria considerar a sazonalidade semanal.',  # held-out r1 (written after the r1 grader)
        # review r2 (adversarial set of the reviewer, g1.py)
        'The type I error is 5% and the result is significant; ship it.',
        'A conversão aumentou 18% e o p é 0,041: significativo.',
        'The p-value of 0.041 is fine; false positives are at 5%.',
        'O efeito é de 15% (IC de 2% a 28%); conclusão válida.',
        'O teste é válido: o erro tipo I é 5%, não 18%.',
        # review r2: held-out set written after the fix (one right answer missed, 'existe')
        'O teste é significativo (p = 0,041); a conversão subiu.',
        'The effect is 1.2 percentage points with p = 0.041, a real lift.',
        'Basta corrigir por Bonferroni entre as métricas; o resto está certo.',
        'O problema é o tamanho da amostra pequeno, não a forma de olhar.',
        'The issue is multiple metrics, not the stopping rule.',
        'Reporte o intervalo de confiança de 95% e lance a variante.',
        'A diferença de 18% na conversão é estatisticamente significativa.',
        'O erro tipo II é alto; aumentem a amostra.',
        'Use um teste t de Welch em vez do z; o p muda pouco.',
        'Ship it: the p-value is below 0.05 and power was 80%.',
    ],
}

KNOWN_FALSE_PASSES = {  # negations and dismissals: the executor has no negative grader (v2.7.5)
    "GMM": ["Não use estimation.gmm; prefira o IVGMM.", "A função gmm não existe no módulo estimation.",
            "O módulo estimation não tem gmm; use o NonlinearIVGMM do statsmodels.",
            "Use IVGMM; a função gmm em estimation.py não serve para modelos não lineares.",
             'In Stata: gmm (estimation.gmm is not available here).',
             'Use o momentfit do R; não há estimation.gmm aqui.'],
    "PEEK": ["O erro tipo I não aumenta por acompanhar o teste.", "Não houve peeking, pois o teste parou no dia planejado.",
             "Acompanhar diariamente não invalida o p-valor.", "Usar Pocock aqui seria exagero; a conclusão é válida.",
             "A taxa de falsos positivos não é afetada pelo acompanhamento diário.",
             "Acompanhar o p-valor diariamente não é problema; a conclusão está correta.",
             "Não é preciso usar SPRT ou Pocock aqui; a conclusão vale.",
             'There is no peeking problem: the test stopped at the planned date.',
             'O alfa não é inflado; o teste é válido.',
             'Não houve parada opcional; a conclusão está correta.',
             'Use um SPRT da próxima vez é opcional; aqui está tudo certo.',
             'Peeking would be a problem only if they had looked; they did not.'],
}
