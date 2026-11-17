---
type: regex
pattern: '^(?=[\s\S]*(\bp[oó]s[- ]?tratamento(?!\s+(d[oa]s?|de)\s)|\bpost[- ]?treatment(?! period)|\b(depois|ap[oó]s) (d?o )?tratamento(?!\s+(d[oa]s?|de)\s)|\bposterior(es)? ao tratamento|\bafter (the )?treatment(?! of)|controle ruim|controles ruins|bad control|\bmediador|\bmediadora|\bmediat(or|ing)))(?=[\s\S]*(\bOster\b|robustness value|valor de robustez|sensibilidade a (confundidor|n[aã]o observ|vari[aá]ve(l|is) omitid)|sensitivity to (unobserv|omitted)|confundidor(es)? n[aã]o (observad|medid)|(fator(es)?|vari[aá]ve(l|is)|caracter[ií]sticas?) n[aã]o observad|n[aã]o observ[aá]ve(l|is)|unobserv|unmeasured confound|vari[aá]ve(l|is) omitid[ao]s?|omitted variable|auto-?s?sele[cç]|vi[eé]s de sele[cç]|self-selection|selection bias))'
flags: i
target: last_message
---
