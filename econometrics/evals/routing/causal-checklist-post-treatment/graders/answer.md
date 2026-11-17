---
type: regex
pattern: '^(?=[\s\S]*(\bp[oó]s[- ]?tratamento(?!\s+(d[oa]s?|de|n[oa]s?)\s+(dados|base|bancos?|amostras?|vari[aá]ve|outliers?|missing|valores|observa[cç]))(?!\s+estat[ií]stic)|\bpost[- ]?treatment(?! period)|\b(depois|ap[oó]s) (d?o )?tratamento(?!\s+(d[oa]s?|de|n[oa]s?)\s+(dados|base|bancos?|amostras?|vari[aá]ve|outliers?|missing|valores|observa[cç]))(?!\s+estat[ií]stic)|\bposterior(es)? ao tratamento|\bafter (the )?treatment(?! (of|periods?)\b)|controle ruim|controles ruins|bad control|\bmediador|\bmediadora|\bmediat(or|ing)))(?=[\s\S]*(\bOster\b|robustness value|valor de robustez|sensibilidade a (confundidor|n[aã]o observ|vari[aá]ve(l|is) omitid)|sensitivity to (unobserv|omitted)|confundidor(es)? n[aã]o (observad|medid)|(fator(es)?|vari[aá]ve(l|is)|caracter[ií]sticas?) n[aã]o observad|n[aã]o observ[aá]ve(l|is)|unobserv|unmeasured confound|vari[aá]ve(l|is) omitid[ao]s?|omitted variable|auto-?s?sele[cç]|vi[eé]s de sele[cç]|self-selection|selection bias))'
flags: i
target: last_message
---
