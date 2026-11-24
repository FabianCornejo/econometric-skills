---
type: regex
pattern: '((erros?|EP)[- ]?(padr(ão|ões))?[^.]{0,80}(errad|incorret|subestim|inv[aá]lid|n[aã]o (s[aã]o |é )?v[aá]lid|corrigir)|(errad|incorret|subestim|inv[aá]lid)[^\s.]*[^.]{0,40}(erros?[- ]padr|\bEP\b)|corrigir os erros|standard errors?[^.]{0,40}(wrong|incorrect|understated)|(wrong|incorrect) standard errors?|infer[eê]ncia (errad|n[aã]o)|re-?estim[^\s.]* com (o estimador|m[ií]nimos quadrados em dois)|precis[aã]o [eé] ilus[oó]ria|EP correto|o correto [eé] maior|t de 7,5 n[aã]o vale|obter erros?[- ]padr(ão|ões|ao|oes) v[aá]lid)'
flags: i
target: last_message
---
