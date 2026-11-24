---
type: regex
pattern: '((erros?[- ]padr(ão|ões|ao|oes)|\bEPs?\b|vari[aâ]ncia|\b(the|reported|its) SEs?\b|standard errors?|desvio-padr[aã]o)[^.]{0,80}(errad|incorret|subestim|inv[aá]lid|n[aã]o (s[aã]o |é )?v[aá]lid|viesad|n[aã]o serve|mal calculad|inflad|too small|wrong|understated|ignor)|(errad|incorret|subestim|inv[aá]lid|viesad|mal calculad)[^\s.]*[^.]{0,40}(erros?[- ]padr(ão|ões|ao|oes)|\bEPs?\b|vari[aâ]ncia|\b(the|reported|its) SEs?\b|standard errors?|desvio-padr[aã]o)|corrigir os erros|(wrong|incorrect) standard errors?|infer[eê]ncia (errad|n[aã]o[:;,])|infer[eê]ncia[^.]{0,30}inv[aá]lid|re-?estim[^\s.]* com (o estimador|m[ií]nimos quadrados em dois)|precis[aã]o [eé] ilus[oó]ria|EP correto|o correto [eé] maior|t de 7,5[^.]{0,20}(n[aã]o vale|inflad)|obter erros?[- ]padr(ão|ões|ao|oes) v[aá]lid|\bEP v[aá]lido|(\bEP|erros?[- ]padr|\bthe SE|2SLS|IV2SLS)[^.]{0,40}\b0[,.]15\d?\b|res[ií]duos com (o |os )?x previsto|res[ií]duos (usam|calculados com) (x̂|os valores previstos)|res[ií]duos do segundo est[aá]gio[^.]{0,30}(n[aã]o os|em vez|errad)|n[aã]o os estruturais|σ̂ errado|regressor foi estimado|first-stage estimation)'
flags: i
target: last_message
---
