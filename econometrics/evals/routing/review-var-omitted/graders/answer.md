---
type: regex
pattern: '(vari[aá]ve(l|is)[^.]{0,40}omitid|omitid[ao]s?[^.]{0,30}vari[aá]ve|omiss[aã]o de vari[aá]ve|vi[eé]s de vari[aá]vel omitida|deix(a|am|ou|aram|ad[ao]s?)[^.]{0,30}de fora|fic(a|am|ou|aram) de fora|fora do (VAR|modelo|sistema)|causas? comu(m|ns)|fator(es)? comu(m|ns)|terceira vari[aá]vel|confundidor|confounder|dessazonaliz|filtrad|erro de medi|n[aã]o (est[aá]|est[aã]o|entra|entram) no (VAR|modelo|sistema)|ignora[mr]? (os )?(determinantes|causas|fatores|vari[aá]veis)|(choque|determinante|fator|condutor)(es|s)? comu|common (driver|cause|factor|shock)|afeta(m)? (os dois|ambos|as duas)|influencia(m)? (os dois|ambos|as duas)|por tr[aá]s d(os|as) (dois|duas)|(vari[aá]ve(l|is)|fator(es)?|determinantes?|causas?|petr[oó]leo|c[aâ]mbio)[^.]{0,40}(ausente|n[aã]o inclu[ií]d)|\bomits? (the |a |an )?(oil|common|relevant|key|third|variable|driver|factor|price))'
flags: i
target: last_message
---
