---
type: regex
pattern: '(vari[aá]ve(l|is)[^.]{0,40}omitid|omitid[ao]s?[^.]{0,30}vari[aá]ve|omiss[aã]o de vari[aá]ve|vi[eé]s de vari[aá]vel omitida|deix(a|am|ou|aram|ad[ao]s?)[^.]{0,30}de fora|fic(a|am|ou|aram) de fora|fora do (VAR|modelo|sistema)|causas? comu(m|ns)|fator(es)? comu(m|ns)|terceira vari[aá]vel|confundidor|confounder|dessazonaliz|filtrad|erro de medi|n[aã]o (est[aá]|est[aã]o|entra|entram) no (VAR|modelo|sistema)|n[aã]o inclu[ií]d|ausente|ignora[mr]? (os )?(determinantes|causas|fatores|choques|vari[aá]veis)|(choque|determinante|fator|condutor)(es|s)? comu|\bomit|common (driver|cause|factor|shock)|afeta(m)? (os dois|ambos|as duas)|influencia(m)? (os dois|ambos|as duas)|por tr[aá]s d(os|as) (dois|duas))'
flags: i
target: last_message
---
