---
type: regex
pattern: '(amostras? (diferentes?|distintas?)|outras? amostras?|(mesma|mesmas) (amostra|observa[cç][oõ]es)|mistur[^\s.]*[^.]{0,20}amostras?|amostras?[^.]{0,15}mistur|inconsist[eê]ncia de amostra|subconjunto|subamostra|amostra (da m[eé]dia|de estima[cç][aã]o|da regress[aã]o)[^.]{0,30}(difer|n[aã]o)|n[aã]o (corresponde|[eé]) (a|à) (amostra|m[eé]dia) da (regress|amostra)|different sample|same sample|sample mismatch|m[eé]dia (dos|nos|sobre os) (mesmos )?111|mesm[ao]s 111)'
flags: i
target: last_message
---
