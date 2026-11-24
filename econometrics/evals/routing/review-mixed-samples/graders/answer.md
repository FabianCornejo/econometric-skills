---
type: regex
pattern: '(amostras? (diferentes?|distintas?)|(vem|vêm|vieram|tirad[ao]|calculad[ao]|é) (de|em|d[ae]) outra amostra|(mesma|mesmas) (amostra|observa[cç][oõ]es)|mistur[^\s.]*[^.]{0,20}amostras?|amostras?[^.]{0,15}mistur|inconsist[eê]ncia de amostra|\b(um|de um|é um|são um) (subconjunto|subgrupo)|m[eé]dia de (uma )?subamostra|amostra (da m[eé]dia|de estima[cç][aã]o|da regress[aã]o)[^.]{0,30}(difer|n[aã]o)|n[aã]o (corresponde|[eé]) (a|à) (amostra|m[eé]dia) da (regress|amostra)|different sample|same sample|sample mismatch|m[eé]dia (dos|nos|sobre os) (mesmos )?111|mesm[ao]s 111|\b7[,.]2\d?\b|\b64\b[^.]{0,80}\b111\b|\b111\b[^.]{0,80}\b64\b|base inconsistente|incoer[eê]ncia)'
flags: i
target: last_message
---
