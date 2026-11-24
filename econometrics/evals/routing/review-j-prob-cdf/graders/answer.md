---
type: regex
pattern: '(\bCDF\b|acumulad[ao]|cumulative|1 ?[−-] ?(0[,.]967|Prob)|complemento|cauda (direita|superior)|right[- ]tail|(p[- ]?valor|valor[- ]p|p-value)\s*(\([^)]{0,20}\)\s*)?(=|≈|é|de|is)?\s*(cerca de |about )?0[,.]03|p ≈ 0[,.]03|3[,.]3 ?%|p[- ]?valor[^.]{0,15}0[,.]03|fun[cç][aã]o de distribui|valor cr[ií]tico[^.]{0,30}9[,.]49|n[aã]o [eé] (o|um) p[- ]?valor|P\(chi2? ?<=? ?J\))'
flags: i
target: last_message
---
