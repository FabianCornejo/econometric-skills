---
type: regex
pattern: '(\bCDF\b|(Prob|0[,.]967)[^.;]{0,40}(acumulad|cumulative)|(acumulad|cumulative)[^.;]{0,25}(qui|χ²|chi)|reporta(m)? a acumulada|1 ?[−-] ?(0[,.]967|Prob)|1 menos|[eé] o complemento|(p[- ]?valor|valor[- ]p|p-value)[^.]{0,30}(cauda (direita|superior)|right[- ]tail)|(cauda (direita|superior)|right[- ]tail)[^.]{0,30}(p[- ]?valor|p-value|0[,.]03)|sobram|(p[- ]?valor|valor[- ]p|p-value)\s*(\([^)]{0,20}\)\s*)?(=|≈|é|de|is)?\s*(cerca de |about )?0[,.]03|p[- ]?valor[^.]{0,15}0[,.]03|\bp ?(≈|=) ?0[,.]03|\b3[,.]3 ?%|[eé] a fun[cç][aã]o de distribui|Prob[^.]{0,30}fun[cç][aã]o de distribui|\b10[,.]5[^.]{0,60}(passa|acima|excede|maior|supera)[^.]{0,30}9[,.]49|n[aã]o [eé] (o|um) p[- ]?valor|invertid|n[ií]vel de signific[aâ]ncia observado|[aá]rea [aà] (esquerda|direita)|ser menor que J|P\(chi2? ?<=? ?J\))'
flags: i
target: last_message
---
