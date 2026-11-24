---
type: regex
pattern: '(\bCDF\b|acumulad[ao]|cumulative|1 ?[−-] ?(0[,.]967|Prob)|complemento|cauda (direita|superior)|right[- ]tail|(p[- ]?valor|valor[- ]p|p-value|\bp)\s*(\([^)]{0,20}\)\s*)?(=|≈|é|de|is)?\s*(cerca de |about )?0[,.]03|p ≈ 0[,.]03)'
flags: i
target: last_message
---
