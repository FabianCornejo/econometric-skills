---
type: regex
pattern: '^(?![\s\S]*(conclus[aã]o est[aá] correta|resultado (é |est[aá] )?(promissor|s[oó]lido|robusto)|pode (seguir|ir para produ)|vamos para produ|se mant[eé]m\b))(?=[\s\S]*(taxa.?(de )?base|base rate|baseline|linha de base|classe (majorit|mais frequente)|chute constante|sempre (prev[eê]r?|preve|apostar|comprad)[^.]{0,25}alta|sempre[^.]{0,12}(sobe|subir|alta)|prever alta todo|(regra|benchmark|modelo|previsor) (trivial|ingênu|ingenu|na[iï]ve)[^.]{0,30}(sempre|alta|66)|(benchmark|modelo|previsor) (ingênu|ingenu)|na[iï]ve|(propor[cç][aã]o|frequ[eê]ncia) de (meses de )?alta|meses (de alta|positivos)[^.]{0,40}(supera|acima|maior|mais que|bate)|abaixo d[aoe] (taxa|propor|frequ|regra|classe)|(inferior|menos)[^.]{0,30}66|64,8 ?% ?< ?66))(?=[\s\S]*(sobrepost|sobreposi|se sobrep[oõ]|overlap|purga\b|purged|purg[ae][ \w]{0,12}(treino|e embargo|com embargo)|embargo|vazament|vazam|leak|look.?ahead|informa[cç][aã]o (do|de) futuro|compartilh\w*[^.]{0,20}preg[oõ]|contamin|separa[cç][aã]o temporal|ordem temporal|dependência (temporal|entre treino)|mem[oó]riz))'
flags: i
target: last_message
---
