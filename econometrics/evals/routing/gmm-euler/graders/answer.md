---
type: regex
pattern: '(estimation\.gmm\b|\b(est|es)\.gmm\s*\(|((fun[cç][aã]o|function) `?gmm`?(\(\))?|`?gmm`? (fun[cç][aã]o|function))[^\n.]{0,30}(m[oó]dulo `?estimation`?|module `?estimation`?|`estimation`|estimation\.py|estimation module)|\bgmm\b`?(\(\))? (do|of the|from|em|in) (m[oó]dulo `?estimation`?|module `?estimation`?|`estimation`|estimation\.py|estimation module)|(m[oó]dulo `?estimation`?|module `?estimation`?|`estimation`|estimation\.py|estimation module)`?[^\n.]{0,20}(tem|has|h[aá]|contains|cont[eé]m)[^\n.]{0,12}`?gmm\b)'
flags: i
target: last_message
---
