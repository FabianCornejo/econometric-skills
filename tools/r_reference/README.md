# Reference numbers from R

Some functions are replicated against R packages that the CI and the development container do not have. Each script
here runs **once**, on a machine with R; the numbers it writes are then fixed in the tests, with the script as their
documented origin. The scripts are not part of the skill package.

| Script | Replicates | Panel | Used by |
|---|---|---|---|
| `did_v250_reference.R` | `causal.did_imputation` against R `didimputation`; `causal.twfe_weights` against R `TwoWayFEWeights` | `tests/data/did_v250_panel.csv` (built by `make_did_v250_panel.py`, seed fixed, MD5 `7498036256b9e987f01997bc34a57ea0`) | `tests/test_v250.py` |

Run from the repository root:

```
Rscript tools/r_reference/did_v250_reference.R
```

It installs the missing CRAN packages, prints everything to `output/did_v250_reference.txt` and saves the numbers to
`output/did_v250_reference.json`. Commit both files: they are the record of where the test numbers came from (R and
package versions included).

If the script reports a different MD5 for the panel, the CSV changed on the way (usually CRLF line endings on Windows;
`.gitattributes` keeps `tests/data/*.csv` with LF). Do not fix numbers from a run on a different panel.
