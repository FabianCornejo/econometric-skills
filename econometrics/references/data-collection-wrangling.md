# Collecting and preparing data

Use when the data are not yet a clean table: files in odd formats, several sources to combine, very large files, web pages, PDFs, administrative and judicial records. Code: `scripts/data_prep.py`. Official statistics APIs (BCB, IBGE, FRED…) are in `data-sources.md`.

## 1. Reading files

`read_any(path)` handles the usual cases:

| Situation | Symptom | Handling |
|---|---|---|
| Brazilian CSV (TSE, CNJ, Portal da Transparência) | `;` separator, latin-1 encoding, decimal comma | Separator and encoding are sniffed; numbers with `br_number` |
| Wrong encoding | "setor_censitÃ¡rio", "eleiÃ§Ã£o" | Re-read with the right encoding; `fix_mojibake` repairs strings already damaged |
| ".xls" that is really XML (Stata outreg2, some portals) | `Excel file format cannot be determined` | Detected and parsed (`read_spreadsheetml`) |
| ".xls" that is really HTML | `xlrd` error | `pd.read_html` |
| JSON from APIs | nested fields | `pd.json_normalize` (one row per record) |
| Stata `.dta` | — | `pd.read_stata` (keeps value labels as categories) |
| IDs with leading zeros (CPF, CNPJ, IBGE municipality codes, CNJ numbers) | 0 dropped, scientific notation | Read as `str` (`dtype=str`); never as float |

Then `audit(df)`: types, missing share, number of distinct values, examples, duplicate rows, constant columns, text columns that look numeric and mojibake warnings. Keep the raw file untouched and do every cleaning step in code.

## 2. Large files

Files of several hundred MB to GB (electoral results, judicial datasets, health records):
- Read only the columns needed (`usecols`) and filter while reading: `read_csv_filtered(path, keep=lambda d: d["DS_CARGO"].eq("Senador"), usecols=[...], dtype=str)`.
- Convert repeated strings to `category`; downcast numbers.
- Save intermediate results as Parquet (`df.to_parquet`), which is faster and keeps types.
- For larger-than-memory work, DuckDB (`duckdb.sql("select … from 'file.csv'")`) or Polars query files directly.

## 3. Combining sources

```python
m = safe_merge(cases, judges, on="judge_id", how="left", validate="m:1")
```

- State the expected cardinality: `1:1`, `m:1` (many cases, one judge) or `1:m`. A merge on a non-unique key (e.g. candidate **name** instead of the electoral ID; municipality name instead of the IBGE code) silently duplicates rows and inflates N.
- Inspect unmatched rows (`_merge == "left_only"`): accents, case, trailing spaces, leading zeros, different code systems (TSE vs IBGE municipality codes need a crosswalk).
- Aggregations before merging (cases per judge-year): `groupby([...]).agg(...)`; build keys with consistent types (an integer year and a string year do not match).
- Panel structure: one row per unit-period; check uniqueness of (unit, period) with `df.duplicated([unit, period]).sum() == 0`.

## 4. Variables

- Dummies: `pd.get_dummies(df, columns=[...], drop_first=True, dtype=int)` for scikit-learn; in statsmodels formulas use `C(var, Treatment('ref'))` and let the formula create them.
- Missing values: do not drop rows by reflex (`dropna` on all columns can remove most of the sample and select it); look at the pattern first; for regressions, report N and how missingness was handled. See "Missing data" below.

### Missing data

Start with `data_prep.missing_report(df, cols=[y, *xs], predictors=[...])`: missing share per variable, the complete-case N, the most frequent patterns, and whether observed variables predict missingness.

| Mechanism | Meaning | Testable? |
|---|---|---|
| MCAR | Missingness unrelated to anything | Can be rejected (missingness predicted by observables) |
| MAR | Depends only on observed variables | Not testable against MNAR |
| MNAR | Depends on the missing value itself (high earners hide income) | No — argue from the data-generating process |

What to do, for a regression of y on x:
- **Listwise deletion (complete cases)** is consistent whenever missingness depends only on the regressors — even if it is not MCAR — because conditioning on x does not change E(y\|x) (Wooldridge, exogenous sample selection). It is **biased** when missingness depends on y or on the error (e.g. firms that went bankrupt stop reporting). The cost otherwise is only precision. Verified by simulation in `tests/test_missing_data.py`.
- **Missing y**: imputing the outcome adds nothing to estimation under MAR given x; it matters only for MNAR (Heckman-type selection, `limited-dependent.md` §6).
- **Missing x, MAR**: **multiple imputation** with chained equations keeps the cases and propagates imputation uncertainty (Rubin's rules). Include y in the imputation model (leaving it out biases coefficients toward zero). Use at least as many imputations as the percentage of incomplete cases (20+).
  ```python
  from statsmodels.imputation.mice import MICEData, MICE
  imp = MICEData(df[["y", "x1", "x2", "x3"]])            # chained equations; numeric columns
  res = MICE("y ~ x1 + x2 + x3", sm.OLS, imp).fit(n_burnin=10, n_imputations=20)
  print(res.summary())                                    # combined by Rubin's rules
  ```
- **Avoid by default**: mean imputation (shrinks variance, biases slopes and SEs) and the missing-indicator method (fill with 0 and add a dummy), which is biased in observational data (Jones, 1996); the latter is acceptable only for pre-treatment covariates in a randomized experiment.
- **Time series**: interpolate only for plots. For estimation, state-space models (SARIMAX, UnobservedComponents) handle NaN in the Kalman filter natively.
- **Panels**: attrition that depends only on the unit effect is handled by FE; attrition related to shocks is not — compare early leavers with stayers at baseline.
- Report: N per model, the complete-case share, the `missing_report` test, and whether conclusions change between complete cases and multiple imputation.
- Dates: `pd.to_datetime(..., dayfirst=True)` for dd/mm/yyyy; check the parsed range.
- Deflate monetary values (IPCA) when comparing across years.

## 5. Web data

Order of preference:
1. **Official APIs and bulk downloads** — stable, documented, legal. Brazil: CNJ **DataJud** public API (case metadata for all courts), TSE open data (candidates, results, campaign finance), Portal da Transparência and Tesouro APIs, CADE (SEI search and open data), Querido Diário (municipal gazettes), INEP microdata, base dos dados (BigQuery mirror of many public datasets).
2. **Static pages**: `requests` + BeautifulSoup. `polite_get` adds a user agent, a minimum interval between requests, retries with backoff and **TLS verification** (`verify=False` should not be used to "fix" certificate errors).
3. **Dynamic pages** (content loaded by JavaScript, buttons, forms): first look for the JSON endpoint the page calls (browser developer tools → Network); use Selenium/Playwright only if there is none.

Good practice:
- Check `robots_allowed(url)` and the terms of use; identify yourself; throttle requests; scrape at off-peak hours; cache raw HTML so parsing can be re-run without re-downloading.
- **Pagination bugs** are common: a loop over pages whose URL never includes the page number downloads page 1 many times. Check that the number of unique records matches the total announced by the site.
- Personal data in judicial and administrative records falls under the LGPD: collect only what the research needs, pseudonymize, store securely, and do not republish identifiable data.
- Record the collection date: pages and search results change.

## 6. PDFs and scanned documents

- Digital PDFs: `pdf_text(path)` (PyMuPDF) returns text per page; tables with `pdfplumber` or `camelot`.
- Scanned pages have no text layer: `pdf_text(path, ocr=True)` rasterizes them and runs EasyOCR (Portuguese). Check OCR quality on a sample; character errors propagate into regex and classifiers.
- Remove repeated headers, footers and page markers before text analysis.

## 7. Reproducibility

Raw data read-only; a script or notebook that goes from raw to analysis file; versions of packages; collection dates and URLs; a data dictionary with source, unit and transformation for each variable.
