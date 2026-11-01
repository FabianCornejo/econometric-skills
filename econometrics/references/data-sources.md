# Data sources for econometrics

Priority: (1) files the user attaches; (2) official APIs via Python; (3) enabled connectors/plugins; (4) web search to locate a specific series. Always record the source, series code, extraction date and unit.

## International

| Source | How to access | Good for |
|---|---|---|
| FRED (St. Louis Fed) | `pandas_datareader.data.DataReader("CPIAUCSL", "fred", start)` or `fredapi` with a free key | US series and many global ones |
| World Bank | API `https://api.worldbank.org/v2/country/{ISO3}/indicator/{IND}?format=json` or `wbgapi` | Country panels |
| IMF | SDMX API; `imfp` package | WEO, IFS, balance of payments |
| OECD | SDMX API `https://sdmx.oecd.org/public/rest/` | Comparable indicators |
| Eurostat | `eurostat` package | Europe |
| BIS | SDMX API | Policy rates, credit, effective exchange rates |
| National statistical offices and central banks | Usually open APIs or CSV downloads | Country-specific series |

If a connector or plugin for official statistics or macro data (e.g., an OECD/IMF/World Bank aggregator, or a paid macro data provider) is enabled in the conversation, use it. Check that it is active before relying on it; do not assume.

## Brazil

### Central Bank (BCB) — SGS (time-series management system)

Public API, no key. Format:

```
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{CODE}/dados?formato=json&dataInicial=01/01/2015&dataFinal=31/12/2025
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{CODE}/dados/ultimos/{N}?formato=json
```

Daily series are limited to 10 years per request: split into windows.

```python
import pandas as pd, requests

def sgs(code, start="01/01/2010", end=None):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}"
    if end:
        url += f"&dataFinal={end}"
    df = pd.DataFrame(requests.get(url, timeout=60).json())
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.set_index("data")["valor"].rename(code)
```

Alternative: `pip install --break-system-packages python-bcb` → `from bcb import sgs; sgs.get({"selic": 432}, start="2015-01-01")` (handles window splitting itself).

Useful codes (confirm the name on the SGS portal before publishing):

| Code | Series | Frequency |
|---|---|---|
| 432 | Selic target set by Copom (% p.a.) | daily |
| 11 | Daily Selic (% p.d.) | daily |
| 4390 | Selic accumulated in the month (% p.m.) | monthly |
| 433 | IPCA (CPI) — monthly change (%) | monthly |
| 13522 | IPCA — 12-month accumulated (%) | monthly |
| 189 | IGP-M — monthly change (%) | monthly |
| 1 | USD/BRL PTAX selling rate | daily |
| 3698 | USD/BRL PTAX selling — monthly average | monthly |
| 24363 | IBC-Br activity index (not seasonally adjusted) | monthly |
| 24364 | IBC-Br (seasonally adjusted) | monthly |
| 4380 | Monthly GDP — current values (R$ million) | monthly |
| 24369 | Unemployment rate — PNAD Contínua (%) | monthly (rolling quarter) |

Market expectations (Focus survey): Olinda API `https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/` or `from bcb import Expectativas`.

### IBGE — SIDRA

```
https://apisidra.ibge.gov.br/values/t/{TABLE}/n1/all/v/{VARIABLE}/p/{PERIODS}
```

- `n1/all` = Brazil; `n3/all` = states; `n6/{code}` = municipality.
- `p/all`, `p/last 12`, `p/202001-202512`.
- The first row of the response is the header: discard it.
- Example: IPCA monthly change → table 1737, variable 63.
- To find a table/variable: https://sidra.ibge.gov.br (search by theme) or `https://servicodados.ibge.gov.br/api/v3/agregados/{TABLE}/metadados`.
- Package: `sidrapy` (`sidrapy.get_table(table_code="1737", territorial_level="1", ibge_territorial_code="all", variable="63", period="last 24")`).

Most used surveys: IPCA/INPC (prices), PMC (retail), PMS (services), PIM-PF (industry), PNAD Contínua (labor), quarterly national accounts (GDP).

### Ipeadata

OData API, no key; aggregates series from many sources (BCB, IBGE, FGV, Funcex...) with long histories.

```
http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='BM12_TJOVER12')
http://www.ipeadata.gov.br/api/odata4/Metadados('BM12_TJOVER12')
```

Response in `json["value"]`, fields `VALDATA` and `VALVALOR`. Package: `ipeadatapy`.

### Judicial, electoral and administrative records
- **CNJ DataJud** — public API with case metadata (class, subject, court, movements) for all Brazilian courts; key from the CNJ portal.
- **TSE open data** — candidates, results by electoral zone, campaign finance, party affiliation (CSV, `;`, latin-1).
- **Court websites** (e-SAJ, PJe consultations) — decisions and case data; scrape politely (see `data-collection-wrangling.md`).
- **CADE** — merger and conduct cases (SEI search, open data); **Querido Diário** — municipal official gazettes; **base dos dados** — cleaned public datasets on BigQuery.

### Other Brazilian sources
- CVM (listed-company open data): https://dados.cvm.gov.br
- National Treasury (government bonds, yield curve): Tesouro Transparente.
- B3 (prices, indices): no broad public API; prefer files the user provides.
- Federal Revenue (tax collection): open data.

## Market data (prices, indices, FX, futures)

| Source | How to access | Notes |
|---|---|---|
| Yahoo Finance | `yfinance.download(tickers, start, auto_adjust=True)` | Free, unofficial; recent versions return MultiIndex columns (`df.columns = df.columns.get_level_values(0)` for one ticker). Brazilian tickers end in `.SA` (`PETR4.SA`, `^BVSP`); FX as `BRL=X`; futures as `CL=F`, `GC=F` |
| Stooq | `pandas_datareader.data.DataReader("^SPX", "stooq")` | Free daily history, no key |
| FRED | yields, spreads, VIX (`VIXCLS`), exchange rates | Official, stable codes |
| B3 / CVM | files provided by the user | Official Brazilian market data |
| NEFIN-USP | nefin.com.br, "Risk factors" (CSV downloads) | Brazilian Market, SMB, HML, WML, IML factors and risk-free rate, daily and monthly (`finance_empirics.factor_model`) |
| Kenneth French data library | mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html; `pandas_datareader.famafrench` | Fama-French 3/5 factors, momentum, for the US, developed and emerging markets |
| BCB SGS series 12 | `python-bcb` (`sgs.get({"cdi": 12})`) | Daily CDI rate in % per day: the usual risk-free rate for Brazilian betas and Sharpe ratios |

Care with free market data: prices are adjusted retroactively for dividends and splits (a backtest run today and a year ago can differ), constituents of today's index suffer from survivorship bias, and there are gaps and stale prices on holidays. Use adjusted prices for returns and unadjusted prices for anything quoted in currency (levels, stop prices).

## If network access fails

Some environments block external domains. If a call fails:
1. Do not try to work around it with another command — the block is the same.
2. Ask the user to export the CSV/XLSX from the portal (most official portals have an export button) and attach it.
3. For a few observations, web search can locate values, but confirm them at the official source.

## Data good practices
- Document units: % per month vs. % per year, index vs. change, current vs. constant currency.
- Convert monthly rates into chained indices before deflating: `index = (1 + cpi/100).cumprod()`.
- Align dates to the start of the month (`to_period("M")`) to merge monthly series from different sources.
- Prefer non-seasonally-adjusted series plus your own treatment when modeling seasonality; use official seasonally adjusted series when the focus is elsewhere.
- Save the raw extracted data to a CSV next to the script, for reproducibility.
