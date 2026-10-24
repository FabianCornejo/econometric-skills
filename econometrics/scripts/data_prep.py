"""
Getting raw data into an analysis-ready table: reading awkward files, auditing,
merging safely, fixing Brazilian formats, extracting text from PDFs and collecting
web pages politely.

Functions
---------
read_any(path)                          csv/txt (sniffs separator and encoding), xlsx/xls, SpreadsheetML/HTML
                                        "xls" exports, json (nested -> flat), dta, parquet
read_csv_filtered(path, keep, ...)      stream a large CSV in chunks, keep only the rows/columns needed
read_spreadsheetml(path)                parse XML "Excel 2003" files (outreg2/esttab exports, gov portals)
audit(df)                               types, missing share, cardinality, examples, duplicates, constants
fix_mojibake(s)                         'setor_censitÃ¡rio' -> 'setor_censitário'
br_number(s)                            '1.234.567,89' -> 1234567.89 ; 'R$ 3,5 mil' handled up to the unit
safe_merge(left, right, on, how, validate)   merge with cardinality check and a match report
pdf_text(path, ocr=False)               text per page (PyMuPDF); optional OCR for scanned pages
robots_allowed(url) / polite_get(url)   check robots.txt; GET with rate limit, retries and a UA
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def _sniff(path, n=20000):
    raw = Path(path).read_bytes()[:n]
    for enc in ENCODINGS:
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(txt.split("\n", 5)[0] + "\n", delimiters=";,|\t").delimiter
    except csv.Error:
        sep = max(";,|\t", key=lambda c: txt.split("\n", 1)[0].count(c))
    return enc, sep


def read_spreadsheetml(path) -> pd.DataFrame:
    """'Excel 2003 XML' files (often saved with an .xls extension by Stata's outreg2,
    some government portals and old exporters). pandas cannot read them directly."""
    x = Path(path).read_text(encoding="latin-1")
    rows = re.findall(r"<Row[^>]*>(.*?)</Row>", x, re.S)
    data = []
    for r in rows:
        cells = re.findall(r"<Cell[^>]*?>(?:<Data[^>]*>(.*?)</Data>)?", r, re.S)
        data.append([re.sub(r"<[^>]+>", "", c).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()
                     for c in cells])
    width = max(map(len, data)) if data else 0
    return pd.DataFrame([d + [""] * (width - len(d)) for d in data])


def read_any(path, sheet=0, **kw) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    head = p.read_bytes()[:400].lower()
    if ext in (".csv", ".txt", ".tsv"):
        enc, sep = _sniff(p)
        return pd.read_csv(p, sep=kw.pop("sep", sep), encoding=kw.pop("encoding", enc), low_memory=False, **kw)
    if ext in (".xls", ".xlsx", ".xlsm"):
        if head.startswith(b"<?xml") and b"urn:schemas-microsoft-com:office:spreadsheet" in Path(path).read_bytes()[:3000].lower():
            return read_spreadsheetml(p)
        if head.lstrip().startswith((b"<html", b"<!doctype", b"<table")):
            return pd.read_html(p)[0]
        return pd.read_excel(p, sheet_name=sheet, **kw)
    if ext == ".json":
        obj = json.loads(p.read_text(encoding=_sniff(p)[0]))
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list)]
            obj = lists[0] if len(lists) == 1 else obj
        return pd.json_normalize(obj, **kw)
    if ext == ".dta":
        return pd.read_stata(p, **kw)
    if ext == ".parquet":
        return pd.read_parquet(p, **kw)
    raise ValueError(f"Unsupported extension: {ext}")


def read_csv_filtered(path, keep=None, usecols=None, chunksize: int = 200_000, dtype=None, **kw) -> pd.DataFrame:
    """Read a CSV too large for memory: only `usecols`, in chunks, keeping rows where
    keep(chunk) is True (e.g. lambda d: d['DS_CARGO'].eq('Senador')). Reading every column
    as str (dtype=str) avoids type guessing errors and leading-zero loss in IDs."""
    enc, sep = _sniff(path)
    parts = []
    for ch in pd.read_csv(path, sep=kw.pop("sep", sep), encoding=kw.pop("encoding", enc), usecols=usecols,
                          dtype=dtype, chunksize=chunksize, low_memory=False, **kw):
        parts.append(ch[keep(ch)] if keep else ch)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def audit(df: pd.DataFrame, max_examples: int = 3, show: bool = True) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().astype(str).unique()[:max_examples]
        rows.append({"column": c, "dtype": str(s.dtype), "missing_%": 100 * s.isna().mean(),
                     "n_unique": s.nunique(dropna=True), "examples": " | ".join(e[:30] for e in ex)})
    tab = pd.DataFrame(rows).set_index("column")
    if show:
        print(f"\nAudit: {len(df):,} rows × {df.shape[1]} columns; exact duplicate rows: {df.duplicated().sum():,}")
        print(tab.to_string(float_format=lambda v: f"{v:.1f}"))
        const = tab.index[tab["n_unique"] <= 1].tolist()
        if const:
            print("  Constant or empty columns:", const)
        obj_num = [c for c in df.columns if df[c].dtype == object and
                   df[c].dropna().astype(str).str.fullmatch(r"-?[\d\.\s]+,\d+|-?\d+(\.\d+)?").mean() > 0.9]
        if obj_num:
            print("  Text columns that look numeric (convert with br_number or pd.to_numeric):", obj_num)
        if any(df[c].astype(str).str.contains("Ã|Â", regex=True).any() for c in df.columns[:200] if df[c].dtype == object):
            print("  Mojibake detected (Ã, Â): file read with the wrong encoding; re-read or use fix_mojibake.")
    return tab


def fix_mojibake(s):
    def f(x):
        if isinstance(x, str) and re.search("[ÃÂ]", x):
            try:
                return x.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return x
        return x
    return s.map(f) if isinstance(s, (pd.Series, pd.Index)) else f(s)


def br_number(s):
    """Brazilian number strings to float: '1.234,56' -> 1234.56; strips 'R$', '%' and spaces."""
    def f(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        t = re.sub(r"[R$\s%]", "", str(x))
        if "," in t:
            t = t.replace(".", "").replace(",", ".")
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(f) if isinstance(s, pd.Series) else f(s)


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on, how: str = "left", validate: str = "m:1",
               show: bool = True) -> pd.DataFrame:
    """Merge with an explicit cardinality contract (validate='1:1', 'm:1', '1:m') and a
    report of matched / unmatched rows. Silent many-to-many joins (e.g. merging candidates
    on NAME, which is not unique) duplicate rows and are a classic source of wrong results."""
    out = pd.merge(left, right, on=on, how=how, validate=validate, indicator=True)
    if show:
        vc = out["_merge"].value_counts()
        print(f"\nMerge on {on} ({how}, {validate}): left {len(left):,} rows → result {len(out):,} rows")
        print("  both: {:,} | left_only: {:,} | right_only: {:,}".format(
            vc.get("both", 0), vc.get("left_only", 0), vc.get("right_only", 0)))
        if vc.get("left_only", 0):
            print("  Inspect unmatched keys (formatting, accents, leading zeros, trailing spaces) before dropping them.")
    return out


def pdf_text(path, ocr: bool = False, ocr_lang: str = "pt", min_chars: int = 30, zoom: float = 3.0) -> list[str]:
    """Text per page with PyMuPDF (pip install pymupdf). Pages with almost no text layer are
    scanned images: with ocr=True they are rasterized and read with easyocr (pip install easyocr)."""
    import pymupdf as fitz
    doc = fitz.open(path)
    pages = []
    reader = None
    for page in doc:
        t = page.get_text()
        if ocr and len(t.strip()) < min_chars:
            if reader is None:
                import easyocr
                reader = easyocr.Reader([ocr_lang], gpu=False)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            t = "\n".join(reader.readtext(pix.tobytes("png"), detail=0, paragraph=True))
        pages.append(t)
    doc.close()
    return pages


def robots_allowed(url: str, user_agent: str = "*") -> bool:
    from urllib.parse import urlparse
    from urllib.robotparser import RobotFileParser
    u = urlparse(url)
    rp = RobotFileParser(f"{u.scheme}://{u.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        return True
    return rp.can_fetch(user_agent, url)


_last_call = {"t": 0.0}


def polite_get(url: str, session=None, min_interval: float = 1.0, retries: int = 3, timeout: int = 30,
               user_agent: str = "research-crawler (contact: your-email)", **kw):
    """GET with a minimum interval between calls, exponential backoff on errors/429 and
    TLS verification ON (verify=False hides man-in-the-middle problems; fix certificates instead)."""
    import requests
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", user_agent)
    for k in range(retries):
        wait = _last_call["t"] + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call["t"] = time.time()
        try:
            r = s.get(url, timeout=timeout, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"status {r.status_code}")
            return r
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 ** k * min_interval)
