"""
Text as data for empirical research: turning decisions, contracts, tenders, news or
reviews into variables that enter a regression — and validating them.

Functions
---------
normalize_text(s)                      lowercase, strip accents/digits/extra spaces
regex_flags(texts, patterns, window)   0/1 indicators + matched snippet with context (dictionary coding)
regex_extract(texts, patterns)         first capture group per named pattern (parties, amounts, dates)
tfidf(texts, lang="pt", ...)           TF-IDF matrix with Portuguese/English stopwords
lda_topics(texts, k)                   LDA topics: top words and document-topic shares
choose_k(X, ks)                        silhouette / Davies-Bouldin for K-means on text vectors
cluster_texts(texts or X, k)           K-means clusters, top terms and the most representative documents
train_classifier(texts, labels)        TF-IDF + logistic regression with stratified CV and a held-out test set
validate_coding(auto, manual)          agreement of automatic coding (regex/ML/LLM) with a hand-coded sample
rogan_gladen(p_obs, sens, spec)        prevalence corrected for misclassification
similarity_to_reference(X, ref_mask)   cosine similarity of each document to a reference set's centroid
embed(texts, model)                    sentence-transformer embeddings (optional dependency)

LLM annotation and valid downstream inference
---------------------------------------------
label_probs_from_logits(logits, ids)   next-token probabilities restricted to the label tokens + mass on labels
logtoku_uncertainty(logits, k)         aleatoric / epistemic uncertainty and reliability from top-k logits (LogTokU)
llm_classify(texts, labels, ...)       constrained label coding with any next-token-logits backend
hf_backend(model_name)                 local Hugging Face causal LM as that backend (optional dependency)
dsl_sampling_probs(score, n_expected)  known inclusion probabilities for hand coding, higher where the LLM is unsure
draw_validation_sample(pi, seed)       Bernoulli draw of the documents to hand-code
dsl_regression(df, y, x, ...)          OLS/logit with LLM-coded variables, valid with a probability hand-coded subsample
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

PT_STOPWORDS = set("""
a à ao aos aquela aquelas aquele aqueles aquilo as às até com como da das de dela delas dele deles depois do
dos e é ela elas ele eles em entre era eram essa essas esse esses esta estas este estes eu foi foram há isso
isto já la lhe lhes mais mas me mesmo meu minha muito na nas nem no nos nós o os ou para pela pelas pelo pelos
por qual quando que quem se sem ser seu sua suas seus só também te tem têm ter teu tua um uma umas uns você
vocês vos sobre sob são está estão pois onde num numa porque assim fls folha folhas art arts inc lei nº n
""".split())


# ---------------------------------------------------------------------------
# Cleaning and dictionary coding
# ---------------------------------------------------------------------------
def normalize_text(s, strip_accents: bool = True, remove_digits: bool = False) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    if strip_accents:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    if remove_digits:
        s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def regex_flags(texts: pd.Series, patterns: dict[str, str], window: int = 60,
                flags=re.IGNORECASE | re.DOTALL) -> pd.DataFrame:
    """One 0/1 column per pattern plus '<name>_snippet' with `window` characters of context
    around the first match (to audit false positives). Missing texts give NaN, not 0."""
    texts = pd.Series(texts)
    out = pd.DataFrame(index=texts.index)
    for name, pat in patterns.items():
        rx = re.compile(pat, flags)

        def f(t):
            if not isinstance(t, str):
                return (np.nan, None)
            m = rx.search(t)
            if not m:
                return (0, None)
            a, b = max(0, m.start() - window), min(len(t), m.end() + window)
            return (1, t[a:b].replace("\n", " "))
        res = texts.map(f)
        out[name] = res.map(lambda r: r[0])
        out[f"{name}_snippet"] = res.map(lambda r: r[1])
    return out


def regex_extract(texts: pd.Series, patterns: dict[str, str], flags=re.IGNORECASE | re.DOTALL) -> pd.DataFrame:
    """First capture group (or whole match) of each pattern, stripped. Use lookarounds or
    groups, e.g. {'plaintiff': r'requerente:\\s*(.*?)\\s*requerido:'}."""
    texts = pd.Series(texts)
    out = pd.DataFrame(index=texts.index)
    for name, pat in patterns.items():
        rx = re.compile(pat, flags)

        def f(t):
            if not isinstance(t, str):
                return None
            m = rx.search(t)
            if not m:
                return None
            g = m.group(1) if m.groups() else m.group(0)
            return re.sub(r"\s+", " ", g).strip() if g else None
        out[name] = texts.map(f)
    return out


# ---------------------------------------------------------------------------
# Vector representations
# ---------------------------------------------------------------------------
def tfidf(texts, lang: str = "pt", max_features: int = 5000, ngram_range=(1, 2), min_df: int | float = 2,
          max_df: float = 0.9, strip_accents: bool = True):
    """Returns (sparse matrix, fitted vectorizer). Fit on training texts only when the
    representation feeds a predictive model evaluated out of sample."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    stop = sorted({normalize_text(w) for w in PT_STOPWORDS}) if lang == "pt" else "english"
    vec = TfidfVectorizer(preprocessor=lambda s: normalize_text(s, strip_accents, remove_digits=True),
                          stop_words=stop, max_features=max_features, ngram_range=ngram_range,
                          min_df=min_df, max_df=max_df, token_pattern=r"(?u)\b[a-z]{3,}\b")
    X = vec.fit_transform([t if isinstance(t, str) else "" for t in texts])
    return X, vec


def lda_topics(texts, k: int = 8, n_top: int = 12, lang: str = "pt", max_features: int = 5000, seed: int = 0):
    """Latent Dirichlet Allocation on word counts. Returns (top_words DataFrame,
    document-topic matrix, model). Label topics by reading top words AND top documents."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    stop = sorted({normalize_text(w) for w in PT_STOPWORDS}) if lang == "pt" else "english"
    cv = CountVectorizer(preprocessor=lambda s: normalize_text(s, True, True), stop_words=stop,
                         max_features=max_features, min_df=2, max_df=0.9, token_pattern=r"(?u)\b[a-z]{3,}\b")
    C = cv.fit_transform([t if isinstance(t, str) else "" for t in texts])
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, learning_method="batch").fit(C)
    vocab = np.array(cv.get_feature_names_out())
    top = pd.DataFrame({f"topic_{i}": vocab[np.argsort(c)[::-1][:n_top]] for i, c in enumerate(lda.components_)})
    theta = pd.DataFrame(lda.transform(C), columns=[f"topic_{i}" for i in range(k)])
    return top, theta, lda


def embed(texts, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          batch_size: int = 32) -> np.ndarray:
    """Dense sentence embeddings (pip install sentence-transformers). Portuguese legal text:
    consider domain models such as 'stjiris/bert-large-portuguese-cased-legal-...' or
    'rufimelo/bert-large-portuguese-cased-sts'. Long documents are truncated by the model
    (often 256-512 tokens): chunk them and average, or embed the relevant section only."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    return np.asarray(m.encode([t if isinstance(t, str) else "" for t in texts], batch_size=batch_size,
                               show_progress_bar=False, normalize_embeddings=True))


# ---------------------------------------------------------------------------
# Unsupervised: clusters
# ---------------------------------------------------------------------------
def choose_k(X, ks=range(2, 11), seed: int = 0, sample: int = 5000) -> pd.DataFrame:
    """Silhouette (higher is better) and Davies-Bouldin (lower is better) for each k.
    The elbow of inertia alone is subjective; use these plus interpretability."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    if hasattr(X, "toarray"):
        if X.shape[0] * X.shape[1] <= 5e7:
            Xd = X.toarray()
        else:
            from sklearn.decomposition import TruncatedSVD
            Xd = TruncatedSVD(100, random_state=seed).fit_transform(X)
    else:
        Xd = np.asarray(X)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        lab = km.labels_
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, lab, metric="cosine", sample_size=min(sample, X.shape[0]),
                                                    random_state=seed),
                     "davies_bouldin": davies_bouldin_score(Xd, lab)})
    return pd.DataFrame(rows).set_index("k")


def cluster_texts(texts, k: int = 5, X=None, vectorizer=None, n_terms: int = 10, n_docs: int = 3,
                  seed: int = 0) -> dict:
    """K-means on TF-IDF (default) or on any matrix X (e.g. embeddings). Returns labels,
    top terms per cluster (TF-IDF only) and the documents closest to each centroid,
    which are what you read to name the clusters."""
    from sklearn.cluster import KMeans
    texts = list(texts)
    if X is None:
        X, vectorizer = tfidf(texts)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    lab = km.labels_
    dist = km.transform(X)
    reps = {c: [texts[i][:300] for i in np.argsort(dist[:, c])[:n_docs] if lab[i] == c] for c in range(k)}
    terms = None
    if vectorizer is not None:
        vocab = np.array(vectorizer.get_feature_names_out())
        terms = pd.DataFrame({f"cluster_{c}": vocab[np.argsort(km.cluster_centers_[c])[::-1][:n_terms]]
                              for c in range(k)})
    sizes = pd.Series(lab).value_counts().sort_index()
    return {"labels": lab, "sizes": sizes, "top_terms": terms, "representative_docs": reps, "model": km}


# ---------------------------------------------------------------------------
# Supervised: classifiers and validation
# ---------------------------------------------------------------------------
def train_classifier(texts, labels, test_size: float = 0.2, cv: int = 5, lang: str = "pt",
                     C: float = 4.0, class_weight: str | None = "balanced", seed: int = 0, show: bool = True) -> dict:
    """TF-IDF + logistic regression (a strong, interpretable baseline that often matches
    heavier models on legal text). Stratified CV on the training part, final report on
    an untouched test set. Labels must come from HUMAN coding, not from a regex: a model
    trained on regex labels can at best reproduce the regex."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    texts = pd.Series(texts).fillna("").astype(str).to_numpy()
    y = np.asarray(labels)
    stop = sorted({normalize_text(w) for w in PT_STOPWORDS}) if lang == "pt" else "english"
    pipe = make_pipeline(
        TfidfVectorizer(preprocessor=lambda s: normalize_text(s, True, True), stop_words=stop,
                        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
                        token_pattern=r"(?u)\b[a-z]{3,}\b"),
        LogisticRegression(C=C, max_iter=3000, class_weight=class_weight))
    Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=test_size, stratify=y, random_state=seed)
    cvs = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(cv, shuffle=True, random_state=seed), scoring="f1_macro")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    out = {"model": pipe, "cv_f1_macro_mean": cvs.mean(), "cv_f1_macro_sd": cvs.std(),
           "test_f1_macro": f1_score(yte, pred, average="macro"),
           "report": classification_report(yte, pred, output_dict=True, zero_division=0),
           "confusion": pd.crosstab(pd.Series(yte, name="true"), pd.Series(pred, name="pred"))}
    vec, clf = pipe.named_steps["tfidfvectorizer"], pipe.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    if clf.coef_.shape[0] == 1:
        c = clf.coef_[0]
        out["top_features"] = pd.DataFrame({f"→ {clf.classes_[1]}": vocab[np.argsort(c)[::-1][:15]],
                                            f"→ {clf.classes_[0]}": vocab[np.argsort(c)[:15]]})
    if show:
        print(f"\nText classifier: CV macro-F1 {cvs.mean():.3f} ± {cvs.std():.3f}; held-out macro-F1 {out['test_f1_macro']:.3f}")
        print(out["confusion"].to_string())
        base = pd.Series(ytr).value_counts(normalize=True).max()
        print(f"  Majority-class share in training: {base:.1%}. Accuracy near this means the model learned little.")
    return out


def validate_coding(auto, manual, positive=1, show: bool = True) -> dict:
    """Compare automatic codes (regex, classifier, LLM extraction) with a random hand-coded
    sample. Report this table in the paper: it is the measurement-error evidence."""
    from sklearn.metrics import cohen_kappa_score
    a = pd.Series(auto).reset_index(drop=True)
    m = pd.Series(manual).reset_index(drop=True)
    ok = a.notna() & m.notna()
    a, m = a[ok], m[ok]
    tp = ((a == positive) & (m == positive)).sum(); fp = ((a == positive) & (m != positive)).sum()
    fn = ((a != positive) & (m == positive)).sum(); tn = ((a != positive) & (m != positive)).sum()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    out = {"n": int(ok.sum()), "accuracy": float((a == m).mean()), "kappa": cohen_kappa_score(m, a),
           "sensitivity(recall)": sens, "specificity": spec, "precision": prec,
           "prevalence_manual": float((m == positive).mean()), "prevalence_auto": float((a == positive).mean())}
    if show:
        print("\nValidation against hand coding: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                             for k, v in out.items()))
        print("  Kappa > 0.8 strong, 0.6-0.8 substantial, < 0.6 revise the coding rule.")
    return out


def rogan_gladen(p_obs: float, sensitivity: float, specificity: float) -> float:
    """True prevalence implied by an imperfect classifier: (p + spec - 1) / (sens + spec - 1)."""
    return float(np.clip((p_obs + specificity - 1) / (sensitivity + specificity - 1), 0, 1))


def similarity_to_reference(X, reference_mask, exclude_self: bool = True) -> np.ndarray:
    """Cosine similarity of each row of X (TF-IDF or embeddings) to the centroid of the rows
    flagged in reference_mask (e.g. a judge's decision vs her peers' decisions). With
    exclude_self, a document in the reference set is compared with the centroid of the others."""
    from sklearn.preprocessing import normalize
    Xn = normalize(X)
    ref = np.asarray(reference_mask, bool)
    s = np.asarray(Xn[ref].sum(axis=0)).ravel()
    nref = ref.sum()
    rows = Xn.toarray() if hasattr(Xn, "toarray") else np.asarray(Xn)
    sims = np.empty(rows.shape[0])
    for i in range(rows.shape[0]):
        c = s - rows[i] if (exclude_self and ref[i]) else s
        nc = nref - 1 if (exclude_self and ref[i]) else nref
        c = c / max(nc, 1)
        denom = np.linalg.norm(c)
        sims[i] = rows[i] @ c / denom if denom > 0 else np.nan
    return sims


# ---------------------------------------------------------------------------
# LLM annotation: constrained label coding and token-level uncertainty
# ---------------------------------------------------------------------------
def _softmax(z, axis=-1):
    z = np.asarray(z, float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def label_probs_from_logits(logits, label_token_ids) -> dict:
    """Probabilities over a closed label set from next-token logits.

    Restricting the softmax to the label tokens is the same as setting every other logit to
    -inf (a constrained-decoding LogitsProcessor): the answer can only be one of the labels.
    `mass_on_labels` is the share of the *unconstrained* probability that already fell on the
    labels; a low value means the model wanted to say something else (a preamble, a refusal,
    another category), so the forced label deserves a human look.

    logits: (V,) or (n, V). label_token_ids: list of token ids, one per label.
    """
    L = np.atleast_2d(np.asarray(logits, float))
    ids = np.asarray(label_token_ids, int)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Two labels share the same token id; use labels whose first tokens differ "
                         "(e.g. answer codes A/B/C mapped to categories in the prompt).")
    full = _softmax(L, axis=1)
    probs = _softmax(L[:, ids], axis=1)
    out = {"probs": probs, "label_hat": probs.argmax(axis=1), "mass_on_labels": full[:, ids].sum(axis=1)}
    if np.ndim(logits) == 1:
        out = {"probs": probs[0], "label_hat": int(out["label_hat"][0]), "mass_on_labels": float(out["mass_on_labels"][0])}
    return out


def logtoku_uncertainty(logits, k: int = 25) -> dict:
    """Token-level uncertainty from raw logits (LogTokU; Ma, Chen, Wang & Zhang, 2025, arXiv:2502.00290).

    The top-k logits are read as Dirichlet evidence alpha_j (negative logits are treated as zero evidence):
      aleatoric  AU = -sum_j (alpha_j/alpha_0) [digamma(alpha_j + 1) - digamma(alpha_0 + 1)]
      epistemic  EU = k / sum_j (alpha_j + 1)
      reliability  R = 1 / (AU * EU)
    High AU with low EU: several answers are strongly supported (genuine ambiguity in the text).
    High EU: little total evidence for any answer (the model does not know). Softmax probabilities
    cannot separate these two cases, which is the motivation of the method.

    Caveats: needs raw logits (gray-box; not available from most commercial APIs); assumes the logit
    scale carries evidence, which fails for models whose logits were rescaled (e.g. some distilled
    models); EU has no ground truth. Use it to rank documents for human review or to set sampling
    probabilities (`dsl_sampling_probs`), never as a measure of accuracy by itself.
    """
    from scipy.special import digamma
    L = np.atleast_2d(np.asarray(logits, float))
    k = int(min(k, L.shape[1]))
    top = -np.sort(-L, axis=1)[:, :k]
    a = np.clip(top, 0, None)
    a0 = a.sum(axis=1)
    w = np.where(a0[:, None] > 0, a / np.where(a0 > 0, a0, 1)[:, None], 1.0 / k)
    au = -(w * (digamma(a + 1) - digamma(a0[:, None] + 1))).sum(axis=1)
    au = np.where(au > 1e-12, au, 0.0)                     # >= 0 in theory; avoids -0.0 -> reliability -inf
    eu = k / (a + 1).sum(axis=1)
    with np.errstate(divide="ignore"):
        rel = 1.0 / (au * eu)
    out = {"aleatoric": au, "epistemic": eu, "reliability": rel}
    if np.ndim(logits) == 1:
        out = {kk: float(v[0]) for kk, v in out.items()}
    return out


def llm_classify(texts, labels: dict | list, prompt: str, next_token_logits, encode,
                 label_prefix: str = " ", k: int = 25, show: bool = True) -> pd.DataFrame:
    """Code each document into one of a closed set of labels with a local LLM, keeping the
    probabilities and an uncertainty score for every decision.

    texts             iterable of documents
    labels            {"A": "granted", "B": "denied", ...} (answer code -> category) or a list of codes
    prompt            template with {text}; it must end where the answer code starts, e.g. "...Answer:"
    next_token_logits callable(prompt_str) -> 1-D array of vocabulary logits for the next token
    encode            callable(str) -> list of token ids (the backend tokenizer, without special tokens)
    label_prefix      text placed before each code when tokenizing it (" A" after "Answer:" is usually
                      a single token different from "A")

    Use `hf_backend(model_name)` for a local Hugging Face model, or wrap any server that returns
    logits. Returns one row per document: label_hat, p_<category> per label, mass_on_labels,
    aleatoric, epistemic, reliability. Validate against hand coding (`validate_coding`) and use
    `dsl_regression` when the codes enter a regression.
    """
    codes = list(labels.keys()) if isinstance(labels, dict) else list(labels)
    names = [labels[c] for c in codes] if isinstance(labels, dict) else codes
    ids = []
    for c in codes:
        toks = encode(label_prefix + c)
        if not toks:
            raise ValueError(f"Label code {c!r} encodes to no tokens.")
        ids.append(int(toks[0]))
    if len(set(ids)) != len(ids):
        clash = [c for c, i in zip(codes, ids) if ids.count(i) > 1]
        raise ValueError(f"Codes {clash} start with the same token; a first-token constraint cannot tell them "
                         "apart. Use single-letter or numeric codes and map them to categories in the prompt.")
    rows = []
    for t in texts:
        lg = np.asarray(next_token_logits(prompt.format(text=t)), float).ravel()
        lp = label_probs_from_logits(lg, ids)
        unc = logtoku_uncertainty(lg, k=k)
        row = {"label_hat": names[lp["label_hat"]], "mass_on_labels": lp["mass_on_labels"], **unc}
        row.update({f"p_{n}": float(p) for n, p in zip(names, lp["probs"])})
        rows.append(row)
    out = pd.DataFrame(rows)
    if show and len(out):
        low = (out["mass_on_labels"] < 0.5).mean()
        print(f"\nLLM coding of {len(out)} documents: label shares "
              + ", ".join(f"{n} {s:.1%}" for n, s in out["label_hat"].value_counts(normalize=True).items()))
        print(f"  {low:.1%} of documents put less than half of the unconstrained probability on the labels: "
              "read some of them; the prompt or the codebook may not fit those texts.")
        print("  These are predictions, not measurements: validate on a random hand-coded sample.")
    return out


def hf_backend(model_name: str, chat: bool = False, device: str | None = None, revision: str | None = None):
    """(next_token_logits, encode) for a local Hugging Face causal LM (needs `transformers` and `torch`).

    Runs locally, so documents with personal data never leave the machine. Record model_name and
    revision in the report: codes are only reproducible with the same weights. With chat=True the
    prompt is wrapped in the model's chat template and the logits are those of the first answer token.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    try:                                                   # transformers >= 4.56 / 5.x name the argument `dtype`
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype="auto")
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()

    def encode(s):
        return tok.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def next_token_logits(prompt_str):
        if chat and getattr(tok, "chat_template", None):
            # render to text first: the tensor-returning form changed type across transformers versions
            text = tok.apply_chat_template([{"role": "user", "content": prompt_str}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        else:
            ids = tok(prompt_str, return_tensors="pt").input_ids
        logits = model(ids.to(model.device)).logits[0, -1]
        return logits.float().cpu().numpy()

    return next_token_logits, encode


# ---------------------------------------------------------------------------
# Valid inference with LLM-coded variables (design-based correction)
# ---------------------------------------------------------------------------
def dsl_sampling_probs(score, n_expected: float, uniform_share: float = 0.2, max_pi: float = 1.0) -> np.ndarray:
    """Inclusion probabilities for the hand-coded validation sample.

    pi_i mixes a uniform part with a part proportional to `score` (higher = less sure), scaled so
    that sum(pi) = n_expected and capped at max_pi. Sampling more where the LLM is unsure reduces
    the variance of `dsl_regression` (Zrnic & Candes, 2024, "active statistical inference"); the
    uniform part keeps every pi_i away from zero, which is what keeps the weights 1/pi stable.
    Good scores: sqrt(p(1-p)) of the label probability, 1 - max label probability, or
    1/reliability from `logtoku_uncertainty`.
    The probabilities must be fixed BEFORE anything is hand-coded and stored with the data.
    """
    s = np.asarray(score, float)
    n = len(s)
    if not 0 < n_expected <= n:
        raise ValueError("n_expected must be between 0 and the number of documents.")
    s = np.where(np.isfinite(s), s, np.nanmax(s[np.isfinite(s)]) if np.isfinite(s).any() else 1.0)
    s = np.clip(s, 0, None)
    base = s / s.sum() if s.sum() > 0 else np.full(n, 1 / n)
    shape = (1 - uniform_share) * base + uniform_share / n
    pi = np.minimum(shape * n_expected, max_pi)
    for _ in range(100):                                   # redistribute what the cap removed
        short = n_expected - pi.sum()
        free = pi < max_pi - 1e-12
        if short <= 1e-9 or not free.any():
            break
        pi[free] = np.minimum(pi[free] + short * shape[free] / shape[free].sum(), max_pi)
    return pi


def draw_validation_sample(pi, seed: int = 0) -> np.ndarray:
    """Independent Bernoulli(pi_i) draws: True = send the document to human coders."""
    rng = np.random.default_rng(seed)
    return rng.random(len(pi)) < np.asarray(pi, float)


def dsl_regression(df: pd.DataFrame, y: str, x: list[str], labeled, pi, coded: dict | None = None,
                   model: str = "ols", add_const: bool = True, cluster=None, show: bool = True) -> dict:
    """Regression where some variables were coded by an LLM (or any classifier), with valid
    standard errors thanks to a hand-coded subsample drawn with known probabilities.

    Every model is a moment condition E[psi(D; b)] = 0 (OLS: x(y - x'b); logit: x(y - Lambda(x'b))).
    With surrogate data D_hat for all documents and true data D for the hand-coded ones,
        psi_tilde_i = psi(D_hat_i; b) + (R_i / pi_i) [psi(D_i; b) - psi(D_hat_i; b)]
    has the same expectation as psi(D_i; b) for ANY surrogate, however biased, as long as R_i was
    drawn with the known probability pi_i > 0. So the estimate is consistent and the sandwich SE is
    valid; a better surrogate only makes it more precise. This is the design-based idea of DSL
    (Egami, Hinck, Stewart & Wei, 2023) written as an inverse-probability-weighted prediction-powered
    moment (Angelopoulos et al., 2023; Zrnic & Candes, 2024).

    df       one row per document
    y, x     variable names in the model (no formula; build dummies/interactions beforehand)
    labeled  boolean array/column name: True for hand-coded documents
    pi       inclusion probabilities used to draw `labeled` (array or column name)
    coded    {var: (true_col, surrogate_col)} for each variable measured by the LLM; true_col is NaN
             outside the hand-coded sample. For a binary outcome, the surrogate can be the label
             probability rather than the 0/1 label (lower variance). Variables not in `coded` are
             taken as exact from df[var].
    model    "ols" or "logit" (logit: the outcome must be 0/1 in the hand-coded data)
    cluster  optional array/column of cluster ids for clustered SEs (draws must then be by document)

    Returns dict: table (DSL), naive (surrogates treated as truth), labeled_only (IPW on the
    hand-coded documents alone), n, n_labeled, design_effect.
    """
    from scipy import stats
    coded = coded or {}
    col = lambda v: np.asarray(df[v] if isinstance(v, str) else v)
    R = col(labeled).astype(bool)
    p = col(pi).astype(float)
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("Every inclusion probability must be in (0, 1].")
    names = list(x)
    n = len(df)

    def build(kind):
        def get(v):
            if v in coded:
                return np.asarray(df[coded[v][0] if kind == "true" else coded[v][1]], float)
            return np.asarray(df[v], float)
        X = np.column_stack([get(v) for v in names]) if names else np.empty((n, 0))
        if add_const:
            X = np.column_stack([np.ones(n), X])
        return get(y), X

    yh, Xh = build("hat")
    yt, Xt = build("true")
    if np.isnan(yh).any() or np.isnan(Xh).any():
        raise ValueError("Surrogate data have missing values; impute or drop those documents first.")
    if np.isnan(yt[R]).any() or np.isnan(Xt[R]).any():
        raise ValueError("Some hand-coded documents have no true value for a coded variable.")
    yt = np.where(R, yt, 0.0); Xt = np.where(R[:, None], Xt, 0.0)
    w = np.where(R, 1.0 / p, 0.0)
    cols = (["const"] if add_const else []) + names

    def psi(b):
        if model == "ols":
            eh, et = yh - Xh @ b, yt - Xt @ b
        elif model == "logit":
            eh, et = yh - 1 / (1 + np.exp(-Xh @ b)), yt - 1 / (1 + np.exp(-Xt @ b))
        else:
            raise ValueError("model must be 'ols' or 'logit'")
        return Xh * eh[:, None] + w[:, None] * (Xt * et[:, None] - Xh * eh[:, None])

    def jac(b):                                            # d sum(psi) / d b  (negative definite in large samples)
        if model == "ols":
            gh = np.ones(n); gt = np.ones(n)
        else:
            ph, pt = 1 / (1 + np.exp(-Xh @ b)), 1 / (1 + np.exp(-Xt @ b))
            gh, gt = ph * (1 - ph), pt * (1 - pt)
        return -((Xh * gh[:, None]).T @ Xh + (Xt * (w * gt)[:, None]).T @ Xt - (Xh * (w * gh)[:, None]).T @ Xh)

    def solve(start):
        b = start.copy()
        for _ in range(100):
            step = np.linalg.solve(jac(b), psi(b).sum(axis=0))
            b = b - step
            if np.max(np.abs(step)) < 1e-10:
                break
        return b

    def sandwich(b, P, J):
        if cluster is not None:
            g = pd.Series(col(cluster)).values
            P = pd.DataFrame(P).groupby(g).sum().values
            G = P.shape[0]
            meat = P.T @ P * G / (G - 1)
        else:
            meat = P.T @ P
        Ji = np.linalg.inv(J)
        return Ji @ meat @ Ji.T

    def table(b, V):
        se = np.sqrt(np.diag(V)); z = b / se
        q = stats.norm.ppf(0.975)
        return pd.DataFrame({"coef": b, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)),
                             "ci_low": b - q * se, "ci_high": b + q * se}, index=cols)

    b0 = np.linalg.lstsq(Xh, yh, rcond=None)[0] if model == "ols" else np.zeros(Xh.shape[1])
    if model == "logit":                                   # naive logit on surrogates as start
        for _ in range(50):
            ph = 1 / (1 + np.exp(-Xh @ b0))
            b0 = b0 + np.linalg.solve((Xh * (ph * (1 - ph))[:, None]).T @ Xh, Xh.T @ (yh - ph))
    fit0 = Xh @ b0 if model == "ols" else 1 / (1 + np.exp(-Xh @ b0))
    g0 = np.ones(n) if model == "ols" else fit0 * (1 - fit0)
    naive_V = sandwich(b0, Xh * (yh - fit0)[:, None], -(Xh * g0[:, None]).T @ Xh)
    b = solve(b0)
    V = sandwich(b, psi(b), jac(b))

    # hand-coded documents alone, inverse-probability weighted (what you would do without the LLM)
    Xl, yl, wl = Xt[R], yt[R], w[R]
    if model == "ols":
        bl = np.linalg.solve((Xl * wl[:, None]).T @ Xl, (Xl * wl[:, None]).T @ yl)
        Pl = Xl * (wl * (yl - Xl @ bl))[:, None]; Jl = -(Xl * wl[:, None]).T @ Xl
    else:
        bl = b.copy()
        for _ in range(100):
            pl = 1 / (1 + np.exp(-Xl @ bl))
            step = np.linalg.solve((Xl * (wl * pl * (1 - pl))[:, None]).T @ Xl, (Xl * wl[:, None]).T @ (yl - pl))
            bl = bl + step
            if np.max(np.abs(step)) < 1e-10:
                break
        pl = 1 / (1 + np.exp(-Xl @ bl))
        Pl = Xl * (wl * (yl - pl))[:, None]; Jl = -(Xl * (wl * pl * (1 - pl))[:, None]).T @ Xl
    Vl = np.linalg.inv(Jl) @ (Pl.T @ Pl) @ np.linalg.inv(Jl).T

    out = {"table": table(b, V), "naive": table(b0, naive_V), "labeled_only": table(bl, Vl),
           "n": n, "n_labeled": int(R.sum()), "min_pi": float(p.min()),
           "design_effect": pd.Series(np.diag(V) / np.diag(Vl), index=cols, name="var_DSL / var_labeled_only")}
    if show:
        print(f"\nDSL {model.upper()} with LLM-coded {sorted(set(coded) & set([y] + names)) or '(none)'}: "
              f"{n} documents, {R.sum()} hand-coded (min pi = {p.min():.3f}).")
        print(out["table"].round(4).to_string())
        print("\n  Naive estimates (LLM codes treated as truth) — biased whenever coding errors relate to the model:")
        print(out["naive"][["coef", "se"]].round(4).to_string())
        print("  Variance relative to using the hand-coded documents alone (< 1 = the LLM codes add precision):")
        print("  " + ", ".join(f"{k} {v:.2f}" for k, v in out["design_effect"].items()))
        if p.min() < 0.02:
            print("  WARNING: some pi are below 0.02; their weights 1/pi are large and the SEs can be unstable.")
        print("  Valid only if `labeled` was drawn with these pi, decided before hand coding.")
    return out
