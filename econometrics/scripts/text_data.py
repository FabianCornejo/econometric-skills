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
