# Text as data

Use when the variables of interest are buried in documents: court decisions, tenders and contracts, regulatory opinions, laws, minutes, news, product reviews, earnings calls. The aim is a variable that enters an econometric model, and evidence that it measures what it claims to measure. Code: `scripts/text_data.py`. Getting the documents (scraping, PDFs, OCR) is in `data-collection-wrangling.md`.

## 1. Pipeline

1. **Corpus definition**: which documents, from where, which period; what is excluded and why. Selection into the corpus (only published decisions, only cases that reached a ruling) is a sample-selection problem.
2. **Preprocessing**: `normalize_text` (lowercase, accents, digits, spaces). Remove boilerplate that every document shares (headers, page markers like "fls. 12/30", signatures) before computing similarities.
3. **Measurement**, from simplest to most complex: dictionaries/regex → bag of words (TF-IDF) → topics → supervised classifiers → embeddings → LLM extraction.
4. **Validation** against a hand-coded random sample (always).
5. **Use** in the econometric model, accounting for measurement error.

Grimmer, Roberts & Stewart (2022) summarize the principles: text models are useful, not true; validate everything; the right method depends on the question, not the fashion.

## 2. Dictionaries and regular expressions

Best for information with a fixed form: parties ("Requerente: … Requerido: …"), case numbers (CNJ pattern `\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}`), amounts ("R$ 1.234,56"), dates, outcome formulas ("julgo procedente", "indefiro").

```python
flags = regex_flags(df["decision"], {
    "granted": r"julgo\s+procedente|defiro\s+(?:a|o)\s+(?:tutela|pedido)",
    "denied":  r"julgo\s+improcedente|indefiro",
    "partial": r"parcialmente\s+procedente"})
parties = regex_extract(df["decision"], {"plaintiff": r"requerente:\s*(.*?)\s*requerid[oa]:",
                                         "defendant": r"requerid[oa]:\s*(.*?)\s*juiz"})
```

- `regex_flags` returns 0/1 plus a context snippet for each match, so false positives can be audited; missing texts are NaN, not 0.
- Order and negation matter: "improcedente" contains "procedente"; "não defiro" contains "defiro". Test patterns on real examples (regex101.com) and on the hand-coded sample.
- Use vectorized pandas (`str.contains`, `str.extract`) or these helpers instead of looping row by row with growing lists, which is slow and error-prone (lists of different lengths silently misalign).

## 3. Bag of words, TF-IDF and topics

- `tfidf(texts, lang="pt")` — TF-IDF with Portuguese stopwords, accents stripped, unigrams and bigrams, terms in at least 2 documents and in at most 90% of them.
- `lda_topics(texts, k)` — LDA topics (scikit-learn); name topics by reading both the top words and the documents with the highest share. Choose k by interpretability and stability across seeds, not by perplexity alone. Structural topic models (STM, in R) let topic prevalence depend on covariates, which is often the research question.
- Stemming (Snowball) helps with small corpora and hurts interpretability; lemmatization (spaCy `pt_core_news`) is a better compromise.

## 4. Unsupervised clustering

```python
X, vec = tfidf(texts)
choose_k(X, range(2, 11))                # silhouette and Davies-Bouldin, not only the elbow
cl = cluster_texts(texts, k=5, X=X, vectorizer=vec)
cl["top_terms"], cl["representative_docs"]
```

Clusters are exploratory: they suggest categories, which then need a coding rule and validation before becoming variables. Embeddings (`embed`) capture meaning beyond shared words and usually give cleaner clusters than TF-IDF for short texts.

## 5. Supervised classification

```python
clf = train_classifier(hand_coded["text"], hand_coded["label"])   # TF-IDF + logistic regression
df["label_hat"] = clf["model"].predict(df["text"])
```

- Labels must come from **human coding** of a random sample (with a written codebook and, ideally, two coders and Cohen's kappa). Training a classifier on labels produced by a regex reproduces the regex and adds nothing.
- Split before anything is fitted: vectorizer and model are fitted inside a pipeline on the training part; report cross-validated macro-F1 and the held-out test F1, plus the confusion matrix. Accuracy alone is misleading with imbalanced classes.
- TF-IDF + logistic regression is a strong, interpretable baseline (top features per class). Embeddings + logistic regression or fine-tuned BERT-type models (BERTimbau; legal Portuguese models) help when wording varies a lot. Running dozens of classifiers and picking the best on the test set (e.g. "lazy" model comparisons) overfits the test set.

## 6. LLM extraction

LLMs can read decisions and fill structured fields (outcome, amount, legal basis). Treat them like any other coder:
- Fixed prompt and codebook, temperature 0, model version recorded.
- Validate on a hand-coded random sample with `validate_coding` (accuracy, kappa, sensitivity, specificity), separately by subgroup (court, year) when errors may differ.
- Keep the text span that supports each extracted value, to audit.
- Privacy and terms of use: decisions contain personal data (LGPD); do not send them to external APIs without a legal basis; prefer local models when needed.

RAG (retrieval-augmented generation) is a tool to *query* a corpus, not a measurement method; if it is used to build variables, the same validation applies.

## 7. Using text-based variables in regressions

- **As a regressor**: classification error is measurement error and attenuates coefficients towards zero (non-classical when errors correlate with other variables).
- **As an outcome**: errors uncorrelated with the regressors only add noise; errors that differ across groups bias the comparison (e.g. a regex tuned on one court's style).
- **Prevalence**: an imperfect classifier misstates shares; `rogan_gladen(p_obs, sensitivity, specificity)` corrects a proportion using the validation-sample error rates.
- **Design-based corrections**: with a hand-coded subsample, methods such as prediction-powered inference or design-based supervised learning (Egami et al., 2023) give valid inference on downstream estimates.
- **Similarity outcomes** (`similarity_to_reference`: cosine similarity of each document to a reference group's centroid, e.g. a judge's peers): control for document length and remove boilerplate; compare with a placebo reference group (future peers, random court).

## 8. Reporting

Corpus and selection; preprocessing; method and parameters; validation table (n, accuracy, kappa, sensitivity, specificity by class); examples of correct and incorrect codings; robustness of the main result to alternative measures (regex vs classifier vs LLM).
