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

## 6. LLM extraction and coding

LLMs can read decisions and fill structured fields (outcome, amount, legal basis). Treat them like any other coder:
- Fixed prompt and codebook, model name and revision recorded (codes are reproducible only with the same weights).
- Validate on a hand-coded random sample with `validate_coding` (accuracy, kappa, sensitivity, specificity), separately by subgroup (court, year) when errors may differ.
- Keep the text span that supports each extracted value, to audit.
- Privacy and terms of use: decisions contain personal data (LGPD); do not send them to external APIs without a legal basis; prefer local models when needed (`hf_backend` runs locally).

RAG (retrieval-augmented generation) is a tool to *query* a corpus, not a measurement method; if it is used to build variables, the same validation applies.

### 6.1 Categorical codes: constrain the answer, keep the probabilities

For a closed set of categories, do not parse free text. Read the logits of the next token and restrict them to the label tokens (`llm_classify`): the answer can only be a category, and each document gets a probability per category instead of a hard label.

```python
next_logits, encode = hf_backend("Qwen/Qwen2.5-7B-Instruct", chat=True)   # local model
prompt = ("Read the decision and answer with ONE letter.\nA = granted\nB = denied\nC = partially granted\n"
          "D = no decision on the merits\n\nDecision: {text}\nAnswer with A, B, C or D.")
codes = llm_classify(df["text"], {"A": "granted", "B": "denied", "C": "partial", "D": "no_merits"},
                     prompt, next_logits, encode, label_prefix="")   # "" after a chat template, " " after "Answer:"
```

- **Use one-token answer codes** (A/B/C, 1/2/3) and define them in the prompt. A first-token constraint cannot separate labels that start with the same token ("procedente" and "parcialmente procedente" may share it); `llm_classify` refuses such label sets.
- **`mass_on_labels`**: the unconstrained probability that already fell on the codes. Low values mean the model wanted to write something else (a preamble, a refusal, another category); read those documents, the codebook may not fit them. Forcing a label does not make it right.
- **Probabilities are not calibrated** accuracy: a 0.9 does not mean 90% correct. Their value is to rank documents by doubt and to serve as the surrogate in §7.
- Any API or server that returns logits can be plugged in as `next_token_logits`; APIs that return only text cannot use this function (code the text answer and validate it as usual).

### 6.2 Uncertainty: which documents deserve human eyes

- **Label margin**: 1 − max probability, or sqrt(p(1−p)) for binary codes. Simple and, in practice, usually the most informative.
- **LogTokU** (`logtoku_uncertainty`; Ma et al., 2025): reads the top-k logits as evidence and separates *aleatoric* uncertainty (several answers well supported: the text is genuinely ambiguous) from *epistemic* uncertainty (little evidence for any answer: the model does not know). Probabilities cannot tell these apart. It needs raw logits, assumes the logit scale carries evidence (fails for models whose logits were rescaled), and its epistemic part has no ground truth. In tests with a small instruction model its scores barely varied across documents while the label margin did, so treat it as a complementary signal and check on your validation sample which score better predicts errors.

### 6.3 Design of the validation sample

Draw the documents to hand-code **at random with known probabilities**, decided before anyone codes:

```python
pi = dsl_sampling_probs(1 - codes.filter(like="p_").max(axis=1), n_expected=400, uniform_share=0.2)
df["to_code"] = draw_validation_sample(pi, seed=2026); df["pi"] = pi        # store both with the data
```

Give the draw its own seed: a seed that also generated, simulated or shuffled the data reuses the same random numbers, makes the sample a function of the data, and can leave a whole group out (`dsl_regression` then stops and names the regressor that does not vary). Oversampling uncertain documents gives more precision for the same coding budget (Zrnic & Candès, 2024); the uniform share keeps every probability away from zero. Picking documents by hand, or "the 400 the model was least sure about" with probability 1 and the rest with 0, breaks the design: there is no valid way back to the population.

## 7. Using text-based variables in regressions

- **As a regressor**: classification error is measurement error and attenuates coefficients towards zero (non-classical when errors correlate with other variables).
- **As an outcome**: errors uncorrelated with the regressors only add noise; errors that differ across groups bias the comparison (e.g. a regex tuned on one court's style).
- **Prevalence**: an imperfect classifier misstates shares; `rogan_gladen(p_obs, sensitivity, specificity)` corrects a proportion using the validation-sample error rates.
- **High accuracy is not enough**: 90% agreement with hand coding still biases a regression when the 10% of errors are concentrated in some groups (one court's style, longer decisions, a period). Error rates that look harmless in a validation table can move a coefficient by a third.

### 7.1 Design-based correction (`dsl_regression`)

With the hand-coded sample of §6.3, `dsl_regression` estimates OLS or logit using the LLM codes for every document and the human codes where they exist:

```python
df = df.join(codes)                                                     # p_granted, p_denied, ...
df["granted_true"] = np.where(df["to_code"], df["granted_human"], np.nan)
res = dsl_regression(df, y="granted", x=["female_judge", "log_amount"], labeled="to_code", pi="pi",
                     coded={"granted": ("granted_true", "p_granted")}, model="logit")
res["table"]; res["naive"]; res["design_effect"]
```

- **Why it works**: each document's moment condition is the LLM-based moment plus (R/π)·(human-based − LLM-based). The correction term has mean zero in the population for *any* coding error, so the estimate is consistent and the sandwich SE is valid whatever the LLM gets wrong; a better LLM only makes it more precise (Egami, Hinck, Stewart & Wei, 2023; written here as an inverse-probability-weighted prediction-powered moment, Angelopoulos et al., 2023; Zrnic & Candès, 2024).
- **What it needs**: a random sample with the stored π; human codes as the truth (codebook, ideally two coders and kappa); no missing surrogate values. Coded variables can be the outcome, regressors, or both.
- **Surrogate choice**: for a binary outcome, use the label probability rather than the 0/1 label (lower variance). Predictions from a classifier *trained on the hand-coded documents* must be cross-fitted (`sklearn.model_selection.cross_val_predict`) or the correction term is no longer mean zero; zero-shot LLM codes need nothing.
- **Read three outputs**: `table` (the estimate to report), `naive` (LLM codes treated as truth: the gap shows how much the coding errors matter), and `design_effect` (variance relative to using only the hand-coded documents; below 1 means the LLM codes added information). If the hand-coded logit alone is separated or singular, `design_effect` is NaN with a warning and the DSL estimate is still valid.
- **Identification checks**: before estimating, every regressor must vary among the hand-coded documents, the hand-coded design must have full rank and, for a logit, both outcomes must appear; otherwise the function stops with the cause. Code more documents or fix the draw.
- **Clustered designs**: pass `cluster=` at the level where the regressor of interest varies (for a judge characteristic, the judge, not the court); documents must still be drawn individually with their π. Report the number of clusters: with few judges the effective sample is the judges, not the decisions.
- **Sample size**: the SE shrinks with both the hand-coded n and the surrogate quality. Run a pilot (100–200 documents), look at `design_effect`, then decide how many more to code. Very small π give huge weights: keep the uniform share in §6.3.
- Simulation in `tests/test_llm_annotation_segments.py`: with x-dependent coding errors, treating LLM codes as truth gives 0.54 for a true effect of 0.80 (regressor) and 0.86 for 1.00 (logit outcome); the corrected estimates average 0.80 and 0.99 with 93–94% coverage of nominal 95% intervals.
- R users: the `dsl` package implements the original estimator (with cross-fitted predictions that also use covariates).
- **Similarity outcomes** (`similarity_to_reference`: cosine similarity of each document to a reference group's centroid, e.g. a judge's peers): control for document length and remove boilerplate; compare with a placebo reference group (future peers, random court).

## 8. Reporting

Corpus and selection; preprocessing; method and parameters (for LLMs: model, revision, prompt, answer codes); validation table (n, accuracy, kappa, sensitivity, specificity by class); how the hand-coded sample was drawn (π rule, expected and realized n); naive vs corrected estimates side by side; examples of correct and incorrect codings; robustness of the main result to alternative measures (regex vs classifier vs LLM).
