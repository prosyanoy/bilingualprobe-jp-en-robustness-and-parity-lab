# BilingualProbe: JP/EN Robustness and Parity Lab

A test harness that measures whether a text classifier performs equally well in Japanese and English, and how it breaks under realistic perturbations.

## Features

- **Parallel dataset generator.** Template-based JP/EN binary sentiment pairs (`positive`/`negative`). Each pair shares an id and a label. Five sentence frames, including a contrastive "A is good, but B is bad" frame. Japanese is rendered in polite or plain register. The test split never shares a string with training.
- **Public dataset loader.** `load_public_dataset` reads CSV/TSV/JSONL with `text,label,lang` columns (e.g. an export of multilingual Amazon reviews or JGLUE-MARC-ja). Pass `--public-data` or set `BP_PUBLIC_DATA`.
- **Three local classifiers**, all trained on mixed-language data:
  - `tfidf_lr`: TF-IDF char 1-3-grams + logistic regression.
  - `tfidf_lr_nfkc`: the same with Unicode NFKC normalisation.
  - `neural`: small MLP (PyTorch if installed, otherwise scikit-learn `MLPClassifier`). Features are TF-IDF, or optional multilingual Hugging Face embeddings.
- **Perturbation suite** (per-language failure rates):
  `typo`, `width` (halfwidth katakana for JA, fullwidth Latin for EN), `politeness_casual` and `politeness_keigo` (JA only), `negation` (label flips), `synonym` (held-out synonyms).
- **Parity score.** Accuracy gap EN - JA with a 95% **paired** bootstrap CI over parallel pairs, plus a min/max ratio. Each perturbation also gets a JA-EN failure-rate gap with CI and a significance flag.
- **Report and dashboard.** `report.md` / `results.json` with models side by side and the worst failure categories ranked, and a Streamlit app with an interactive "probe a sentence" tab.

## Architecture

```
bilingualprobe/
  lexicon.py    adjectives (+ held-out synonyms), nouns
  data.py       template generator, public loader, JSONL IO
  perturb.py    perturbation functions + registry
  models.py     TF-IDF/LR, NFKC variant, neural classifier, optional embeddings
  stats.py      bootstrap CI, paired/independent gap, parity ratio
  evaluate.py   clean + perturbed evaluation per model
  report.py     ranking of worst categories, Markdown report
  pipeline.py   data -> train -> perturb -> evaluate -> write artifacts
  cli.py        `python -m bilingualprobe {generate,run}`
  dashboard.py  Streamlit app
tests/          pytest suite
```

Perturbed inputs are built once, model-independently, and then scored by every model, so models are compared on identical inputs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional extras
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
cp .env.example .env   # optional; no secrets are required
```

## Run

```bash
python -m bilingualprobe run                 # writes artifacts/results.json + report.md, prints the report
python -m bilingualprobe run --models tfidf_lr neural --perturbations typo negation
python -m bilingualprobe run --public-data my_reviews.csv
python -m bilingualprobe generate --out artifacts/data   # just dump the synthetic data
streamlit run bilingualprobe/dashboard.py    # dashboard (runs the pipeline itself if no results exist)
```

A default run (1200 train rows, 600 test rows, 1000 bootstrap resamples) takes a few seconds on a laptop CPU.

## Test

```bash
pytest
```

## Results

Default run: `python -m bilingualprobe run` (seed 0, 1,200 training rows, 600 test rows = 300 parallel pairs, 1,000 bootstrap resamples, scikit-learn backend for `neural`). **All numbers come from synthetic template data (see limits below).**

### Clean accuracy and parity (EN vs JA)

| model | acc EN | acc JA | gap EN-JA [95% CI] | parity ratio |
|---|---|---|---|---|
| tfidf_lr | 96.7% | 86.3% | +10.3 pp [+6.7, +14.3] | 0.893 |
| tfidf_lr_nfkc | 96.7% | 86.3% | +10.3 pp [+6.7, +14.3] | 0.893 |
| neural (MLP) | 96.3% | 85.0% | +11.3 pp [+7.3, +15.3] | 0.882 |

The CI excludes 0 for every model, so the JA deficit is not bootstrap noise on this test set.

### Failure rate by perturbation (P(wrong after | right before))

| perturbation | tfidf_lr EN / JA | tfidf_lr_nfkc EN / JA | neural EN / JA |
|---|---|---|---|
| negation (label flips) | 100.0% / 79.9% | 100.0% / 79.9% | 97.2% / 88.6% |
| synonym | 21.4% / 42.1% | 21.4% / 42.1% | 36.0% / 41.2% |
| width | 48.3% / 8.1% | 0.0% / 0.0% | 49.5% / 7.5% |
| typo | 2.1% / 7.8% | 2.1% / 7.8% | 2.4% / 7.1% |
| politeness_casual (JA only) | n/a / 7.4% | n/a / 7.4% | n/a / 3.0% |
| politeness_keigo (JA only) | n/a / 1.1% | n/a / 1.1% | n/a / 2.4% |

### Worst failure categories (tfidf_lr)

1. `negation` / EN: 100.0%
2. `negation` / JA: 79.9%
3. `width` / EN: 48.3%
4. `synonym` / JA: 42.1%
5. `synonym` / EN: 21.4%

Readings that the numbers support:

- Negation is the worst category everywhere, because no model sees negated training examples. It says little about JA vs EN.
- Width variants break English (fullwidth Latin) but barely touch Japanese for the non-normalising models, and NFKC normalisation removes the failure entirely. An input-normalisation fix beat a model change.
- Synonym swaps hurt Japanese more than English for the TF-IDF models (gap CI excludes 0); for the MLP the difference is not significant.
- The Japanese-specific politeness shifts (casual, keigo) barely hurt (1-7%). On this data, register is a weak failure source, probably because the sentiment words are unchanged by it. That is a limit of the templates, not evidence that keigo is harmless in real text.

I did not run an ablation to determine *why* Japanese is 10 pp worse on clean data. The contrastive "A is good, but B is bad" frame is my hypothesis, since the label comes from the second clause, but it is untested.

## Limits of the synthetic data

- Every sentence comes from five templates with 10 nouns and 10 adjectives. Models reach ~96% on English because the task is easy, so absolute accuracy says nothing about real-world performance.
- The Japanese is template-generated and I wrote the grammar rules (negation, casual/keigo rewrites) myself. They cover only regular i-adjective predicates, not natural Japanese. Nouns and adjectives are paired freely, so some sentences are semantically odd.
- EN and JA are parallel by construction, which makes the paired bootstrap valid but also makes the languages artificially comparable. Real translations differ in length, ambiguity and register.
- Perturbation failure rates measure sensitivity to *these* edits on *this* vocabulary. The train/test split shares nouns, adjectives and frames; it only guarantees no identical strings.
- Parity gaps here show how the harness behaves, not how any production model performs. For realistic text use `--public-data`; then the lexicon-based perturbations mostly will not apply, and only typo and width remain meaningful.
- No human-written or native-speaker-validated Japanese test set was used.

## What I learned

- A single parity number (gap plus CI) is easy to compute, but its meaning depends on the sampling design. Pairing by sentence id gives a tighter and more honest CI than treating the two languages as independent samples.
- Conditioning failure rates on "correct before perturbation" matters; otherwise baseline errors, which differ by language, leak into the robustness comparison.
- Unicode details are a real failure source: without NFKC, fullwidth Latin broke the English models almost half the time while barely affecting Japanese, which is the opposite of what I first expected.
- Perturbations need to be checked for validity. Several of my first-draft rewrite rules produced wrong Japanese (verb vs i-adjective conjugation, plain-form past), so I restricted templates to forms the rules handle and wrote tests for each rule.
- A benchmark can come back "all fine" for the wrong reason (politeness shifts barely failing); a low failure rate needs a look at whether the perturbation actually touched what the model relies on.
- Environment constraints are worth documenting: the PyTorch path and the dashboard tables could not be run in my low-disk build environment, and the README says exactly what was and was not verified.

## About this project (CV / interview notes)

CV line: *Designed a bilingual robustness benchmark with Japanese-specific perturbations and a parity metric with confidence intervals.*

Be ready to discuss only what the repo supports: how the perturbations and parity bootstrap work, why NFKC fixes the width failures, and what the synthetic data cannot tell you. Japanese-language claims should match your actual level, because the Japanese rules (i-adjective negation, です/ございます/と存じます rewrites) are simple template rules and will be probed in an interview. If your Japanese is limited, say that the grammar rules were deliberately kept to regular forms and have not been reviewed by a native speaker.

**Demo GIF:** none is included. I could not record a real dashboard session in the build environment (no browser, and `pyarrow` could not be installed), and I did not want to ship a mock-up. To add one, run `streamlit run bilingualprobe/dashboard.py`, record the three tabs with a screen recorder, save it as `docs/demo.gif`, and embed it here with `![dashboard demo](docs/demo.gif)`.

## Key design decisions

- **Sentiment only.** The spec says "sentiment/intent"; I implemented binary sentiment because negation then has a well-defined label flip.
- **Failure rate is conditional**: P(wrong on perturbed | right on clean), with the expected label flipped for negation. This keeps baseline errors from being counted as perturbation damage. Items where a perturbation does not apply (e.g. politeness on English, or on already-plain Japanese) are excluded and counts are reported.
- **Paired bootstrap** for parity, because EN/JA rows are translations of each other. For unpaired public data (or failure rates, whose denominators differ per language) it falls back to independent resampling; the result records which was used.
- **Typo rate** is a fixed 8% of characters per language, not per word, so the two scripts get comparable noise.
- **Held-out synonyms** are never used in training, so the synonym swap measures lexical generalisation.
- **Offline first.** Embeddings are opt-in and any failure (no package, no network) falls back to TF-IDF with a log warning. No secrets are read anywhere.
- **Synthetic data caveat.** Templates pair nouns and adjectives freely ("the software is comfortable"), so sentences can be semantically odd. The data is a controlled probe, not a realistic corpus; accuracy ceilings and gaps reflect template structure. Use `--public-data` for realistic text; lexicon-driven perturbations (negation, synonym, politeness) then mostly will not apply, and typo/width still do.
- **Neural backend.** The PyTorch path (2-layer MLP, Adam, 40 epochs) is used automatically when `torch` is importable; the MLPClassifier fallback has the same shape. The reported `backend` field records which ran. In the environment where this was built the disk was too small to install PyTorch, so the PyTorch branch has not been executed there and results above come from the scikit-learn fallback. The test suite exercises the torch path automatically when torch is installed.
- **Dashboard verification.** The app was exercised with Streamlit's `AppTest` (all tabs, widget interactions, no exceptions) and started headless, where it answered `/_stcore/health` with `ok`. The low-disk build environment could not install `pyarrow` (a normal Streamlit dependency installed by `pip install -r requirements.txt`), so table rendering was stubbed in that check.
