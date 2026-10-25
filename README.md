# fairytale-nlp

[![CI](https://github.com/tc020/fairytale-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/tc020/fairytale-nlp/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/tc020/fairytale-nlp/branch/main/graph/badge.svg)](https://codecov.io/gh/tc020/fairytale-nlp)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/github/license/tc020/fairytale-nlp)](LICENSE)
[![Release](https://img.shields.io/github/v/release/tc020/fairytale-nlp?include_prereleases&sort=semver)](https://github.com/tc020/fairytale-nlp/releases)
[![Container](https://img.shields.io/badge/container-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/tc020/fairytale-nlp/pkgs/container/fairytale-nlp)
[![Built with spaCy](https://img.shields.io/badge/built%20with-spaCy-09a3d5)](https://spacy.io)
<!-- After the first release archived on Zenodo, add:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

**fairytale-nlp** extracts the characters (e. g. snow white or the stepmother) of German fairy tales and the adjectives that describe them. Every single assignment stays traceable to its source sentence, so results can be checked, cited and corrected.

The question behind the project: *Which adjectives characterise which characters, and does this differ between collections (Grimm, Andersen, translated tales)?*

## Example

Input (from *Schneewittchen*):

> "Zu dem Fest wurde aber auch Sneewittchens gottlose Stiefmutter eingeladen."

One assignment row in the output (field names are German, as is the whole code base) for every adjective:

```json
{
  "adjektiv": "gottlose",
  "lemma": "gottlos",
  "maerchen": "Korpus/Brüder Grimm/Sneewittchen.txt",
  "sammlung": "Brüder Grimm",
  "satz_id": 150,
  "methode": "original",
  "numerus": "Sing",
  "kollektiv": false,
  "coref_sicher": null,
  "coref_belegklasse": null,
  "bezug_klasse": "attributiv",
  "bezug_sicher": true,
  "lemma_norm": "gottlos"
}
```

Aggregated per character and collection, the same data becomes a profile:

```json
"Stiefmutter": {
  "adjektive_haeufigkeit": [{"lemma": "bös", "anzahl": 3}, {"lemma": "rein", "anzahl": 1}, {"lemma": "bleich", "anzahl": 1}, {"lemma": "heftig", "anzahl": 1}, {"lemma": "gottlos", "anzahl": 1}],
  "adjektive": ["bleich", "bös", "gottlos", "heftig", "rein"],
  "gesamt_nennungen": 7,
  "anzahl_maerchen": 5,
  "maerchen": ["Korpus/Brüder Grimm/Aschenputtel.txt", "Korpus/Brüder Grimm/Brüderchen und Schwesterchen.txt", "Korpus/Brüder Grimm/Frau Holle.txt", "Korpus/Brüder Grimm/Schneewittchen.txt", "Korpus/H. C. Andersen/Die wilden Schwäne.txt"],
  "vorkommen": [>>it's the data structure from above for every adjective<<],
  "numerus": {"Sing": 7},
  "nennungen_kollektiv": 0,
  "bezug_klassen": {"objektspraedikativ": 1, "praedikativ": 1, "attributiv": 4, "artangabe": 1},
  "nennungen_bezug_unsicher": 1,
  "adjektive_norm_haeufigkeit": [{"normform": "bös", "anzahl": 3}, {"normform": "rein", "anzahl": 1}, {"normform": "bleich", "anzahl": 1}, {"normform": "heftig", "anzahl": 1}, {"normform": "gottlos", "anzahl": 1}],
   "adjektive_norm": ["bleich", "bös", "gottlos", "heftig", "rein"]
}
```

Each entry in `vorkommen` points back to the tale, the sentence id and the method (`original` or `coref`) that produced it.

## Quick start

```bash
# 1. Install the package and the base language model
pip install git+https://github.com/tc020/fairytale-nlp.git
python -m spacy download de_core_news_lg

# 2. Install the fine-tuned character model (FIGUR label) from the release assets
pip install https://github.com/tc020/fairytale-nlp/releases/download/v0.1.0/de_fairytale-nlp_ner-0.1.0-py3-none-any.whl

# 3. Run the pipeline on a corpus folder
fairytale-nlp run data/raw/ --out results/
```

The corpus folder is scanned recursively: `data/raw/<collection>/<title>_text.txt`. The first sub-folder is the collection, the file name without `_text` is the title. Raw Gutenberg-style text (hard line breaks, hyphenation) is fine; normalisation is the first pipeline step.

With Docker, no local Python setup is needed:

```bash
docker run --rm -v "$PWD/data:/data" ghcr.io/tc020/fairytale-nlp:latest run /data/raw --out /data/results
```

Output: `results/ergebnisse_alle.json` (corpus overview, per-collection and corpus-wide profiles, all assignment rows with provenance, raw results per tale) and `results/adjektive_fuer_clustering.{txt,json}` for downstream analysis.

## How it works

Each tale passes through nine steps; the batch loop then aggregates across tales.

```
raw text
  │ 1  normalisation (line breaks, hyphenation)
  │ 2  NER with label FIGUR  →  character index with three-stage canonicalisation
  │      (manual aliases · lemma merging of inflected forms · multi-word names)
  │ 3  coreference resolution over the whole text (sliding-window module)
  │ 4  sentence segmentation with ids (quote-aware boundaries, span repair)
  │ 5  sentence mapping original ↔ coreference text (1:1 or difflib alignment)
  │ 6  POS filter: keep sentences containing adjectives (STTS ADJA / ADJD)
  │ 7  dependency parsing: adjective → character
  │      (attributive, predicative/adverbial, object-predicative, relative clause)
  │ 8  open cases are kept and labelled (no head word, pronoun, head is not a character)
  │ 9  pronoun resolution via the sentence mapping, second pass (methode = "coref")
  ▼
assignment rows  →  profiles per tale, per collection, corpus-wide  →  JSON export
```

Two design decisions shape everything:

- **Analyse the original text, not the resolved one.** POS tagging and parsing run on the original sentences, where annotation quality is highest. The coreference-resolved text is only a lookup table for pronouns. Every row records which of the two produced it.
- **Aliases only for unambiguous cases.** The alias table merges surface forms that always mean the same character (`Mann`/`Manne` → `Vater`). Context-dependent cases (`Frau` as mother in one tale, stepmother in another) are left to coreference resolution.

The run is fault-tolerant: a failing tale is logged in the corpus overview and does not stop the batch. If coreference resolution fails, the tale is processed without it (`coref_ok = false`).

Two spaCy models are involved: `de_core_news_lg` for lemmatisation, POS tagging and dependency parsing (TIGER dependency scheme), and a fine-tuned NER model with the single label `FIGUR` that also recognises role nouns such as *Stiefmutter* or *Prinz*, which standard NER (PER) does not.

## Model

| | |
|---|---|
| Base | `de_core_news_lg` (tok2vec reused), NER component re-initialised with label `FIGUR` |
| Training data | 14 annotated tales (Grimm, Andersen, Arabian Nights and Japanese tales in German translation), 1,447 sentences, 1,884 FIGUR entities |
| Labels | NER: manual gold annotation. POS: silver labels from a majority vote of spaCy, Stanza and Flair |
| Split | 80 / 20 at sentence level (1,157 / 290 sentences) |
| Result (dev set) | F1 90.5 · precision 88.0 · recall 93.0 (best checkpoint) |

The dev set was used for model selection, so these numbers are optimistic, and a sentence-level split lets sentences of the same tale appear on both sides. A held-out evaluation on unseen tales, a baseline comparison and a transformer variant are the next milestones (see [Roadmap](#roadmap)). Training scripts and the annotation guideline live in [`training/`](training/) and [`data/ANNOTATION.md`](data/ANNOTATION.md).

## Data

The corpus consists of public-domain German fairy tales: the Grimm *Kinder- und Hausmärchen* (edition of 1857), tales by Hans Christian Andersen in a 19th-century German translation, and German translations of tales from the *Arabian Nights* and from Japan. Sources are Project Gutenberg and zeno.org.

Provenance, text basis, translator and licence status are documented **per text** in [`data/README.md`](data/README.md), together with SHA-256 checksums of the raw files. Raw texts are kept unchanged in `data/raw/`; everything derived (CoNLL files, `.spacy` binaries) lives in `data/processed/`.

## Project structure

```
fairytale-nlp/
├── src/fairytale-nlp/      # package: corpus discovery, normalisation, ner, coref,
│                           #          sentences, assignment, aggregation, export, cli
├── training/               # project.yml, config.cfg, data preparation and evaluation
├── data/
│   ├── raw/<collection>/   # source texts, unchanged
│   ├── processed/          # CoNLL and .spacy files
│   ├── README.md           # provenance and licences per text
│   └── ANNOTATION.md       # labelling guideline for FIGUR
├── tests/                  # unit, golden and end-to-end tests
├── .github/workflows/      # ci.yml (lint, tests), release.yml (image build and push)
├── Dockerfile
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Development

```bash
git clone https://github.com/tc020/fairytale-nlp.git
cd fairytale-nlp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

make test      # pytest with coverage
make lint      # ruff check + ruff format --check
make image     # build the container image locally
```

Tests are split into fast unit tests for the pure Python logic (CoNLL parsing, BIO span conversion, canonicalisation, aggregation), golden tests on constructed sentences with known expected assignments, and one end-to-end run on a mini corpus (marked `slow`, skipped unless the models are installed). CI runs lint and tests on Python 3.11 and 3.12 for every push and pull request; tagged commits build the image and push it to the GitHub Container Registry.

## Known limitations

- German only; adjectives only (`ADJA`, `ADJD`). Other characterising devices (verbs, metaphors, direct speech) are out of scope.
- Coreference resolution is a sliding-window heuristic, not a trained model; its errors propagate into rows marked `methode = "coref"`, which is why the method is recorded.
- The alias table is manual and corpus-specific.
- Model quality is only measured on the dev set so far (see [Model](#model)).

## Roadmap

Progress is tracked in [milestones](https://github.com/tc020/fairytale-nlp/milestones) and [issues](https://github.com/tc020/fairytale-nlp/issues?q=is%3Aissue+label%3Aroadmap).

- [Reproducible training](https://github.com/tc020/fairytale-nlp/milestone/2): split by tale, held-out test set, baseline (`de_core_news_lg` PER) vs. fine-tuned model, learning curve, several seeds.
- [Transformer variant](https://github.com/tc020/fairytale-nlp/milestone/3): token classification with Hugging Face Transformers / PyTorch on the same test set.
- [REST API](https://github.com/tc020/fairytale-nlp/milestone/4): FastAPI service for `/predict` and `/profile`.
- [Deployment and monitoring](https://github.com/tc020/fairytale-nlp/milestone/5): container on Azure Container Apps, MLflow model registry, latency and drift monitoring.

## Licence

- **Code**: [MIT](LICENSE).
- **Texts** in `data/raw/`: public domain (authors and translators died more than 70 years ago); see [`data/README.md`](data/README.md) for the status of each text. Public-domain texts are not re-licensed.
- **Annotations** (CoNLL files in `data/processed/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Model weights**: MIT, derived from `de_core_news_lg` (MIT) and the annotations above.

## Citation

If you use this software or the annotated data, please cite it. Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it under *Cite this repository*.

## About

Developed by <Your Name> as the final project of the M.Sc. Information Science at Hochschule Darmstadt (2026). Built with [spaCy](https://spacy.io); POS silver labels with [Stanza](https://stanfordnlp.github.io/stanza/) and [Flair](https://github.com/flairNLP/flair). Texts from [Project Gutenberg](https://www.gutenberg.org) and [zeno.org](http://www.zeno.org).

Questions and suggestions: please [open an issue](https://github.com/tc020/fairytale-nlp/issues/new).
