---
license: apache-2.0
language: [multilingual]
library_name: onnx
pipeline_tag: text-classification
tags: [country-codes, iso-3166, entity-normalization, selective-prediction, calibration, abstention, kyb, onnx]
base_model: jhu-clsp/mmBERT-small
---

# CountryCode-Decide (small, v1.2.1)

Maps a country or US state written as free text — any language or script, misspellings,
abbreviations, historical names, register junk — to ISO 3166-1 alpha-2 / ISO 3166-2:US codes with
calibrated probabilities, and **abstains** when the requested error rate cannot be met.
75 MB, about 4 ms per string on a CPU.

Code, evaluation and documentation: https://github.com/aliildan/country-code-decide · [Paper](https://doi.org/10.5281/zenodo.23289343)

## Intended use

Normalising country / jurisdiction / nationality fields in company-register data, KYB onboarding
and similar back-office pipelines, with a person reviewing abstentions and candidate lists.

**Out of scope:** full postal addresses (the model reads one field), subdivisions other than US
states, legal advice, any decision about a person without human review.

## How to use

Serving needs only `onnxruntime`, `tokenizers` and `numpy` (no PyTorch). The decision logic
(calibration, thresholds, candidate lists) lives in the repository's `CountryCoder`:

```bash
git clone https://github.com/aliildan/country-code-decide && cd country-code-decide
uv sync --no-default-groups --group serve && source .venv/bin/activate
uvx --from huggingface_hub hf download aildan/country-code-decide-small --exclude model.onnx \
  --local-dir models/country-code-decide-small
```

```python
from countrycode.infer.pipeline import CountryCoder

coder = CountryCoder.load("models/country-code-decide-small")
d = coder.code("Untted Klngdom", field="country", domain="register", alpha=0.05)
d.target, d.candidates, d.abstain     # ('GB', ('GB',), False); d.confidence ≈ 0.95
```

Input: `text`, optional `field` (`country`, `jurisdiction`, `nationality`, `unknown`) and
`register_country` (ISO alpha-2 of the register, or `US-XX` for a US state register). Output per
string: `target` (alpha-2, `US-XX`, `NONE` or `None` = abstain), `country`, `us_state`,
`confidence`, `candidates` (codes with calibrated probability ≥ 0.05, at most three), `abstain`,
for an error-rate target `alpha` (certified targets — curated: 5 %; register: 5 %; `None` always answers). Pass
`domain="register"` for register and form fields: the certified register calibration applies only
then; without a domain the stricter of the two policies applies, which is conservative but not
itself certified.

## Files

| File | Content |
|---|---|
| `model.int8.onnx` | the model: encoder + two heads, 8-bit weights (served; the CPU runtime computes its matrix products with int8 activations) |
| `model.onnx` | the same model in fp32 (reference) |
| `tokenizer.json`, `tokenizer_config.json` | the trimmed mmBERT tokenizer |
| `calibration.json` | temperatures, certified thresholds and candidate rule per domain (`register`, `curated`) |
| `meta.json` | label space, code names, context format, training run, quality-gate status |

## Training data

The model was trained on three kinds of strings. **About a third of the training examples are
generated (synthetic)**; no generated string is used for calibration or evaluation.

- **Lexicon** — 372,683 names in 1,274 languages (CLDR 47, Wikidata, codes, reviewed overrides); 86,406 distinct keys, 1,947 kept as genuinely ambiguous.
- **Real register strings** of the training split (Colorado, Connecticut, UK Companies House,
  Brønnøysund), labelled by a person or by the register's own code; no normalised key shared with
  any evaluation part.
- **Generated variants**, one release per generator, each made by code from a lexicon name with a
  known code (so the label is known by construction):
  - `keyboard` — typos on real keyboard layouts chosen by the name's language, with the number and
    kind of edits fitted to real register typos;
  - `format` — case, spacing, comma inversion, postcode noise, field truncation, abbreviation
    prefixes (labelled with every code whose name starts that way);
  - `translit` — romanisations and umlaut conventions;
  - `geo` — a country or US state written with one of its GeoNames regions or cities;
  - `pairs` — two countries in one string (labelled as the set of both);
  - `affix` — a country with a compass or region qualifier;
  - `caprov` — Canadian province codes in a US register's jurisdiction field (→ `CA`);
  - `junk`, `regjunk` — strings that are not places: placeholders, dates, numbers, generated
    company names, legal terms, street lines (→ `NONE`).

  Every release the model uses passed the quality gates before training: **G1** label correctness (each string
  checked against every name of every code; strings closer to another code are dropped) plus a
  human audit of 300 random items per release (every item when a release is smaller) with zero
  label errors required; **G2** realism (edit statistics compared with real typos — reported, no
  pass/fail threshold); **G3**
  usefulness (kept only if it improved held-out real strings; the OCR, LLM-paraphrase and
  region generators failed this and are not used); **G4** no overlap with any evaluation string; **G5** reproducibility
  (code, config, seed and source item recorded per string).

| generator | training examples | human audit (errors / audited) |
|---|---|---|
| junk@2 | 1,375 | 0 / 300 |
| keyboard@2 | 9,812 | 0 / 300 |
| format@4 | 13,411 | 0 / 300 |
| translit@4 | 4,396 | 0 / 300 |
| geo@2 | 7,381 | 0 / 300 |
| pairs@3 | 3,988 | 0 / 300 |
| affix@2 | 785 | 0 / 300 |
| caprov@1 | 78 | 0 / 39 |
| regjunk@1 | 2,731 | 0 / 300 |

- 142,925 training examples: 98,517 lexicon names and codes (68.9 %), 43,957 generated (30.8 %), 451 real register strings (0.3 %).
- Real register strings are labelled by a person or by the register's own code; none shares a normalised key with any evaluation part.

- No personal data: register sources were read with aggregate queries (distinct value and count),
  never records; generated company names come from word lists.

## Evaluation

Fresh registers never used for training, calibration or model selection (New York, Oregon): every
distinct string of their jurisdiction and country fields, labelled by exact lexicon match or by a
human reviewer, read once after a frozen pre-registration (release 1.2). Release 1.2.1 keeps the
weights and corrects the certificate after an external review, so its numbers below are post-hoc.

| Fresh registers, 361 distinct strings | CountryCode-Decide small v1.2.1 |
|---|---|
| Accuracy, every string answered | 88.6 % (95 % CI 85.3–91.7) |
| Right country (a US state counts as US) | 91.2 % |
| At the certified 5 % error target | answers 88 % automatically, realised error 2.5 % |
| Best non-LLM baseline (exact lexicon + context rules) | 89.5 % |
| Zero-shot LLM (qwen3.5_35b-a3b-q4_K_M) | 85.0 %, ~29× slower (LLM on a GPU, this model on a CPU) |
| Model file · CPU latency | 75 MB (int8 ONNX) · p50 4.1 ms per string |

| Predictor | automatic labels (exact lexicon match, n = 261) | hand-labelled (n = 100) | hand-labelled answered |
|---|---|---|---|
| CountryCode-Decide (always answers) | 99.2 % | 61.0 % | 100 |
| CountryCode-Decide at the certified 5 % target | 98.1 % | 56.0 % | 61 |
| Zero-shot LLM (qwen3.5_35b-a3b-q4_K_M) | 94.3 % | 61.0 % | 97 |
| Lexicon + context rules | 99.6 % | 63.0 % | 62 |
| multilingual-e5-small nearest name | 100.0 % | 37.0 % | 100 |
| exact lexicon match | 100.0 % | 24.0 % | 23 |

Automatic labels come from an exact lexicon match, so a lexicon is right on them by construction; on
the hand-labelled strings the model, the LLM and an exact lexicon with the training context rules are
about level. On held-out strings of the training registers (656 strings, many misspelled or abbreviated; other strings of the same registers were in training) the model is right on 82.6 %, the best other predictor (qwen3.5_35b-a3b-q4_K_M zero-shot) on 73.8 %, the lexicon with context rules on 35.8 %; on the 472 of them with no exact lexicon match 81.1 % against 73.1 %. On GBIF's curated list the LLM is better: 69.6 % against 58.8 %.

- Country-level accuracy (right country; a US state counts as US): 0.912 (highest: v1_model 0.921).
- On the 33 strings not in the lexicon (no exact name match) the model's accuracy is 0.212.
- At the certified 5 % error target the model answers 88% of the strings automatically with a realised error of 2.5 %; non-places get a code 0.0 % of the time (n = 4). The 5 % target is certified over 318 register units (one per register and normalised key); 1 % and 2 % are not.
- The zero-shot LLM (qwen3.5_35b-a3b-q4_K_M) is less accurate when every string must be answered (0.850 vs the model's 0.886); its own confidence covers 71% of the strings at 5 % empirical risk, without a certificate. It is ~29× slower (the LLM on a GPU, this model on a CPU).

The guarantee is an average over strings like the calibration strings, and the errors concentrate on hard ones: at the 5 % target 13.1 % of the 61 answered hand-labelled strings are wrong, 0.0 % of the 256 answered strings that an exact lexicon match labelled.

**The certificate, corrected.** The pre-registered certificate treated the 592 register calibration strings as independent draws and certified 2 % and 5 %. The same string often appears in several fields of a register, so they are not independent: they form 318 register-and-key units. Counted per unit — a correction made after the test read, prompted by an external review — 2 % is not certified, and 5 % is, at λ = 0.762. This release (small-v1.2.1) ships that calibration; the weights are unchanged.

| predictor | n | acc | acc (freq-weighted) | country acc | coverage @1 % risk | coverage @5 % risk | non-place false accept | ECE | ms / item |
|---|---|---|---|---|---|---|---|---|---|
| lexicon_context | 361 | 0.895 | 1.000 | 0.901 | 0.000 | 0.895 | 0.250 | 0.015 | 0.0 |
| model | 361 | 0.886 | 1.000 | 0.912 | 0.856 | 0.911 | 0.750 | 0.058 | 4.2 |
| v1_model | 361 | 0.884 | 1.000 | 0.921 | 0.197 | 0.864 | 1.000 | 0.253 | 3.9 |
| model@0.05 | 361 | 0.864 | 1.000 | 0.873 | 0.856 | 0.878 | 0.000 | 0.033 | 4.2 |
| llm_qwen3.5_35b-a3b-q4_K_M | 361 | 0.850 | 0.997 | 0.858 | 0.000 | 0.709 | 0.000 | 0.108 | cached |
| e5_knn | 361 | 0.825 | 0.995 | 0.847 | 0.091 | 0.825 | 0.750 | 0.108 | 0.6 |
| lexicon_fuzzy | 361 | 0.795 | 0.994 | 0.802 | 0.000 | 0.820 | 0.500 | 0.054 | 1.0 |
| lexicon_exact | 361 | 0.789 | 0.994 | 0.793 | 0.000 | 0.787 | 0.250 | 0.014 | 0.0 |
| ons_style | 361 | 0.787 | 0.994 | 0.793 | 0.000 | 0.792 | 0.500 | 0.044 | 0.3 |
| countrynames | 361 | 0.634 | 0.761 | 0.745 | 0.000 | 0.000 | 0.000 | 0.226 | 2.7 |
| pycountry | 361 | 0.634 | 0.761 | 0.776 | 0.000 | 0.000 | 0.250 | 0.184 | 1.5 |
| coco | 361 | 0.596 | 0.761 | 0.601 | 0.000 | 0.000 | 0.250 | 0.120 | 0.5 |
| model@0.01 | 361 | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | 4.1 |
| model@0.02 | 361 | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | 4.1 |

Pre-registered hypotheses of release 1.2, as read (P1 and P2 rest on the item-level certificate
corrected above):

| hypothesis | result | evidence |
|---|---|---|
| P1 | met | certified=True, lambda=0.827, answered_n=309, errors=3, p_value_risk_le_2pct=0.947, contradicted=False |
| P2 | met | answered=0.856 |
| P3 | met | size_mb=74.732, cpu_p50_ms=4.092, tokens_identical=1.000, select_accuracy_drop=0.003 |
| P4 | met | model_accuracy=0.886, best_non_llm=e5_knn, best_non_llm_accuracy=0.825 |
| A1 | reported (not judged) | certified=False, answered=0.000, risk_answered=nan |

Gold labels: The register calibration strings were labelled a second time, blind (no proposal and no first label shown): agreement 99.2 % on 592 strings (5 disagreements); one-sided 95 % upper bound on disagreement 1.8 %. On the 218 hand-labelled strings the passes disagree on 5 (2.3 %); the 374 automatic or preset labels agree by construction. Test labels were labelled once.

Latency and size: p50 4.1 ms, p95 5.1 ms per string on CPU (4 threads, ONNX Runtime, tokenisation included); model file 75 MB.

## Limitations and risks

- For clean fields an exact lexicon with context rules is as accurate as this model; use the model
  for what a lexicon cannot read. Its advantage was measured on held-out strings of registers whose
  other strings were in training; on fresh registers, strings with no exact lexicon match are hard.
- The certificate holds for strings drawn like the calibration strings (register fields), counted
  per register and normalised key, each target on its own (with probability ≥ 90 % over the
  calibration draw); errors concentrate on hard strings. Recalibrate before relying on it for a
  different source. The curated GBIF domain is much harder than registers for this model, and there
  a 35B LLM is more accurate.
- One human reviewer labelled the register strings; the calibration strings were labelled twice by
  that reviewer, the second time blind (consistency, not inter-annotator agreement); test labels
  were labelled once.
- Of the 4,997 evaluation strings, 5 are not in Latin script, so accuracy on other scripts is not measured.
- One training seed (13); the usefulness gate compared runs whose spread is about its margin.
- Ambiguous names are returned as candidate lists; downstream systems must not silently pick one.
- Known gaps: bare Canadian province codes in US jurisdiction fields (`ON`, `BC`) are not recognised
  (every real such string is an evaluation string, so the no-overlap gate removed them from the
  generated province variants); `Congo` without context is answered as `CG`.
- Not legal advice; keep a person in the loop.

## Citation

```bibtex
@misc{ildan2026countrycode,
  title = {From Messy Country Names to ISO 3166 Codes: A Small Model That Abstains, and When a Lexicon Is Enough},
  author = {Ildan, Ali},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23289343}
}

@software{ildan2026countrycode_software,
  title = {CountryCode-Decide: messy country names to ISO 3166 codes, with calibrated abstention},
  author = {Ildan, Ali},
  year = {2026},
  url = {https://github.com/aliildan/country-code-decide},
  doi = {10.5281/zenodo.23289336}
}
```

## Attribution

Unicode CLDR (Unicode License v3) · Wikidata (CC0) · GeoNames (CC BY 4.0) · GBIF parsers
(Apache-2.0) · Colorado, Connecticut and Oregon open data (public domain) · UK Companies House
(OGL v3) · Brønnøysund Register Centre (NLOD) · New York open data (OPEN-NY terms, evaluation only)
· mmBERT (MIT).
