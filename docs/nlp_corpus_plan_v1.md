# NLP Corpus Plan v1

Status date: 2026-11-01. This is a data-acquisition and budget plan. The
TinyStories seed is downloaded; the mixed corpus has not yet been assembled.

## 1. Budget rule

The parameter budget is fixed by:

```text
unique trainable parameters = total corpus tokens / 20
```

`total corpus tokens` means tokenizer-native non-padding tokens across train,
validation, and test. The count includes EOS, counts tied input/output
embeddings once, and includes typed adapters, hierarchy-flow heads, and all
language-decoder heads. The allowed error is `2%`.

The initial tiers are:

| Tier | Total tokens | Parameters | Context | Vocab |
|---|---:|---:|---:|---:|
| Pilot | 46M | 2.3M | 512 | 8,192 |
| Screen | 92M | 4.6M | 1,024 | 8,192 |
| Full | 230M | 11.5M | 1,024 | 16,384 |

These token counts are targets after filtering and exact/near deduplication.
Tokenizer fitting uses the train split only.

This rule supersedes treating pretrained GPT-2 small as the primary matched
backbone. GPT-2 remains an architectural reference and optional tokenizer
baseline, not a 124M-parameter model competing under this corpus budget.

## 2. Frozen-family target mixture

| Family | Share | Required content |
|---|---:|---|
| General language | 25% | TinyStories plus optional public-domain prose |
| Code | 35% | Python 12%, Rust 12%, C++ 11% |
| Model architecture | 20% | transformer/attention, flow/diffusion LMs, SSM/recurrent, MoE, tokenization, evaluation |
| Engineering acceleration | 20% | Triton 10%, Linux 10% |

Code sources combine documentation and source. Documentation carries language
semantics and API intent; source carries syntax and implementation structure.
The corpus does not treat a file as "Python/Rust/C++" merely because a code
fence mentions that language.

The model-architecture family uses papers, model cards, technical reports, and
documentation from traceable sources. Only text with an explicit license and
source revision enters the manifest.

The engineering family covers Triton language/compiler/tutorial material and
Linux documentation for scheduling, memory, I/O, tracing, eBPF, profiling, and
relevant kernel internals.

## 3. Source and license controls

Every record must carry:

```text
source_url, upstream_revision, retrieved_at, license, license_url,
content_sha256, split, language, family
```

Preferred licenses are CC0, MIT, Apache-2.0, BSD, and PSF. CC-BY, CC-BY-SA,
GPL, and LGPL sources may be used only after per-source review. Copyleft text is
kept in a separately indexed data partition and is not redistributed with the
Python package or model weights.

No model-quality run starts while any source has an unknown license.

## 4. Cleaning and split rules

1. Strip navigation and site chrome from documentation while preserving code
   fences, formulas, headings, and signatures.
2. Remove generated, minified, vendored, and duplicate files.
3. Apply exact whole-document deduplication, then deterministic 5-gram MinHash
   near-deduplication.
4. Quarantine secrets, credentials, personal contact data, and benchmark
   contamination before tokenization.
5. Split by source snapshot, repository, file, or document family. Never split
   lines randomly within a file.
6. Do not allow a repository or a near-duplicate document family to cross
   train/validation/test.

The TinyStories seed already has exact cross-split leakage checks. Those checks
must be repeated after adding every new family.

## 5. Curriculum

The intended order is:

```text
general language
    -> code + model architecture
    -> engineering acceleration
    -> frozen mixed replay
```

The replay phase restores the target mixture and is mandatory. Otherwise gains
on Triton/Linux or code may be accompanied by language-forgetting that is hidden
by the final training batch.

## 6. Gates

- Every token traces to a hashed source and an explicit license.
- No exact or near-duplicate content crosses splits.
- The tokenizer is train-only and hash-frozen.
- Code covers Python, Rust, and C++ at the declared shares.
- Model-architecture and engineering families exist at the declared shares.
- Every tier obeys the 1/20 token-to-parameter rule within `2%`.
- Benchmark contamination is removed before training.
- A mixed-corpus quality claim requires the matched control comparison; corpus
  preparation alone is not evidence.

## 7. Current seed

The downloaded seed is:

```text
.data/nlp/tinystories_small_v1/
```

It contains approximately 123.7 MB of JSONL:

- train: 109,894 rows, 97.1M characters;
- validation: 11,058 rows, 9.64M characters;
- test: 10,932 rows, 9.43M characters.

The report is
`outputs/reports/nlp_dataset_download_v1.json`. It is a general-language seed
only. Token counts and the mixed-family corpus remain to be materialized after
the tokenizer is frozen.

Audit command:

```bash
uv run python scripts/audit_nlp_corpus_plan.py \
  --output outputs/reports/nlp_corpus_plan_audit_v1.json
uv run pytest tests/test_nlp_corpus_plan.py -q
```
