# Data-Efficient Learning v1

## Goal

Improve information gained per training token on the existing P5M/P12M corpora while
preserving independent supervision for coding, JSON, AST, execution, engineering,
architecture, language-retention, and compositional gaps.

The open-capability track may temporarily use a larger model. Data efficiency remains
useful because it increases the quality ceiling before later tensor-ring or low-rank
compression.

## Mechanisms

### Gap- and family-balanced sampling

Each document receives deterministic gap tags from its family, language, structure, and
strict JSON parse status. The sampler tracks recent family and gap exposure and raises
the weight of underrepresented supervision. It does not sample by raw token volume
alone: Verilog or long engineering files must not crowd out Python execution or
language retention.

### Curriculum

The default schedule has four contiguous stages:

1. `grammar_foundation`: short language and code units;
2. `code_fluency`: Python, Rust, C++, and structural code;
3. `verifier_grounded`: execution, JSON, AST, and engineering constraints;
4. `composition_replay`: harder cross-gap examples with replay.

Progress is a declared training fraction, never a final-test score.

### Deduplication and exposure

Exact SHA-256 duplicates are rejected. Near duplicates use a 64-bit SimHash over
lower-cased token trigrams and a configurable Hamming threshold. A small exposure
penalty prevents repeated examples from monopolizing the sampler.

### Hard-example memory and replay

Per-example training losses can update a bounded deterministic memory. High-loss
examples receive a curriculum-stage bonus, while a fixed replay fraction protects
previously learned language and code behavior.

### Gap adapter reuse

`gap_training_examples` returns only rows supervised for the requested gap plus an
explicitly bounded auxiliary replay tail. Channel checkpoints therefore stay isolated
while still receiving stable gradient support.

`json_validity` and `compositional_emergence` may have no direct raw-corpus rows in a
bounded probe. Their supervision comes from the separate generated JSON-policy and
held-out composition data contracts; the probe reports this explicitly instead of
silently treating the gaps as covered.

## Matched arms

| Arm | Selection |
|---|---|
| U | uniform over admitted rows |
| F | family-target balanced |
| C | family + gap + curriculum difficulty + hard memory + replay |

All arms receive the same row pool, seed, example count, model, steps, and inference
budget. The CPU probe compares held-out byte-bigram NLL and data diagnostics only.
Neural loss, executable generation, and family forgetting require the GPU extension.

## Forbidden practices

- no final-test read during sampling or checkpoint selection;
- no repair, retry, or best-of-N;
- no claim that better sampling metrics equal coding SOTA;
- no cross-gap replay without marking it as auxiliary supervision.
