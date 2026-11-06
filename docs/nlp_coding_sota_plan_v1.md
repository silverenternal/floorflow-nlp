# Small Coding Model: Target and Claim Ladder

Status date: 2026-11-01. This document defines the task target. It contains no
capability or SOTA evidence.

## 1. Defensible target

The initial phrase "the world's smallest SOTA-level coding model with
engineering ability" is too broad to test directly. It contains three different
claims:

1. the model is useful for coding;
2. it has executable engineering ability;
3. it is the smallest model at a declared SOTA level.

The working target is therefore:

> The smallest reproducible model that enters a bounded SOTA band on coding
> and engineering tracks, optimized for task success per parameter, FLOP, and
> training token under a complete evaluation protocol.

A literal world-SOTA claim is permitted only after C4, including a frozen
leaderboard snapshot, all benchmark versions, all reproducible baselines, three
training seeds, and an independent rerun.

## 2. Why not start with full training

Direct training would confound architecture, tokenizer, data mixture, task
format, evaluation harness, and scale. The first full run would consume compute
without telling us which mechanism caused a result.

The order is:

```text
benchmark/executor freeze
    -> tokenizer and data contract
    -> P2M training-stack smoke
    -> P5M/P12M matched architecture screen
    -> task specialization at surviving scale
    -> P23M only after a measured failure; P46M diagnostic only
    -> SOTA certification
```

Short training starts early, but only as a smoke or matched screen. Full
pretraining does not start until the control, tokenizer, mixture, and evaluator
are frozen.

## 3. Capability tasks

The target suite has six tracks, centered on AST-first structure:

| Track | Weight | Primary check |
|---|---:|---|
| Python code generation | 15% | HumanEval+/MBPP+ pass@1 |
| Multilingual code | 15% | MultiPL-E Python/Rust/C++ pass@1 |
| AST structure | 20% | node/edge F1, parse validity, tree edit distance |
| Repository patch | 15% | executable all-tests-pass rate |
| Engineering acceleration | 25% | Triton/Linux/performance verifier success |
| Language retention | 10% | held-out NLL regression |

Benchmark availability and versions must be audited before freezing. The custom
engineering suite will require deterministic executors rather than keyword
scoring.

Engineering ability means:

- multi-file Python repair with tests;
- Rust or C++ compile-and-test repair;
- Triton kernels checked by numerical equivalence;
- Linux process/memory/I/O diagnosis with expected actions;
- performance patches that retain correctness and meet a latency guard.

Primary generation uses temperature zero, one sample, and no best-of-N.

## 4. Parameter ladder

Every tier follows `corpus_tokens = 20 * unique_trainable_parameters`:

| Tier | Parameters | Corpus tokens | Role |
|---|---:|---:|---|
| P2M | 2.3M | 46M | contract/tokenizer smoke |
| P5M | 4.6M | 92M | first matched architecture screen |
| P12M | 11.5M | 230M | minimum credible coding candidate |
| P23M | 23M | 460M | maximum primary claim tier |
| P46M | 46M | 920M | diagnostic only; never a headline |

The current 11.5M full tier is not assumed to be sufficient for coding. It is
the lower capability screen. Scaling beyond it requires a measured capability
gain, not an attempt to rescue a flat P5M/P12M curve.

Parameter count includes embeddings, tied output projection, typed adapters,
flow/hierarchy heads, and decoder heads. Tied weights count once.

## 4.1 Anti-scaling contract

This line does not compete with GPT-class models through parameter or data
stacking.

- The primary claim cap is `23M` unique trainable parameters.
- `46M` is the absolute experiment cap and a non-claim diagnostic.
- Generated experts reallocate capacity inside the declared tier.
- A birth requires retirement, merge, or explicit preallocated headroom.
- External models above the cap are out-of-class references only.
- Generic Transformer layer growth is not a valid response to a failed screen.
- Ensembling and unbounded test-time expert creation are excluded from the
  primary endpoint.

Every result reports task success per unique parameter, per inference FLOP, per
training token, and per solved-task wall-clock time.

## 5. Specialization, not generic instruction tuning

The task-specific components are:

1. a train-only code-aware tokenizer audited for indentation, punctuation,
   identifiers, and code fences;
2. fill-in-the-middle and diff objectives;
3. typed syntax/symbol/discourse/execution/engineering memory;
4. document/segment/token/entity hierarchy;
5. compiler, unit-test, numerical, and shell-execution feedback;
6. rejection or preference learning only after supervised tests pass.

The hierarchy is retained because it is the FloorFlow mechanism under test. The
project does not add a generic mixture-of-experts or a large external model to
hide a failed architecture transfer.

## 6. Claim ladder

| Level | Claim |
|---|---|
| C0 | Infrastructure, leakage controls, and executable evaluators work |
| C1 | Candidate beats a matched from-scratch control on at least one primary track |
| C2 | Candidate is non-dominated on the parameter-capability frontier |
| C3 | Candidate enters the frozen SOTA band on every primary track |
| C4 | Candidate is the smallest reproducible SOTA-band model at the snapshot |

No claim above C0 is available now. `P12M` is a capability screen, not a SOTA
announcement.

## 7. Immediate execution decision

Do not launch full pretraining yet. The next work is:

1. benchmark availability and contamination audit;
2. repository-patch, Triton, Linux, and performance test fixtures;
3. train-only tokenizer;
4. P2M unchanged-control smoke;
5. matched P5M/P12M scaling screen.

The first meaningful model-selection decision is whether P5M/P12M improves on
the matched control. If it does not, task-specific SFT is unlikely to repair a
broken backbone mechanism.

Audit command:

```bash
uv run python scripts/audit_nlp_coding_target.py \
  --output outputs/reports/nlp_coding_target_audit_v1.json
uv run pytest tests/test_nlp_coding_target.py -q
```
