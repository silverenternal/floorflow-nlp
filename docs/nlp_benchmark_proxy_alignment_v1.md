# Benchmark Proxy Alignment v1

## Question

Do our internal training and validation metrics measure the same capability axes as
current coding and engineering LLM benchmarks? Any external metric without a
directionally aligned internal proxy, or proxy without a joint-loss path, is treated as
a missing capability component.

This document does not claim results on those benchmarks. It freezes the mapping and
the correlation experiment needed to validate the proxy.

## External Capability Axes

| Benchmark family | Main public metrics |
|---|---|
| HumanEval+/MBPP+ | pass@1 with extra tests |
| MultiPL-E Rust/C++ | pass@1, compilation, runtime |
| LiveCodeBench | generation, self-repair, execution, test-output prediction |
| BigCodeBench | pass@1, diverse library calls, branch coverage |
| SWE-bench | resolved, patch apply, fail-to-pass, pass-to-pass |
| RepoBench | retrieval recall/MRR, cross-file completion |
| CRUXEval | input/output execution prediction |
| ClassEval | class-level pass@1, method dependencies |
| DS-1000 | execution and surface constraints |
| IFEval | strict prompt-level and instruction-level compliance |
| LiveBench | contamination-limited objective accuracy across tracks |
| MMLU-Pro | multiple-choice accuracy, calibration, selective risk |
| GPQA | closed-book knowledge accuracy and calibration |
| GSM8K/MATH | final-answer exact match and correctness |
| TruthfulQA | truthfulness and informativeness |
| SafetyBench | attack success and refusal accuracy |
| RULER | long-context retrieval recall |

## Internal Proxy Map

- pass@1 → executable outcome loss plus final unit-test pass rate.
- compile/runtime success → language-specific compile and execution outcome heads.
- extra-test robustness → metamorphic consistency under test/spec perturbations.
- self-repair → diagnostic-conditioned patch success; no hidden inference retry.
- execution trace/test output → trace-step, trace-result, and output-prediction heads.
- BigCodeBench/DS-1000 → AST call-edge, library/API constraint, and execution losses.
- SWE-bench → file/span localization, patch apply, repository resolved, fail-to-pass,
  and pass-to-pass losses.
- RepoBench → query-candidate contrastive retrieval plus line-completion likelihood.
- CRUXEval → forward and inverse execution prediction heads.
- ClassEval → structural completeness and call/dependency AST losses.
- IFEval → independent verifiable-constraint BCE and exact prompt-level satisfaction.
- LiveBench calibration → task-success calibration and selective-risk diagnostics.
- contamination-limited coverage → data governance gate, not a differentiable loss.

## Expanded Network Heads

The benchmark proxy bank now adds direct differentiable surfaces for:

- self-repair success;
- line-completion semantics;
- class-level success;
- library/API-call correctness;
- branch coverage;
- forward and inverse execution prediction;
- token-level test-output prediction;
- runtime, memory, and latency ratios;
- knowledge, answer, and multiple-choice correctness;
- selective correctness and calibration;
- truthfulness, informativeness, and safety compliance;
- long-context retrieval through the retrieval projector.

These heads are available in the model forward result. Dataset-specific labels still
need to be wired before each benchmark axis becomes training evidence.

## Identified Missing Joint-Loss Paths

The original token CE plus AST/JSON/emergence objectives did not directly train:

- compile and runtime outcome predictors;
- execution trace and test-output prediction;
- repository retrieval and cross-file context selection;
- file/span localization and patch application;
- fail-to-pass / pass-to-pass repository outcomes;
- verifiable instruction constraints and strict prompt-level satisfaction;
- metamorphic consistency under semantics-preserving test perturbations;
- confidence calibration and selective risk.

`src/floorflow/nlp_flow/model/benchmark_proxy.py` supplies the first modular heads and
losses for these paths. They still need dataset-specific wiring into the joint trainer
and matched ablations.

## Proxy Validation

Freeze at least 24 benchmark families/versions and at least two model scales. For each,
compute internal proxy and external metric under identical checkpoints and inference
budgets. Pass requires:

- Spearman rank correlation ≥ 0.70;
- pairwise sign agreement ≥ 0.75;
- no final-test label or test outcome used at inference;
- every non-governance external metric mapped to an internal proxy;
- every required proxy loss finite on deterministic fixtures.

Failure means the joint loss still lacks a capability axis, or the proxy is measuring a
different construct. A high proxy score alone is never reported as benchmark success.

## Sources

Metadata and abstracts were checked through the official arXiv API on 2026-11-02.

- EvalPlus: https://arxiv.org/abs/2305.01210
- MultiPL-E: https://arxiv.org/abs/2208.08227
- LiveCodeBench: https://arxiv.org/abs/2403.07974
- BigCodeBench: https://arxiv.org/abs/2406.15877
- SWE-bench: https://arxiv.org/abs/2310.06770
- RepoBench: https://arxiv.org/abs/2306.03091
- CRUXEval: https://arxiv.org/abs/2401.03065
- ClassEval: https://arxiv.org/abs/2308.01861
- DS-1000: https://arxiv.org/abs/2211.11501
- IFEval: https://arxiv.org/abs/2311.07911
- LiveBench: https://arxiv.org/abs/2406.19314
