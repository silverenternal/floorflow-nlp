# NLP Task Breakdown v1

Status date: 2026-11-01.

`nlptodo.json` has three planning layers:

- **Task DAG**: a gate-bearing research task with dependencies and failure rules.
- **Work package**: one concrete execution unit with outputs and acceptance
  checks.
- **Decision point**: the stop/continue/scale decision after evidence is frozen.

The current graph contains 38 tasks and 79 work packages. The neuron-growth
branch adds explicit factory, meta-growth, and generated-versus-MoE
disentanglement tasks.

The core "model builds E" target starts at synthesis level S2: generated
rank/width and routing, not only fixed-shape weights or fixed-pool MoE routing.

The primary claim is capped at 23M unique parameters. Learned births reallocate
capacity inside the tier; P46M is diagnostic only. Results are always reported
per parameter, FLOP, training token, and solved-task wall time.

A task passes only when all its required work packages pass and its declared
gate passes. A work package does not become a paper result by itself.

## Current round

Round `NLP-ROUND-01` is CPU-only and has two parallel lanes.

### Lane A: evaluation

```text
WP-010-A benchmark/license registry                  [ready]
    -> WP-010-B pinned runners/contamination         [planned]
    -> WP-010-C metric attacks/SOTA-band contract    [planned]
    -> WP-011-A repository-patch executor            [blocked]
    -> WP-011-B Triton/Linux executors               [planned]
    -> WP-011-C performance/attack fixtures          [planned]
```

The first action is `WP-010-A`: produce the public benchmark availability and
license registry. It must identify official sources, versions, licenses, and
runner dependencies for:

- HumanEval+
- MBPP+
- MultiPL-E Python
- MultiPL-E Rust
- MultiPL-E C++

### Lane B: data

```text
WP-012-A license/revision source registry             [ready]
    -> WP-012-B Python/Rust/C++ code and docs         [planned]
    -> WP-012-C model/Triton/Linux docs               [planned]
    -> WP-012-D clean/dedup/split/token ledger        [planned]
    -> WP-013-A train-only tokenizer fit              [blocked]
    -> WP-013-B compression/syntax audit              [planned]
    -> WP-013-C tokenizer hash/budget freeze          [planned]
```

No corpus text enters the tokenizer before its source has a license decision
and a hash. The selection is not allowed to relax the 20:1 token/parameter rule
to fit the data that happened to be downloaded.

## Milestones

| Milestone | Scope | Current state |
|---|---|---|
| M0 Foundation | protocols, compact architecture contract, Rust macros, engineering substrate, seed data, corpus/target contracts, neuroscience scope | passed |
| M1 Eval + data | benchmarks, executors, mixed source corpus | ready |
| M2 Mechanism | tokenizer, strict JSON schema/head, AST codec, Stage-A AST CFM, Stage-B result CFM, five typed channels, semantic/typed/MiTA/slot attention, staged training, synthetic probe and generated-expert factory | blocked |
| M3 Scale | P2M smoke, P5M/P12M screens, two-stage versus one-stage AST-first gate, conditional extension and generated-versus-MoE audit | blocked |
| M4 Specialize | FIM/diff, test feedback, engineering tasks, optional growth integration | blocked |
| M5 Certify | one-time external test, claim gate, seeds, independent reproduction | blocked |

## Next twelve work packages

The intended order after the current two ready packages is:

1. `WP-010-B`: pin benchmark runners and contamination checks.
2. `WP-012-B`: collect Python/Rust/C++ code and documentation.
3. `WP-010-C`: attack-test metrics and freeze the SOTA-band comparison set.
4. `WP-012-C`: collect model-architecture and Triton/Linux documents.
5. `WP-011-A`: build the repository-patch executor after benchmark freeze.
6. `WP-012-D`: normalize, deduplicate, split, and token-count the corpus.
7. `WP-011-B`: build Triton and Linux diagnostic executors.
8. `WP-013-A`: fit the tokenizer on train only.
9. `WP-011-C`: add performance guards and golden/attack fixtures.
10. `WP-013-B`: audit tokenizer compression and code-syntax round trips.
11. `WP-013-C`: freeze tokenizer hash and token/parameter ledger.
12. `WP-014-A`: implement the shared compact decoder and invariant-global control.

Package 12 is admitted only when Milestone M1 passes.

## Decision points

| Decision | Evidence | Pass | Fail |
|---|---|---|---|
| DP-01 Benchmark | benchmark registry, contamination, metric attacks | freeze comparison set | replace unavailable benchmark; no threshold |
| DP-02 Data | source licenses, corpus, tokenizer budget | authorize P2M smoke | expand only the deficient corpus family |
| DP-03 Mechanism | P2M, P5M/P12M matched screen | choose smallest surviving tier | stop SFT and retain negative result |
| DP-04 Growth | G0-G5 matched screen | optional mainline integration | keep growth isolated |
| DP-05 SOTA band | one-time external evaluation | certify seeds | report highest justified C-level |
| DP-06 Smallest | three seeds and independent rerun | permit C4 | prohibit world-smallest language |
| DP-07 AST-first | two-stage versus matched one-stage result CFM | promote AST-first heterogeneity claim | remove AST-first from the main claim |
| DP-08 JSON hard gate | 100% strict JSON/schema validity with zero repair/retry | allow C1-C4 | block all coding/engineering/SOTA claims |

## Status update protocol

When finishing a work package:

1. mark only that work package as `passed`;
2. attach the output paths and verification command;
3. update the parent task only if all work packages and the task gate pass;
4. update a milestone only when every required task passes;
5. record a failed decision at its decision point instead of deleting the
   failed branch.

Audit after every state change:

```bash
uv run python scripts/audit_nlp_todo.py \
  --output outputs/reports/nlp_todo_audit_v1.json
uv run pytest tests/test_nlp_todo.py -q
```
