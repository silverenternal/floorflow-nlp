# Natural-Language Cross-Domain Plan v1

Status date: 2026-11-01. This document defines a research route only. It
contains no language-quality result.

## 1. Role in the project

The residential floorplan route is the primary task. CHIP is the first
cross-domain validation of the same modelling idea. This document establishes
natural-language modeling as the second cross-domain validation.

The question is deliberately narrower than "build another LLM":

> Under a matched pretrained language backbone, tokenizer, dataset, parameter
> budget, training exposure, and inference budget, does typed hierarchical
> token-state flow improve constrained language generation without material
> held-out NLL regression?

The matched control is a compact hierarchical decoder with an invariant global
document state. The candidate keeps the same core and adds document, segment,
token-span, entity/constraint state, and bounded module synthesis through
zero-gated typed branches. A standard token readout is retained only as the
language interface; this is not a GPT-2 replication.

The primary claim cap is 23M unique trainable parameters. P46M is a
non-headline diagnostic. Generated experts reallocate capacity inside a tier;
they do not increase net parameters in the primary route.

## 2. What transfers

The validation transfers mechanisms, not floorplan or chip-specific code.

| Shared mechanism | Floorplan | Chip floorplanning | Natural language |
|---|---|---|---|
| Target-free condition | contour | design specification | prompt/context |
| Global plan | room program | global spec | document intent and required entities |
| Hierarchy | room / structure / object | L0 / L1 / L2 / L3 blocks | document / segment / token span / entity |
| Typed channels | room / wall / opening / furniture | block / relation / port / constraint | syntax / symbol / discourse / execution / engineering |
| Exact target decode | room/layer composition | exact block area | exact token or segment budget when declared |
| Hard verifier | footprint and structure | legalizer / scale-pack | grammar/schema verifier and bounded projection |
| Failure tail | thin walls and openings | dense-pin and b2b tails | rare constraints, long prompts, entity collisions |

This mapping is a hypothesis. It does not imply that geometric constraints and
linguistic syntax are the same object, nor that a language task has a unique
plan behind every surface sequence.

## 3. Architecture hypothesis

The candidate has four explicit state levels:

1. `L0`: document plan with intent, topic, length, and required entities.
2. `L1`: paragraph, section, clause, and discourse structure.
3. `L2`: surface token/span flow state.
4. `L3`: entity, relation, exact-constraint, and verifier anchors.

Typed memory is separated into syntax, symbol, discourse, execution, and
engineering channels. New branches are zero-gated so loading the language head
into the control backbone is numerically inert before training. A
grammar/schema/execution verifier is an evaluator first. Constrained projection
is admitted only after the verifier is deterministic, attack-sensitive, and
shared by every arm.

The first implementation should not replace autoregressive decoding with CFM.
It should use flow for hierarchical plan/segment state and preserve a standard
token decoder unless a matched no-flow decoder ablation proves otherwise.

## 4. Data ladder

### Stage L1: synthetic grammar

Generate documents from a deterministic latent plan with required entities,
relations, segment order, grammar, and exact token budget. This stage runs on
CPU and exists only to test contracts, gradients, decoding, and verifier
sensitivity. It cannot support a language-quality claim.

### Stage L2: TinyStories

Use the downloaded and pinned general-language seed from
`roneneldan/TinyStories` to compare:

- compact decoder with one invariant global state;
- the same core with typed hierarchy token flow;
- parameter-only and unmatched-compute controls.

The corpus, tokenizer revision, split hashes, effective batch, optimizer, and
number of training tokens must be frozen before results are opened.

The task-specific coding and engineering target, parameter-capability ladder,
and C0-C4 SOTA claim boundary are defined in
[`nlp_coding_sota_plan_v1.md`](nlp_coding_sota_plan_v1.md).

The broader corpus will add Python, Rust, and C++ documentation/source,
model-architecture documents, and Triton/Linux engineering documents. Its
family shares, license controls, cleaning rules, and token-to-parameter tiers
are defined in [`nlp_corpus_plan_v1.md`](nlp_corpus_plan_v1.md).

### Stage L3: executable constraints

Add a benchmark whose outputs have deterministic checks, such as required
entities, relation direction, segment order, exact token budget, and a declared
schema. The verifier is reported as part of the method rather than silently
repairing failures.

### Stage L4: external suite

Freeze unseen prompts and references before generation. Report language
quality, constraint validity, diversity, and cost by prompt length and
constraint type. Do not select prompts after observing outcomes.

## 5. Matched experimental ladder

| Arm | Backbone | Hierarchy | Typed channels | Verifier-aware decode |
|---|---|---:|---:|---:|
| C0 global-state control | compact decoder | no | no | no |
| C1 hierarchy only | compact decoder | yes | no | no |
| C2 typed hierarchy | compact decoder | yes | yes | no |
| C3 verifier-aware | compact decoder | yes | yes | yes |
| C4 parameter-only control | wider matching head | no | no | no |

Each arm must have the same tokenizer, corpus, training tokens, optimizer, and
sampling budget. The parameter ratio is capped at `1.05`. One mechanism is
enabled at a time until the matched screen passes.

## 6. Metrics and gates

Language quality:

- held-out token NLL and perplexity;
- token accuracy;
- prompt- and length-stratified NLL.

Task and structure:

- constraint success;
- exact match and entity/relation F1;
- grammar validity;
- segment-boundary F1;
- verified projection rate.

Distribution and safety:

- unique n-gram ratio and self-diversity;
- held-out coverage;
- no target text, oracle plan, or test metric at inference;
- no hidden retry or best-of-N selection in the primary endpoint.

Promotion requires all of the following:

1. no inference leakage;
2. all matched fields identical;
3. parameter ratio at most `1.05`;
4. relative NLL regression at most `1%`;
5. at least one preregistered task metric improves by `5` points with a paired
   bootstrap interval excluding zero;
6. decoder work is at most `1.25x` the control;
7. the direction persists on two seeds and a frozen unseen suite.

If these fail, the transfer is recorded as a negative result. A language SOTA
claim is outside the scope of every stage in this plan.

## 7. Workspace contract

The route has its own namespace and artifacts:

```text
src/floorflow/nlp_flow/       model and contract implementation
rust/nlp_flow_core/           runtime traits and hierarchy specifications
rust/nlp_flow_macros/         Rust procedural macros
rust/nlp_flow_codegen/        Rust-to-Python contract renderer
configs/nlp_*.json            frozen protocols and experiment configs
scripts/*nlp*.py              audit, train, evaluate and reporting entrypoints
tests/test_nlp_*.py           focused contracts
docs/nlp_*.md                 plans, evidence and claim boundaries
nlptodo.json                  machine-readable research state
```

The first gate is implemented by:

```bash
uv run python scripts/audit_nlp_cross_domain_plan.py \
  --output outputs/reports/nlp_cross_domain_plan_audit_v1.json
uv run pytest tests/test_nlp_flow_protocol.py -q
```

The Rust declarations own the hierarchy and typed-channel boilerplate. Generate
and verify the Python constants with:

```bash
cargo run --manifest-path rust/Cargo.toml -p nlp_flow_codegen -- \
  --format python \
  --output src/floorflow/nlp_flow/_generated_protocol.py
cargo run --manifest-path rust/Cargo.toml -p nlp_flow_codegen -- \
  --format python \
  --check src/floorflow/nlp_flow/_generated_protocol.py
```

No GPU training should start until the synthetic CPU contract, leakage audit,
and matched controls are complete.
