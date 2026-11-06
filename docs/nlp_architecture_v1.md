# Compact Coding SOTA Architecture v1

Status date: 2026-11-01. This is an architecture contract, not a capability
result.

## 1. Ambition

The target is SOTA-level coding and engineering capability inside a compact
parameter/compute class. This is not a GPT replication and does not compete by
increasing parameters or training tokens.

The primary claim cap is 23M unique trainable parameters. P46M is a diagnostic
ceiling only. Standard autoregressive token decoding is retained as the language
interface, while the capability mechanisms come from hierarchy, typed channels,
CFM, sparse routing, verifier feedback, and generated experts.

## 2. Five decoupled channels

| Channel | Content | Separate supervision |
|---|---|---|
| Syntax | indentation, delimiters, grammar, block structure | delimiter/indentation prediction, syntax repair |
| Symbol | identifiers, declarations, imports, API references | identifier masking, definition-use retrieval |
| Discourse | task intent, plan, segment roles, patch intent | plan reconstruction, segment boundaries |
| Execution | tests, compiler diagnostics, runtime failures | error localization, verifier-conditioned patch score |
| Engineering | Triton, Linux, profiling, memory/latency constraints | kernel/spec reasoning, systems diagnosis |

Every channel has its own encoder, output contract, auxiliary objective, and
cross-attention relation mask. Channels are not concatenated feature streams
with one shared loss.

## 3. Attention and routing

The candidate borrows four proven mechanisms:

1. floorplan semantic channel attention: channel queries, relation priors, and
   gated feedback;
2. CHIP typed cross-attention: explicit allowed masks, group-type bias, and
   zero-gated residuals;
3. MiTA route attention: budgeted top-k keys and sparse long-context routing;
4. multiflow slot mixing: bidirectional gated plan/task slots and active-slot
   masks.

Code/token state queries typed memory containing global, syntax, symbol,
discourse, execution, and engineering tokens. Every mechanism is individually
ablatable.

## 4. Two-stage AST-first CFM

The primary differentiation claim is a strict two-stage CFM:

```text
Stage A: prompt/repo/tests
    -> AST root plan
    -> AST subtree and block structure
    -> AST leaf grammar and symbol anchors
    -> generated AST

Stage B: generated AST + prompt/repo/tests
    -> result latent
    -> code/patch/engineering output through token decoder
```

Stage A is itself CFM over a typed AST graph. Its objectives include AST
velocity, endpoint consistency, node-type CE, edge-type/parent-child loss,
grammar validity, and parseability after discrete AST decoding.

Stage B is a second CFM conditioned on the generated AST. It combines result
velocity, endpoint consistency, token/patch reconstruction, executable verifier
feedback, and AST-to-result consistency.

The primary NFE budget is 4 for Stage A plus 4 for Stage B, reported as total 8.
CFM-A 1 + CFM-B 1 and 8 + 8 are secondary budget points.

Training may use a teacher AST early, then generated/perturbed AST, and must end
with generated-AST-only training. Primary inference uses only Stage-A generated
AST. Target AST is never available.

The claim requires a matched one-stage control conditioned directly on the
prompt/repository state. Without that control and the generated-AST endpoint,
the architecture is not evidence for AST-first generation.

Primary AST metrics are:

- parse success and grammar validity;
- node-type and edge-type F1;
- parent-child and scope relation accuracy;
- tree edit distance;
- AST-to-code parse consistency;
- execution success after Stage B.

## 5. Separate training, then joint calibration

The staged training deliberately borrows the channel/scope separation that
worked in the other two domains:

| Stage | Train | Freeze |
|---|---|---|
| S0-base | compact decoder and token readout | none |
| S1-channels | five channel encoders and auxiliary heads | decoder/readout |
| S2-cross-attention | typed cross-attention, MiTA adapters, fusion | decoder/readout/channels |
| S3-cfm | Stage-A AST CFM, Stage-B result CFM, and endpoint adapters | base/channels/fusion |
| S4-joint | hierarchy, fusion, and language heads | tokenizer/data manifest |
| S5-factory | generated experts and growth controller | tokenizer/benchmark/base schedule |

S2 must beat the frozen base before joint training. Both CFM stages must improve
AST parse/validity and executable patch success, not only latent MSE. S5 is
conditional on the generated-versus-MoE gate.

## 6. Data used by training

All previously planned data families are training inputs, not just download
artifacts:

- general language: `.data/nlp/tinystories_small_v1`;
- code: `.data/nlp/mixed_v1/code` with Python, Rust, and C++;
- model architecture: `.data/nlp/mixed_v1/model_architecture`;
- engineering acceleration: `.data/nlp/mixed_v1/engineering`, including
  Triton and Linux.

The P2M engineering template still uses the seed only for the initial training
stack smoke. P5M and larger real screens must use the frozen mixed manifest.

## 7. Experimental arms

```text
A0 compact decoder control
A1 A0 + one-stage result CFM
A2 A0 + two-stage AST-first CFM
A3 A2 + five typed channels
A4 A3 + semantic/typed cross-attention + MiTA + slot mixing
A5 A4 + generated expert factory + verifier feedback
```

Every arm uses the same tokenizer, total trainable parameter cap, training
tokens, benchmark, and inference budget.

## 8. Gates

- no component is promoted by parameter count alone;
- channel ablations are individually measurable;
- cross-attention-only training beats the frozen base;
- Stage A produces parseable AST at the primary NFE;
- Stage B uses only generated AST and improves executable behavior;
- a matched one-stage control isolates the two-stage gain;
- final resident parameters stay at or below 23M;
- final inference is static;
- claims remain restricted to the frozen compact benchmark class.

Machine-readable contract:

```text
configs/nlp_architecture_v1.json
```

Audit:

```bash
uv run python scripts/audit_nlp_architecture.py \
  --output outputs/reports/nlp_architecture_audit_v1.json
uv run pytest tests/test_nlp_architecture.py -q
```
