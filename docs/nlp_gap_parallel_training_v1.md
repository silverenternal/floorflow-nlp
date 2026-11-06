# Parallel Gap Training v1

## Goal

Do not optimize one entangled aggregate loss until it stalls. Split the distance to the
coding/engineering target into verifier-bound gaps, train their lightweight adapters in
parallel, then learn a token-conditioned cross-attention aggregator.

The open-capability track may temporarily use larger physical models. It is explicitly
un-compressed and cannot enter the smallest/SOTA claim until compression and matched
post-compression evaluation pass.

## Gap Channels

| Gap | Supervision | Verifier |
|---|---|---|
| JSON legality | valid/invalid structured outputs | strict parser + schema |
| AST structure | nodes, edges, codec reconstruction | parser / round-trip |
| Stage-B CFM | generated AST to result | execution |
| Python execution | compile/run outcomes | bounded Python executor |
| Rust execution | compile/run outcomes | rustc + tests |
| C++ execution | compile/run outcomes | C++ compiler + tests |
| Engineering acceleration | Triton/Linux/perf tasks | numerical/latency/shell checks |
| Architecture knowledge | grounded model-doc questions | frozen question rubric |
| Language retention | language modeling + grammar | held-out language suite |
| Compositional emergence | basis cycle + surface/execution validity | held-out executable compositions |

Each gap uses its own low-rank adapter and loss head. Disabling a gap removes its state,
loss, and adapter from the active graph. Held-out and final-test rows from one channel
never select another channel's checkpoint.

## Training Sequence

### Stage 1: parallel gap training

Freeze or strongly limit the shared trunk. Train each gap's adapter and loss head with
its own split and checkpoint. This stage is embarrassingly parallel across gaps.

Exit only when every enabled adapter has finite loss, isolated nonzero gradients, and a
frozen verifier report.

### Stage 2: router calibration

Freeze the trunk and gap adapters. Train only the zero-gated cross-attention aggregator.
It must start as identity. Report routing entropy, utilization, balance, and
parameter-freezes instead of treating attention weights as capability.

### Stage 3: joint low-LR tuning

Unfreeze the trunk at a low learning rate and tune the complete composite. Open final
tests once. Compare against monolithic and frozen-router controls matched on physical
parameters, training tokens, optimizer steps, tokenizer, and inference compute.

## Hard Gates

- Native JSON validity remains 1.0 with zero repair and zero retry.
- AST and generated code are accepted by parsers/compilers/executors.
- No held-out or test data selects a model.
- No best-of-N in the primary endpoint.
- The open-capability model remains visibly separate from the compressed primary.

## Compression Handoff

Freeze the best capability checkpoint, resident-parameter ledger, data/tokenizer hashes,
evaluation suites, and aggregate reports. Tensor-ring, low-rank, quantized, and hybrid
compression candidates then start from this checkpoint and must rerun the same gates.
