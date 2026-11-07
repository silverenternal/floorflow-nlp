# Dynamic Serial-Parallel Encoder-Decoder Stack v1

## Purpose

The compact model must test a stronger hypothesis than ordinary parameter scaling:
one parameter-shared transformer cell can be reused for variable serial depth, while
lightweight parallel lanes specialize computation for different token states. The
encoder and decoder expose the same mechanism with bidirectional and causal attention
respectively.

This is a test-time and training-time compute-routing experiment, not evidence that a
small model is equivalent to a large model. Every result must report resident
parameters, effective token depth, serial cell calls, lane calls, and wall-clock cost
separately.

## Architecture

For hidden token states `H` and serial step `t`:

```text
U_t = H_t + depth_encoding(t)
B_t = SharedTransformerCell(U_t)
L_t = TopK(LaneRouter(H_t)) { LaneAdapter_l(U_t) }
C_t = B_t + compose(L_t)
H_(t+1) = H_t + halt_probability_t * (C_t - H_t)
```

The shared cell owns all attention and MLP weights. Serial depth reuses them. Each
parallel lane owns only a small low-rank adapter. Lane routing is bounded by top-k and
must report the number of token-lane evaluations actually selected.

A token halts when cumulative halt probability reaches one. The final configured step
is a hard upper bound. Padding tokens are excluded from depth statistics and cannot
alter valid-token representations.

`DynamicStackDecoder` uses causal attention. `DynamicStackEncoder` uses bidirectional
attention. Both can run at `1..max_serial_steps` without changing the parameter count.

## Learning Objective

```text
L = L_task
  + lambda_depth * (mean_token_depth - target_depth)^2
  + lambda_lane  * lane_balance
  + lambda_ponder * expected_compute
```

The depth and ponder terms may be calibrated only on training and validation splits.
They cannot use final-test success to choose per-example depth. The lane balance term
prevents router collapse but does not reward selecting every lane.

## Diagnostics

At minimum the model returns:

- effective mean and maximum token depth;
- number of serial steps actually executed;
- shared-cell token calls;
- routed token-lane calls;
- halt probabilities or cumulative halt state;
- lane routing weights, utilization, and balance loss.

`effective_mean_depth` is not interchangeable with measured FLOPs. The latter is
derived from the call counters and the fixed cell/adapter shapes.

## Matched Comparisons

| Arm | Physical parameters | Serial depth | Parallel lanes | Purpose |
|---|---|---|---|---|
| F1 | shared cell | fixed 1 | fixed | minimal compute control |
| F2 | shared cell | fixed 2 | fixed | fixed-depth control |
| F4 | shared cell | fixed 4 | fixed | deeper fixed control |
| D | shared cell | adaptive `1..4` | top-k | target mechanism |
| D-equal-compute | shared cell | adaptive budget | top-k | compute-matched target |

All arms use the same tokenizer, corpus exposure, optimizer budget, evaluation
protocol, and no repair/retry/best-of-N. An out-of-class large model can be reported
as a reference only; it cannot turn extra inference compute into a compact-parameter
claim.

## Acceptance Gates

The mechanism passes only if:

1. parameter count is independent of configured maximum serial depth;
2. decoder causality, encoder bidirectional context, and padding invariance pass
   deterministic attacks;
3. actual serial and lane compute are auditable;
4. adaptive routing beats the fixed-depth and equal-compute controls on held-out
   executable coding or engineering tasks;
5. the gain remains after matching resident parameters and inference compute.

Failure of any item blocks claims of "small model with large-model capability."

## References

Metadata and abstracts were checked before freezing this design.

- Dehghani et al. (2019), *Universal Transformers*: https://arxiv.org/abs/1807.03819
- Graves (2016), *Adaptive Computation Time*: https://arxiv.org/abs/1603.08983
- Lan et al. (2020), *ALBERT*: https://arxiv.org/abs/1909.11942
- Raposo et al. (2024), *Mixture-of-Depths*: https://arxiv.org/abs/2404.02258
- Hutchins et al. (2022), *Block-Recurrent Transformer*: https://arxiv.org/abs/2203.07852
