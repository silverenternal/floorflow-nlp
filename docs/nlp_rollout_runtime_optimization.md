# NLP Rollout Runtime Optimization

## Current evidence

The matched structural ablation started on 2026-11-04 at 01:12 local and is still
running under its registered command. At the latest check the process had used
about 14h44m wall time and one CPU core continuously. Hierarchical checkpoints for
seeds 17 and 29 were written at 06:17 and 10:30; seed 43 was still running. The
aggregate report had not yet been written. Sampled GPU utilization was 0% while
the model process retained about 5.3 GiB of GPU memory. These observations show
poor device utilization and long arm latency; they do not identify a single root
cause.

Code inspection confirms repeated work in the autoregressive path:

- `generate_prefix_tree` reconstructs the prompt tensors and growing prefix batch
  inputs for each action.
- `PrefixTreePolicy.forward` re-encodes the unchanged prompt and blueprint on every
  action, then runs the full causal decoder prefix to read only the final step.
- Inference context tensors are copied back to Python lists and integers every
  action, which can synchronize device execution.

At 15:58, `nvidia-smi` identified the NLP process on GPU 0 at 0% utilization; GPU 1 was at 100% for a separate raster-instance training process. The NLP run was launched with `--device cuda:0`; avoid GPU 1 and keep the registered process on its assigned device. This snapshot identifies concurrent hardware use but does not explain the CPU-side runtime by itself.

The registered run has ended. Its source and configuration remain as historical
identities; the incomplete result must not be repaired in place or counted as a
completed arm-seed run.

At the registered six-hour cumulative seed boundary, hierarchical seed 43 was
stopped at 16:26 local. Its interrupt traceback was inside
`TreeCompletionBudget._node_completion_cost`, repeatedly constructing
`ASTCompletionState.open_parent_context`; seed 43 had no checkpoint and the runner
did not write an aggregate report. This identifies repeated completion accounting
as an actual hot path in this run, replacing the earlier decoder-prefix hypothesis
as the first optimization target.

## Research findings

1. Shazeer describes incremental Transformer inference and reduces its KV memory
   bandwidth with shared key/value heads. The relevant transferable principle is
   to reuse prior key/value projections while extending a causal prefix; adopting
   multi-query attention itself would change the architecture and require training
   again. [Fast Transformer Decoding: One Write-Head is All You Need]
   (https://arxiv.org/abs/1911.02150)
2. Katharopoulos et al. reformulate attention as a recurrent linear-attention
   computation for fast autoregressive inference. This changes the attention
   function, so it is a later model-family experiment rather than a semantics-
   preserving fix for the current checkpoint.
   [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention]
   (https://arxiv.org/abs/2006.16236)
3. Stern et al. use blockwise drafts and a scoring model to accept multiple
   autoregressive steps per iteration. This could reduce serial decisions, but
   tree actions mutate parser/grammar state, so proposals must be validated against
   every intermediate state before this method is safe here.
   [Blockwise Parallel Decoding for Deep Autoregressive Models]
   (https://arxiv.org/abs/1811.03115)

## Implementation order after the registered run

1. **Implemented:** cache parser-schema relation indexes and compute the completion
   cost of every existing node once per decoder action with an iterative reverse
   preorder pass. Reuse that snapshot while scoring parent/edge candidates for the
   same action. This removes recursive subtree recomputation and repeated schema
   scans from the observed path.
2. **Implemented:** add cooperative total and per-seed deadlines at optimizer
   batch/step, rollout action, validation case, and validation batch boundaries.
   Save a marked partial checkpoint before free-running evaluation and return a
   budget-stopped partial report with the phase and completed training metrics.
3. Next, rerun focused CPU contracts and validate an interrupted miniature run
   writes its partial report and checkpoint. Then preregister a fresh matched
   comparison; never reuse the incomplete seed 43 attempt as a completed run.
4. After profiling the new run, add an exact incremental inference path that
   encodes prompt and blueprint once, caches decoder self-attention keys/values,
   and reuses fixed cross-attention memory projections. Keep the full-prefix path
   as the reference until parity is established.
5. Verify float32 per-head logits against full-prefix execution on fixed prefixes,
   then verify greedy action, trace, rendered source, and strict parser outcome
   are identical. Benchmark CPU batch construction separately from GPU model
   execution before deciding whether model kernels or Python orchestration dominate.
6. Only after exact parity and CPU contracts pass, run a preregistered, matched
   short GPU runtime comparison. Report wall time, device utilization, action
   count, peak memory, and generated-result parity; do not use historical
   checkpoints for quality claims.

This plan addresses measured latency without changing the trained model's
attention function. Linear attention and blockwise decoding remain separate
research branches because they alter model behavior or require intermediate-state
verification.
