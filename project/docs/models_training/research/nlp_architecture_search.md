# Efficient Architecture Search v1

The v3 model no longer relies on a single hand-selected configuration or a
small manual ablation table. `nlp_architecture_search.json` defines a
budgeted candidate space and a validation-only successive-halving controller.

## Why this is required for a small SOTA attempt

For a compact coding model, the useful question is not only whether one fixed
network can fit a parameter cap. It is whether training can identify the
smallest network that remains on the validation quality plateau, while
separating resident parameters from actual serial/parallel compute.

Candidates are generated from:

- hidden width;
- maximum dynamic serial depth;
- parallel lane and top-k width;
- unified planner width;
- competence-cell rank and top-k;
- generated-E rank and basis count.
- generated-E online/disabled, runtime-competence, and schema-key-head controls.

Every candidate is instantiated with the real model class before entering the
search. Parameter counts therefore use the same unique-trainable-tensor audit as
the training and checkpoint paths.

Disabled generated-E is a real parameter ablation: the factory and basis are not
instantiated. Runtime competence requires online E and therefore cannot be
silently paired with the disabled control.

## Selection protocol

1. Generate deterministic low, high, one-factor, and joint candidates.
2. Reject candidates over the declared parameter or compute budget.
3. Train every surviving candidate with the same seed, batches, tokenizer, and
   token budget.
4. Evaluate validation quality only. Test labels are rejected by the evaluation
   type itself.
5. Keep the better half at each 1x, 2x, and 4x token rung.
6. Compute the Pareto front over quality, resident parameters, training FLOPs
   per token, inference FLOPs, and measured latency.
7. Select the smallest Pareto point within the configured quality plateau.
8. Freeze the selected model configuration and complete search ledger.

The current probe's quality metric is negative validation loss. That is enough
to validate the search machinery, but it is not enough to promote an
architecture. Promotion requires executable success, strict JSON validity, and
AST parse validity on frozen validation suites. Any later tensor-ring or
low-rank compression must rerun the same capability gates.

## Commands

```bash
uv run python project/scripts/experiments/research/run_nlp_architecture_search.py --contract-only
uv run python project/scripts/experiments/research/run_nlp_architecture_search.py --device cuda:1
```

The second command is a short matched screen, not a SOTA result. It writes
`outputs/reports/nlp_architecture_search_probe_v1.json` and freezes the selected
probe config only after validation-based selection.
