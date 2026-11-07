# Compositional Emergence Objective v1

## Material Passport

- Origin: deep-research + experiment-agent + academic-pipeline
- Date: 2026-11-02
- Verification status: literature metadata and abstracts verified; full-text audit pending
- Evidence boundary: research and experiment design only

## Research Question Brief

### Primary RQ

Does adding a sparse basis-composition objective improve executable generalization to
held-out combinations of observed coding atoms at fixed parameters, tokens, and
inference compute?

### FINER

| Criterion | Score | Justification |
|---|---:|---|
| Feasible | 4/5 | Atoms and combinations can be generated from frozen code corpora; held-out execution is runnable. |
| Interesting | 4/5 | Distinguishes grounded recombination from memorization and metric-only emergence. |
| Novel | 4/5 | Combines sparse dictionary learning, learned composition operators, and executable held-out splits under a compact-model budget. |
| Ethical | 5/5 | Uses licensed/local code and synthetic tasks; no human subjects. |
| Relevant | 5/5 | Directly targets coding generalization beyond observed templates. |
| Average | 4.4/5 | |

### Scope

In scope: activation-space basis extraction, sparse coefficient composition,
coefficient-cycle consistency, bounded novelty, and held-out executable composition.

Out of scope: human-like consciousness, general intelligence, unbounded novelty rewards,
benchmark selection, and claims that every discontinuous metric jump is emergence.

## Operational Definition

For an active hidden state `h`, learn a sparse dictionary `V` and encoder `E`:

```text
a = TopK(ReLU(E(h)))
h_hat = V a
```

A learned composition operator `C` combines two sparse supports:

```text
a_c = TopK(ReLU(a_i + a_j + C(a_i, a_j)))
z_c = V a_c
```

“Emergence” means that `z_c` is outside the source-training exemplar support but:

1. reconstructs from the learned basis,
2. round-trips back to the composed coefficient support,
3. stays within an activation-norm envelope,
4. decodes to an output accepted by a task verifier.

The definition is compositional and testable. It does not inherit the scale-based
definition from Wei et al. (2022), nor does it treat a discontinuous metric as proof
of a new capability, following Schaeffer et al. (2023).

## Quantified Emergence Score

The training objective now exposes a bounded differentiable score:

```text
novelty_component = clamp(
  (novel_distance - novelty_margin) / (1 - novelty_margin), 0, 1)

emergence_score =
  novelty_component
  * reconstruction_gate
  * composition_cycle_gate
  * support_recall
  * validity_gate
  * norm_envelope_gate
```

`validity_gate` is derived from:

- grammatical/surface validity;
- execution validity where available;
- composition validity of paired source atoms.

The norm gate rejects novelty obtained by escaping the source activation envelope.
The score is reported as a continuous metric and enters the trained total as:

```text
L_total = ... - lambda_emergence_reward * emergence_score
```

Novelty alone has zero value when reconstruction, cycle, support, validity, or norm
gates fail. The score is not the final claim: held-out executable composition remains
the external endpoint.

## Objective

```text
L_total =
  L_task
  + lambda_rec       L_reconstruction
  + lambda_sparse    L_sparsity
  + lambda_div       L_basis_diversity
  + lambda_comp      (L_cycle + 0.1 * (1 - support_recall))
  + lambda_novel     gate * ReLU(margin - distance_to_seen_support)
  + lambda_balance   (-basis_usage_entropy)
  - lambda_emergence_reward * emergence_score
```

`gate = exp(-cycle_error.detach())`. Novelty is a bounded margin loss, not an
unbounded reward. Reconstruction, cycle, support, and later verifier checks prevent
the model from earning novelty by moving to meaningless regions.

## Literature Synthesis

### Compositional generalization is not guaranteed by structure alone

Lake and Baroni (2018) show that sequence models can mix observed fragments yet fail
on systematic commands. CFQ and its scaled variants (Tsarkov et al., 2020) show that
compositional generalization remains difficult with more data. Wiedemer et al. (2023)
place conditions on the data-generating process and support, while Lippl and
Stachenfeld (2024) show kernel models can still memorize combinations or exploit
shortcuts. Therefore, a composition loss needs explicit held-out support rather than
only entangled-latent regularization.

### Linear factorization and orthogonality are useful inductive biases

Park et al. (2023) formalize concepts as linear directions. Uselis et al. (2026)
derive divisibility, transferability, and stability constraints that require linear,
near-orthogonal per-concept components under standard training. The proposed basis
diversity penalty follows this evidence but remains an inductive bias, not a
guarantee.

### Basis discovery needs sparsity

Elhage et al. (2022) explain how networks store sparse features in superposition.
Cunningham et al. (2023) recover sparse, more monosemantic directions with sparse
autoencoders. The emergence head therefore learns an overcomplete dictionary with
top-k active coefficients.

### Dynamic experts and composition

Shazeer et al. (2017) demonstrate sparse conditional experts; Alet et al. (2018)
compose learned modules in new structures; Ha et al. (2016) generate weights with a
hypernetwork. The project’s generated-expert factory is compatible with this lineage,
but the emergence objective is more specific: it composes semantic basis coefficients
rather than only selecting a fixed expert.

### “Emergence” terminology must remain disciplined

Wei et al. (2022) define emergence operationally through scale discontinuities.
Schaeffer et al. (2023) show such jumps can result from nonlinear metrics. This project
therefore reports continuous and executable metrics, held-out combinations, paired
controls, and effect sizes instead of equating a threshold crossing with emergence.

## Hypotheses

- H1: The full objective reduces composition-cycle error and increases support recall.
- H2: The full objective improves executable success on held-out combinations over
  task-only and novelty-only controls at matched parameters, tokens, and NFE.
- H3: Novelty distance correlates with executable success only when reconstruction,
  cycle, and support gates pass.
- H4: Language retention regresses by no more than one percentage point.

## Ablations

| Arm | Task loss | Reconstruction | Sparse basis | Composition cycle | Novelty margin |
|---|---:|---:|---:|---:|---:|
| C0 | yes | no | no | no | no |
| C1 | yes | no | no | no | yes |
| C2 | yes | yes | yes | no | no |
| C3 | yes | yes | yes | yes | yes |

All arms share the dataset, tokenizer, parameter cap, optimization steps, and inference
budget. Held-out combinations are never used in any loss.

## Data Split

Training contains individual basis atoms and a stratified set of compositions. The
held-out set withholds complete Cartesian combinations, for example:

- `filter` with `sort` while both atoms appear separately;
- `async retry` with `structured logging`;
- typed serialization with nested ownership/borrow constraints;
- Rust iterator composition with C++ range composition.

The held-out executable suite is opened once after C0–C3 checkpoint selection.

## Anti-Gaming Checks

- A novelty-only arm must be retained; if it wins without verifier success, the
  novelty reward is rejected.
- Report activation norm, coefficient entropy, reconstruction error, and support
  recall alongside novelty.
- No best-of-N, retry, or holdout-informed checkpoint selection.
- Report both continuous metrics and discontinuous executable thresholds.

## Experiment Plan

Entry command:

```bash
uv run python project/scripts/experiments/training/models/train_nlp_compact.py \
  --spec project/configs/other/general/chat_sft_canonical/nlp_emergent_chat_sft.json \
  --steps 1500 --batch-size 32 --train-sequences 100000 \
  --validation-sequences 64 --workers 4 --checkpoint-every 100 \
  --validation-batches 2
```

Primary outputs: checkpoint, objective component history, held-out composition report,
parameter/data/compute audit, and negative controls.

## Sources

Metadata and abstracts were verified through the official arXiv API on 2026-11-02.

1. Wei et al. (2022). *Emergent Abilities of Large Language Models*. https://arxiv.org/abs/2206.07682
2. Schaeffer et al. (2023). *Are Emergent Abilities of Large Language Models a Mirage?* https://arxiv.org/abs/2304.15004
3. Lake & Baroni (2018). *Generalization without Systematicity*. https://arxiv.org/abs/1711.00350
4. Tsarkov et al. (2020). *-CFQ*. https://arxiv.org/abs/2012.08266
5. Alet et al. (2018). *Modular Meta-Learning*. https://arxiv.org/abs/1806.10166
6. Shazeer et al. (2017). *Outrageously Large Neural Networks*. https://arxiv.org/abs/1701.06538
7. Ha et al. (2016). *HyperNetworks*. https://arxiv.org/abs/1609.09106
8. Elhage et al. (2022). *Toy Models of Superposition*. https://arxiv.org/abs/2209.10652
9. Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features*. https://arxiv.org/abs/2309.08600
10. Park et al. (2023). *The Linear Representation Hypothesis*. https://arxiv.org/abs/2311.03658
11. Wiedemer et al. (2023). *Compositional Generalization from First Principles*. https://arxiv.org/abs/2307.05596
12. Lippl & Stachenfeld (2024). *When Does Compositional Structure Yield Compositional Generalization?* https://arxiv.org/abs/2405.16391
13. Xu et al. (2022). *Compositional Generalization in Unsupervised Compositional Representation Learning*. https://arxiv.org/abs/2210.00482
14. Uselis et al. (2026). *Compositional Generalization Requires Linear, Orthogonal Representations*. https://arxiv.org/abs/2602.24264

## Devil’s Advocate Checkpoint

- Critical: novelty can be hacked by increasing activation norm. Mitigation: bounded
  margin, norm envelope, cycle gate, and executable verifier.
- Major: a learned basis can be non-identifiable. Mitigation: report behavior, not
  claimed unique “true” concepts.
- Major: generated training compositions can become memorized. Mitigation: a separate
  never-trained held-out combination suite.
- Major: a tiny model may improve proxy metrics without executable gains. Mitigation:
  executable held-out success is the primary claim.

