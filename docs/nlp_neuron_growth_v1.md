# Neuron-as-Submodel and Learned Neurogenesis

Status date: 2026-11-01. This document separates the neuroscience evidence from
the proposed machine-learning mechanism. The mechanism is not yet validated.

## 1. What the evidence supports

Recent single-neuron work supports a richer node abstraction than a scalar
point neuron:

- A 2026 *Journal of Neuroscience* filter-and-fire model reports threefold
  memory capacity over a leaky integrate-and-fire neuron on a precise
  spike-timing task and recognizes spatiotemporal MNIST patterns where the
  simpler model fails.
- A 2025 *Neural Networks* model uses dendritic sublinear branches as
  computational units for feature binding in a single Purkinje-cell model.
- A 2025 *Science Advances* study implements XOR and edge detection within one
  dendritic organic electrochemical neuron.
- A 2025 bioRxiv preprint argues that some fly visual interneurons are
  multi-output devices and, under a linear approximation, equivalent to a
  hierarchy of virtual neurons.
- A 2026 PNAS study finds that subcellular ion-channel placement changes
  single-neuron computation.

These results support local state, multiple internal computations, multiple
outputs, and nonlinear/dendritic gating.

They do not support the universal statement "one neuron equals one arbitrary
model". The evidence is task-, species-, morphology-, and model-dependent, and
one key multi-output source is still a preprint.

## 2. Engineering translation

The testable engineering hypothesis is:

> A stateful local submodel with its own routing and readout can discover useful
> computation more efficiently than treating every neuron as a scalar operator,
> and a bounded controller can learn when to create, route, merge, or retire
> such submodels.

A born neuron module may contain:

- local hidden state;
- fast and slow temporal state;
- dendritic or branch gates;
- a typed expert;
- one or more local outputs.

The child starts as a zero-gated residual on a declared parent. Until trained,
it is numerically inert.

## 3. The model must synthesize the E, not just select one

MoE is a control, not the target. A fixed expert pool already contains every
possible E; the router only selects one. The target mechanism must create a new
expert from the current computational state.

The minimal generated-expert loop is:

```text
attempt task
  -> training-only verifier
  -> residual/error signature
  -> growth controller decides whether an expert is useful
  -> expert factory emits a blueprint and low-rank weight factors
  -> zero-gated expert is inserted
  -> bounded adaptation
  -> measure task gain minus parameter/compute/dead-module cost
  -> commit, merge, or retire
```

The factory consumes parent state, residual error, routing entropy, task-family
state, and remaining budget. It emits an expert blueprint, generated low-rank
weight factors, routing, and a zero-initialized birth gate.

It counts as generating an E only if:

- expert weights depend on the current parent and residual context;
- different contexts produce measurably different parameters;
- the expert is not chosen from a fixed pretrained pool;
- a constant clone loses the measured gain;
- factory and expert parameters count in the budget;
- the child is zero-gated at birth.

Choosing one fixed expert, activating a dormant block, cloning the parent, or
adding a scalar gate is not sufficient.

G4 is therefore the generated-expert arm. G5 is a parameter- and
compute-matched fixed-expert MoE control. G4 must beat G5 with experts that are
non-constant across contexts; otherwise the result is ordinary routing.

### Expert synthesis ladder

| Level | What is generated | Claim status |
|---|---|---|
| S0 | Nothing; route over a fixed expert pool | MoE reference only |
| S1 | Low-rank weights for a fixed expert shape | mechanism probe |
| S2 | Rank/width, input/output routing, gate, and weights | minimum core "model builds E" claim |
| S3 | Typed operation graph plus generated weights | optional frontier |

S1 is useful for debugging the factory, but it is not sufficient for the main
self-generation claim. G4 must implement at least S2. S3 is attempted only after
S2 passes its budget and matched-control gates.

## 4. Growth is not unconstrained architecture search

Training is divided into:

```text
warmup -> bounded growth rounds -> consolidation/pruning -> frozen evaluation
```

There are at most three growth rounds. The primary mode reallocates capacity
inside a declared parameter tier: a birth requires headroom or retirement of
another module. Net parameter growth is forbidden for the primary claim.
Absolute-growth runs are diagnostics only and never enter the headline.

The 20:1 relation remains:

```text
total corpus tokens = 20 * unique trainable parameters
```

Generated parameters count. The primary claim cap is 23M parameters, with 46M as
an absolute diagnostic ceiling. Final inference is a static network. No
evaluator call or new neuron appears during the primary inference endpoint.

## 5. Matched controls

The neuron-growth branch is tested against:

| Arm | Purpose |
|---|---|
| G0 fixed control | no growth |
| G1 static width/depth | ordinary capacity increase |
| G2 random births | same births at random times |
| G3 error-triggered births | non-learned residual trigger |
| G4 learned neurogenesis | learned birth/route/merge/retire |
| G5 matched MoE | routing without growth |

G4 cannot be promoted if it is indistinguishable from G2, loses to G1/G5 at
matched final parameters, or only wins with more inference compute.

## 6. Signals and leakage

The controller may use training loss, gradients, module utilization, routing
entropy, residual error, and allowed task-family metadata during training.

It may not use validation/test labels, benchmark identifiers as shortcuts,
oracle plans, or unbounded external evaluator calls.

## 7. Promotion gates

1. Every child is zero-gated at birth.
2. Generated expert weights vary materially with context and are not a clone.
3. Every child receives gradient or is diagnosed as dead.
4. Learned growth beats random births, static width/depth, and fixed-pool MoE.
5. At least one coding/engineering task gains five percentage points with paired
   evidence.
6. Language retention regresses by at most one percent relative.
7. The final parameter/token ratio is exactly 20:1 within two percent.
8. Final inference contains no growth or evaluator calls.
9. The direction persists across two seeds.

Failure modes are module explosion, routing collapse, dead children, benchmark
overfitting in the birth signal, parameter-budget leakage, and gains explained
by extra compute.

## 8. Relationship to the main coding model

This is a parallel research branch. It does not replace the typed-hierarchy
coding mainline before passing its matched screen. If it wins, the winning
mechanism is integrated as an optional branch and rerun against the original
control. If it fails, both the mechanism and negative result remain in the todo.

Machine-readable config:

```text
configs/nlp_neuron_growth_v1.json
```

Audit:

```bash
uv run python scripts/audit_nlp_growth_plan.py \
  --output outputs/reports/nlp_growth_plan_audit_v1.json
uv run pytest tests/test_nlp_growth_plan.py -q
```
