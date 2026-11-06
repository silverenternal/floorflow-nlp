# NLP Model Architecture v3

## Unifying Principle

The model should have one capability primitive and one planner:

> Generate or route reusable competence cells, compose them conditionally, and
> constrain the result to a shared typed tree.

This replaces three overlapping primitives with:

- `CompetenceCell`: the common low-rank residual/stateful unit;
- `UnifiedCompetenceBank`: named competences plus one context-generated E slot;
- `UnifiedPlanner`: one router for depth bias, parallel lanes, competences, E birth,
  and output-tree mode;
- `TypedTreeNode`: one structural representation for AST, JSON, patch, and tool actions.

Legacy gap ensemble, fixed expert bank, and neuron modules remain as matched controls.
They are not instantiated in the v3 primary path.

## Runtime Flow

```text
token + turn embeddings
  -> unified planner computes depth bias
  -> dynamic shared serial/parallel stack
  -> typed channels and cross-attention
  -> unified planner selects top-k competences
  -> context-generated E slot
  -> one composed residual state
  -> typed tree / JSON action / token heads
```

The same planner produces:

- per-token depth bias for dynamic stack halting;
- parallel-lane routing;
- sparse competence weights;
- generated-E birth probability;
- text/code/JSON/AST mode logits.
- a shared compute-budget probability that scales depth, lane, and competence work.

Planner heads have independent adapters, so depth, lane, competence, birth, and mode
can be frozen or ablated separately while sharing the planner state and budget.

Competence routing now performs real token-gather computation: only tokens selecting a
cell invoke that cell. Diagnostics report selected token-cell calls and the dense
counterfactual count.

Selected competence states are composed by a zero-gated cross-attention residual, so the
planner controls which cells execute while cross-attention learns how their states mix.

## What “Generated E” Means

The current generated E is a context-conditioned low-rank expert:

```text
context -> coefficients, rank, width, input/output scales
left/right weights = combinations of learned basis tensors
```

It is generated during the forward pass, but only inside the parameter manifold learned
during training. It does not:

- create a new independent parameter tensor outside the learned basis;
- prove that the generated expert will be useful;
- establish neuron-level intelligence or large-model equivalence;
- imply that inference parameters are free.

The architecture now exposes birth probability, generated rank/width, residual norm,
generated tensor count, and generated FLOPs separately. Fixed-pool, random-frozen, and
disabled controls share the same trunk placement.

Generated E has two execution modes:

- online: materialize weights from context during each forward;
- materialized/static: export left/right weights once and apply them without rerunning
  the hypernetwork.

Resident basis parameters are counted in both modes.

## Runtime Competence Lifecycle

`RuntimeCompetenceManager` adds the deployment-time lifecycle:

```text
gap detector
  -> synthesize candidate E
  -> attach verifier outcome
  -> promote after repeated valid utility
  -> retain/reuse or retire
```

The store is explicit and bounded. Ordinary model forward never silently persists a
generated expert. Promotion requires a configured number of trials and success rate;
invalid candidates can be retired. Fixed-pool, random-frozen, disabled, online, and
materialized/static paths remain available as controls.

Promoted/candidate records can be exported and restored with:

```text
RuntimeCompetenceManager.save(path)
RuntimeCompetenceManager.load(path)
```

The exported record includes materialized weights, rank/width, trigger evidence,
verifier history, utility, and lifecycle status.

## Typed Tree

`TypedTreeNode` normalizes:

- source AST nodes and typed edges;
- strict JSON objects/arrays/scalars;
- patch file/span trees;
- tool-action trees.

Each node carries domain, kind, value, field-labelled children, metadata, and explicit
structural/schema/execution validity. AST and JSON remain different grammars while
sharing representation, routing, and validation contracts.

AST token endpoints and native JSON generation now both return `TypedTreeNode`; rendering
and validity use the same tree contract.

`TypedTreeGateway` is the only finish path for code/AST, JSON, patch, and tool-action
results. It owns mode selection, rendering, and explicit validity.

## Structural Guiderails

- competence cells and generated E start at zero residual;
- unified planner depth bias starts near zero;
- E controls remain separately freezable;
- one planner replaces independent depth/gap/expert routers;
- no generated parameter is hidden from resident/compute accounting;
- Stage-B cannot consume an AST below its configured quality threshold.

## Evidence Boundary

This is a network-design implementation. No capability, convergence, compression, or
SOTA result is claimed. The required matched experiments remain in `nlptodo.json`.
