# NLP Model Architecture v4

v4 keeps the v3 compact dynamic stack and adds an explicit final inference
contract. It is the architecture target for the next training round; it is not
a capability claim.

## Final inference path

```text
prompt / repository / schema / tool context
  -> UnifiedPlanner
  -> dynamic serial depth + sparse parallel lanes
  -> generated competence residuals
  -> token-endpoint AST
  -> strict AST handoff evidence
  -> Stage-B generated-AST CFM
  -> domain-specific output contract
```

The final path is `sample_verified_cfm` or
`TwoStageASTFirstPipeline.sample_verified`. The lower-level sampling methods
remain available for structural tests, but they cannot support final inference
claims without verified AST evidence.

## Native typed output

JSON action legality is represented directly by the grammar state. Schema
transitions additionally mask:

- closing an object before required fields exist;
- selecting a value type forbidden by the active schema node;
- opening an object/array where the schema requires another type;
- selecting already-seen or undeclared property names.

`JSONSchemaKeyHead` scores only `SchemaGrammarState.allowed_keys()`. This is a
single learned output head, not parser repair or retry.

Validity is domain-specific and generated from Rust declarations:

| Output | Structural | Schema | Executable |
|---|---:|---:|---:|
| text | yes | no | no |
| JSON | yes | yes | no |
| AST | yes | yes | no |
| code | yes | yes | yes |
| patch | yes | yes | yes |
| tool action | yes | yes | no |

## Training and search

Benchmark target names, training-scope groups, architecture axes, JSON actions,
tree domains, output modes, and output contracts are declared once in Rust.
Python imports the generated contract instead of maintaining parallel lists.

Architecture search now:

- exposes generated-E on/off and runtime-competence on/off as candidate axes;
- records required verifier metrics separately from validation quality;
- marks a selected probe ineligible for promotion when executable success,
  strict JSON validity, or AST parse validity is absent;
- keeps validation-only selection and never reads test.

## Evidence boundary

v4 proves architecture and contract wiring only. The current Stage-A token AST
gate remains blocked, so generated-AST Stage-B quality and SOTA claims remain
unavailable until executable validation passes.

## Data-path hardening

The AST path now uses semantic prompt tokens, learned node/source-length
budgets, a fixed-zero-time categorical training pass, language-specific AST
vocabularies, AST leaf-node binding for token endpoints, free-running
consistency, topology-aware Stage-B conditioning, result-token self-attention,
and uncertainty-driven CFM stopping.

See `nlp_data_pipeline_architecture_audit_v2.md` for the full audit.
