# Native Strict-JSON Validity Contract

Status date: 2026-11-01. This is an architecture requirement; model evaluation
has not yet occurred.

## 1. The guarantee

Every machine-actionable JSON output must have:

- strict JSON parse validity `1.0`;
- schema compliance `1.0`;
- zero duplicate keys;
- zero non-finite numbers;
- zero post-hoc repairs;
- zero retries.

This is a hard gate. Any failure blocks C1-C4 and every coding/engineering or
SOTA claim.

## 2. Why training alone cannot guarantee 100%

An unconstrained token decoder assigns probabilities to invalid grammar
transitions. Even a very small invalid-transition probability accumulates over
a long document. Optimizing a training loss reduces the probability but cannot
prove zero probability over all reachable outputs.

The required guarantee is therefore **valid-by-construction through model
structure**, not post-hoc repair.

## 3. Native generation mechanism

The decoder carries:

- a typed JSON tree target;
- an explicit JSON grammar state;
- a schema state stack;
- typed token heads for object, array, property key, string, number, boolean,
  null, and schema reference.

Invalid grammar/schema transitions have structurally zero probability in those
native heads. This is part of the model's decoding structure, not a separate
repair service.

Training includes:

- grammar-state transition cross-entropy;
- typed value head loss;
- object/array closure prediction;
- escaping, unicode, and finite-number objectives;
- duplicate-key prevention;
- malformed negative contrast.

The curriculum progresses from flat objects to nested schemas, engineering
actions, and adversarial strings/numbers.

## 4. Forbidden implementations

- regex extraction from free-form output;
- adding missing braces or brackets;
- parser fallback that rewrites content;
- semantic default-value injection;
- retry until parsing succeeds;
- choosing the first valid sample from best-of-N;
- truncation or auto-closing structures;
- a separate constrained-decoding service outside the model.

## 5. Test suite

- strict parser success on every generated structured output;
- schema validation under every declared schema;
- duplicate-key and `NaN`/`Infinity` attacks;
- quotes, newlines, unicode, and escape sequences;
- deeply nested objects and arrays;
- zero probability for invalid transitions;
- AST JSON round trip;
- generated JSON consumed by Stage B without repair.

Machine-readable contract:

```text
configs/nlp_json_contract_v1.json
```

Audit:

```bash
uv run python scripts/audit_nlp_json_contract.py \
  --output outputs/reports/nlp_json_contract_audit_v1.json
uv run pytest tests/test_nlp_json_contract.py -q
```
