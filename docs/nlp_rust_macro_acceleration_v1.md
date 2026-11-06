# Rust Macro Acceleration v1

The Rust workspace is now the canonical declaration layer for iteration-critical
model contracts. It is not only a protocol renderer.

## Generated surfaces

- benchmark proxy scalar/token/regression targets;
- training-scope group names used by staged and parallel training;
- architecture-search axes;
- native JSON action names;
- typed-tree domains and output modes;
- output validity requirements.

The Python implementation imports these tables directly. A Python scope prefix
missing from Rust, a search axis not declared in Rust, or a stale generated
contract fails immediately.

## Commands

```bash
uv run python scripts/audit_nlp_rust_contracts.py --sync
uv run python scripts/audit_nlp_rust_contracts.py
```

The first command is the development loop. The second is the frozen CI check.

## Why this improves iteration

Adding a benchmark target, training scope, JSON action, or architecture axis is
now a declaration change followed by code generation. Python-side silent drift
is replaced by a compile-time macro validation plus generated-file equality.

## Evidence boundary

This tooling accelerates and checks model wiring. It does not provide coding
capability, emergence, SOTA, or benchmark evidence.
