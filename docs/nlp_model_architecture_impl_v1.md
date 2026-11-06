# Compact Model Implementation v1

Status: first compact decoder layer implemented. Channel fusion, CFM, JSON
grammar, and expert factory remain subsequent layers.

## Layer order

The implementation follows the data path:

```text
token embedding
  -> GQA + RoPE decoder blocks
  -> final RMSNorm
  -> five typed learning heads
  -> typed fusion / AST + result CFM
  -> native JSON grammar heads
  -> tied token output
  -> verifier and task evaluator
```

## Implemented now

- `NLPModelConfig` with a 23M absolute cap check.
- Pre-norm decoder blocks.
- Grouped-query attention with separate q/k RMSNorm.
- Rotary positional encoding.
- SwiGLU feed-forward blocks.
- Causal and key-padding masks through scaled dot-product attention.
- Five independently parameterized typed channel heads.
- Zero-initialized channel residual output, preserving the base decoder.
- Tied token output projection by default.
- Unique trainable parameter accounting through the engineering layer.

The initial P2M configuration is deliberately small: 160 hidden width, three
layers, two KV heads, and 400 SwiGLU width. It is a training-stack and mechanism
test, not the final coding model.

## Not implemented yet

1. masked channel aggregation and typed cross-attention;
2. MiTA sparse routing;
3. bidirectional slot mixing;
4. Stage-A AST CFM and Stage-B result CFM;
5. native JSON grammar/schema heads;
6. generated expert factory;
7. staged training controller and feature caches.

These are blocked by their registered tasks and will be added as zero-gated
residuals rather than replacing the decoder.

## Verification

```bash
uv run pytest tests/test_nlp_model_blocks.py -q
```
