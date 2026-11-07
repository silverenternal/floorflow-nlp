# Fast CFM Training and Strict AST Upstream Gate v1

## Training Framework

The two-stage model is only useful if flow training converges quickly enough to support
rapid architecture iteration. The training utilities now provide:

- logit-normal timestep sampling concentrated away from saturated endpoints;
- a single interpolation/target implementation to prevent state-target drift;
- EMA evaluation weights for stable early checkpoint selection;
- finite-gradient checks and gradient clipping;
- warmup-cosine scheduling;
- deterministic sampling primitives for matched convergence arms.

All convergence comparisons use identical initialization, step count, batch size,
parameters, noise seeds, and evaluation points. Extra NFE is reported separately.

## Stage-A Upstream Quality

Parser validity alone is not sufficient. A grammar-template shell can parse perfectly
while the token endpoint is unusable. Stage B therefore requires the token-endpoint AST
to satisfy:

- node-type F1 ≥ 0.95;
- typed-edge F1 ≥ 0.95;
- scope-relation F1 ≥ 0.90;
- lexical F1 ≥ 0.95;
- normalized tree edit distance ≤ 0.05;
- source path explicitly marked `token_endpoint`.

Grammar-template retrieval is excluded from the Stage-A synthesis gate. If the token
endpoint fails, Stage B does not receive a quality-qualified upstream state.

## Rapid Iteration Protocol

For every CFM change:

1. run deterministic gradient and leakage tests;
2. run the synthetic uniform-versus-logit-normal convergence probe;
3. evaluate Stage-A token endpoint at fixed NFE 1/2/4/8;
4. only if upstream AST thresholds pass, evaluate Stage B from generated AST;
5. report model and EMA quality at the same NFE and compute budget.

Training loss alone cannot promote an architecture branch.
