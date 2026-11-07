# NLP Model Architecture v2

## Scope

This is the structural repair round for the remaining model tasks. GPU experiments are
deferred; the only evidence here is CPU contract testing of shapes, gradients,
zero-gated initialization, routing controls, masks, and loss reachability.

The end-to-end data-flow findings and fixes are recorded in
`project/docs/models_training/audits/nlp_training_inference_information_audit.md`.

## Main Path

```text
token embedding + turn-segment embedding
  -> dynamic shared serial/parallel encoder-decoder
  -> typed channels
  -> typed cross-attention fusion
  -> local neuron residual (zero gated)
  -> generated/fixed/random-frozen expert residual
  -> independently routed gap adapters + zero-gated cross-attention
  -> benchmark-proxy heads and conversation-control heads
  -> token / JSON / typed outputs
```

## Repairs by Task

| Remaining area | Structural implementation |
|---|---|
| dynamic stack | shared serial cell, top-k parallel lanes, early halt, actual compute counters |
| gap router | independent gap adapters, zero-gated aggregator, frozen-router control |
| generated growth | generated expert bank integrated into forward, zero birth gate |
| static controls | fixed expert pool and random-frozen birth mode |
| MoE disentanglement | identical model placement with switchable expert mode |
| conversation | turn embedding, learned continue/stop, boundary, output-mode, language, validity heads |
| native JSON | grammar action head remains the generation path; conversation validity is separately supervised |
| AST-first CFM | endpoint equation corrected from full velocity to remaining-time velocity |
| AST token endpoint | spacing/punctuation gap tokens added to AST vocabulary, target, output head, loss, and decoder |
| upstream quality | grammar-template shortcut disabled by default; Stage B accepts and enforces generated-AST quality |
| benchmark proxy | proxy predictions now flow through the model forward result |
| joint loss | conversation and benchmark-proxy losses can enter the same batch objective |
| staged training | expanded parameter scopes isolate gaps, router, dynamics, proxy, conversation, and growth |

## CFM Correction

For interpolated state

```text
x_t = (1 - t) * noise + t * target
```

the endpoint now uses

```text
x_hat_1 = x_t + (1 - t) * v_theta(x_t, t)
```

The earlier `x_t + v` endpoint was wrong except at the final timestep. This directly
affected node, edge, lexical, gap, parseability, and parent heads because all were
predicted from the malformed endpoint.

## Stage-A Upstream

The token endpoint now predicts:

- node types;
- typed parent edges;
- lexical leaves;
- inter-leaf spacing and punctuation gaps;
- hierarchy levels;
- parent indices;
- template logits for diagnostics only;
- parseability.

The full source reconstruction path includes gaps and lexemes. `decode_generated_template`
requires an explicit opt-in and cannot silently substitute templates for AST synthesis.

## Stage-B Quality

Stage A exposes a confidence proxy. The two-stage pipeline passes this score into Stage
B. Stage B can:

- scale generated-AST conditioning by quality;
- enforce a minimum quality threshold;
- reject unqualified ASTs before result CFM;
- never accept target or teacher AST at inference.

## Growth Boundary

`ExpertResidualBank` supports:

- context-conditioned generated experts;
- fixed-pool MoE controls;
- random-frozen birth controls;
- disabled controls;
- final growth freezing.

All modes share the same trunk placement, so capability differences are not caused by a
different forward topology.

## Conversation Boundary

The model now learns whether to continue, whether a turn ends, response mode
(`text/code/json/ast`), code language, and code/JSON/AST validity. EOS and role-marker
rules remain active fallbacks; learned stopping is disabled unless configured.

## Evidence Boundary

No model capability result follows from this document. Required next experiments:

1. Stage-A token-AST structural and lexical gate;
2. Stage-B generated-AST execution;
3. dynamic vs fixed/equal-compute;
4. gap router vs monolithic/frozen-router;
5. generated vs fixed/random-frozen experts;
6. benchmark proxy vs external rank alignment.
