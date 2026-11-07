# Zero-Tuning Runtime and Training Contract v1

## Principle

The target is zero *tuning*, not zero mathematical constants. A safety cap is
still required for memory, recursion, and malformed states, but it is not a
quality-selected operating point.

The model should decide its own:

- AST node count;
- token-endpoint length;
- Stage-A and Stage-B CFM steps;
- serial depth;
- parallel lane demand;
- generated-E birth and retention.

Configuration may declare a hard maximum, but not a per-task optimum.

## Implemented controls

| Budget | Runtime decision |
|---|---|
| AST node count | learned `node_count_head` from semantic prompt context |
| source length | learned `source_length_head`, decoded as expected token count |
| Stage-A NFE | velocity step RMS compared with learned endpoint uncertainty |
| Stage-B NFE | result-step RMS compared with learned endpoint uncertainty |
| serial depth | unified planner depth bias + learned halting |
| parallel lanes | unified planner lane logits + sparse lane routing |
| generated E | planner birth probability and verifier-grounded runtime store |
| model structure | validation Pareto search rather than hand-picked width/depth |

Training supports schedule-free AdamW, removing the fixed cosine horizon and
the need to retune the schedule when the training horizon changes.

## Evidence used

- AST-T5: structure-aware pretraining and AST-aware span corruption,
  [arXiv:2401.03003](https://arxiv.org/abs/2401.03003),
  [code](https://github.com/gonglinyuan/ast_t5).
- Abstract Syntax Networks: dynamically structured AST decoder,
  [arXiv:1704.07535](https://arxiv.org/abs/1704.07535).
- TreeDiff: AST-guided corruption for diffusion code generation,
  [arXiv:2508.01473](https://arxiv.org/abs/2508.01473).
- AnCoder/AnchorTree: AST anchors for diffusion code generation,
  [arXiv:2602.17688](https://arxiv.org/abs/2602.17688).
- Discrete Flow Matching: non-autoregressive discrete generation,
  [arXiv:2407.15595](https://arxiv.org/abs/2407.15595).
- Flow Matching Guide and Code:
  [arXiv:2412.06264](https://arxiv.org/abs/2412.06264),
  [code](https://github.com/facebookresearch/flow_matching).
- PonderNet: learned computation steps,
  [arXiv:2107.05407](https://arxiv.org/abs/2107.05407).
- Adaptive Computation Modules: per-token progressive learners and learned
  stopping, [arXiv:2312.10193](https://arxiv.org/abs/2312.10193).
- Token-selective adaptive depth without explicit depth regularization,
  [arXiv:2605.05222](https://arxiv.org/abs/2605.05222).
- AlgoPerf self-tuning ruleset and schedule-free winners,
  [arXiv:2502.15015](https://arxiv.org/abs/2502.15015),
  [schedule_free](https://github.com/facebookresearch/schedule_free).

These sources support the direction; they do not establish that our combined
architecture or generated-E mechanism is successful.

## Remaining tuning exposure

- Loss scalars still require learned uncertainty weighting or another
  end-to-end balancing mechanism.
- Competence top-k and lane top-k still have configured safety maxima.
- Confidence thresholds remain available for diagnostics and verified
  training gates, but the final target-free path does not use a tuned
  confidence threshold.
- Optimizer defaults remain numerical safety defaults until a fully
  learning-rate-free optimizer is evaluated in this codebase.

## Evidence boundary

This document defines the zero-tuning contract and implementation direction. It
does not claim zero hyperparameters, SOTA, emergence, or generated-E capability.
