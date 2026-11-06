# Training and Inference Information Audit v1

## Scope

This audit follows information from corpus rows through tokenization, target masks,
model forward paths, auxiliary losses, checkpoint selection, and autoregressive
generation. It is a structural audit; no new capability experiment is claimed.

## Training Information Losses Found and Fixed

### Multi-turn roles collapsed

Previously, historical user/assistant markers were tokenized but every prompt token
received the same `user` turn id. The row encoder now parses `User/Assistant/Tool/Code/
System` spans from tokenizer offsets and assigns their actual turn ids.

### Long-prompt truncation could delete roles and EOS

The encoder previously truncated prefix/response independently. A long response could
drop its EOS and long prompts could lose all structural context. The new policy:

- preserves the full response and EOS;
- skips rows whose response alone cannot fit, rather than training a corrupted target;
- retains prompt head and latest prompt tail when trimming;
- carries labels, turn ids, control targets, and validity masks through truncation.

### Fine-grained supervision was missing

Row-aligned samples now emit:

- answer-only labels;
- continuation targets;
- turn-boundary targets;
- reply-mode targets;
- language targets;
- code/JSON/AST validity targets and masks;
- compile/execution/runtime/calibration proxy targets.

### Auxiliary losses were absent from checkpoint selection

Validation previously used token CE alone. Validation and restore checks now use the
same emergence, dynamic-depth, conversation-control, and benchmark-proxy weights as
training.

### Executable multilingual rows lacked labels

Generated Rust and C++ templates now carry execution-valid labels, so proxy heads no
longer treat known runnable examples as unknown.

### Emergence score was auxiliary-only

The former emergence objective computed validity and novelty diagnostics but did not
expose a normalized score or guarantee that a reward term reached the optimizer.

The joint trainer now:

- computes a bounded `emergence_score`;
- gates novelty by reconstruction, cycle, support recall, validity, and activation norm;
- subtracts `reward_weight * emergence_score` from the task objective;
- includes emergence score/components in training history and reports;
- enables the objective in `configs/nlp_open_emergence_joint_v1.json`.

### AST state information was missing

Stage-A omitted spacing/punctuation gaps and used an incorrect endpoint formula. Both
are fixed:

- endpoint uses remaining-time velocity;
- lexemes and gaps are embedded, predicted, trained, and decoded;
- grammar-template retrieval is disabled by default.

### Model branches were disconnected

Generated/fixed/random-frozen experts, neurons, gap routing, proxy heads, and
conversation heads were exposed separately but not all were in the main forward graph.
They now have explicit zero-gated or diagnostic forward placement.

## Inference Quality Issues Found and Fixed

### Train/inference turn mismatch

Generation previously ignored turn embeddings even when the model used them. Prompt
generation now reconstructs turn ids from role markers using the same role parser as
training.

### Context trimming lost role alignment

Generation now trims ids and turn ids together, retaining the prompt head and latest
conversation tail. The current user request and final assistant marker remain in the
final segment.

### Zero-gated control leaks

Turn embeddings, neuron residuals, expert birth gates, and the gap aggregator begin as
identity paths. Enabling optional architecture therefore does not perturb the base
model at initialization.

### Output-validity controls were unused

The model now exposes learned continuation, turn boundary, reply mode, language, and
code/JSON/AST validity logits. Learned stopping is opt-in and only stops after at least
one generated token.

### AST-to-token inference path was missing

The model now provides prompt-context encoding for CFM, Stage-A/Stage-B sampling, and a
token-logit projection from the generated result state.

## Remaining Experimental Questions

The structural fixes do not establish:

- token-endpoint AST quality;
- executable C++ quality;
- proxy-to-benchmark correlation;
- dynamic/gap/growth gains at matched compute;
- SOTA coding capability.

Those remain explicit gates in `nlptodo.json`.
