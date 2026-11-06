# NLP Data-Pipeline Architecture Audit v2

## Findings and structural repair

| Pipeline point | Previous weakness | Structural repair |
|---|---|---|
| Prompt to AST | SHA-derived random context had no prompt semantics | token-level semantic prompt encoder with language conditioning |
| Corpus metadata | family/language labels were discarded before the model | typed family/language embeddings on every token |
| AST node budget | target `node_mask` leaked into sampling | learned node-count head and target-free mask |
| AST categorical training | heads trained on mixed flow states containing targets | fixed `t=0` categorical endpoint pass |
| AST topology | independent parent outputs could create cycles or no root | fixed root slot, preceding-parent constraint, root head |
| AST endpoint | generic lexical concatenation ignored full structure | AST-aware causal token decoder bound to leaf-node states |
| Token decoder exposure bias | teacher forcing only | free-running consistency pass |
| Stage-B AST condition | pooled AST lost topology | level/parent-distance attention conditioner |
| Stage-B result field | pointwise low-rank field had no token mixing | result-token self-attention and FFN |
| CFM compute | fixed NFE | learned endpoint uncertainty stopping |

## AST-aware token endpoint

Each source token carries:

- its predicted AST leaf-node id;
- a phase id for special, whitespace gap, or lexical token;
- the corresponding generated node state.

The decoder enforces:

```
<bos> (gap, lexeme)* <eos>
```

and cannot terminate before producing a non-whitespace lexeme or before its
learned source-length budget allows it. This is output grammar, not parser
repair.

## Stage-A training

Training has three endpoint views:

1. flow matching over continuous AST states;
2. teacher-forcing AST-to-source token decoding;
3. free-running decoding using the model's own first-pass tokens.

The categorical loss uses a fixed-zero-time endpoint pass so the token/type
heads cannot exploit target information still present in a mixed flow state.

## Current limitation

The 1,500-step synthetic probe now fails honestly with non-empty but invalid
source endpoints. This is not a reason to add parser repair. It indicates that
the next experiment must move from three fixed samples to corpus-scale
AST/source pairs and compare:

- AST-aware token endpoint;
- Seq2Tree/abstract syntax network decoder;
- discrete flow matching over AST codec tokens.

The references motivating those alternatives are recorded in
`nlp_zero_hyperparameter_runtime_v1.md`.
