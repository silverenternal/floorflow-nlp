# Open-Capability Exploratory Track v1

## Decision

The compressed-primary model remains the final smallest/SOTA target. To avoid
premature compression limiting model quality, capability work now proceeds in a
separate open-parameter exploratory track.

An open-capability checkpoint is:

- explicitly labelled un-compressed;
- evaluated with resident parameters, not active or effective parameters;
- excluded from primary smallest/SOTA claims;
- frozen with a complete tensor ledger before compression begins.

## Immediate Architecture

The first open-capability arm uses the existing compact hierarchical decoder with:

- 320 hidden size, six transformer layers, and 1152 SwiGLU intermediate size;
- 512-token context and the frozen 8192-token tokenizer;
- typed channels, typed fusion, JSON action head, generated-expert interface, and
  two-stage AST CFM;
- the full P12M corpus manifest, with test data still closed during training.

The initial target is approximately 11.5M parameters. This is a capability experiment,
not a claim that the model should remain at that size.

## Quality Order

The optimization priority is:

1. valid two-stage AST and executable Python/Rust/C++ behavior;
2. native JSON legality and schema compliance without repair or retry;
3. Triton/Linux/engineering diagnostic capability;
4. multi-turn instruction following and language retention;
5. efficiency at fixed quality.

Every checkpoint is evaluated without best-of-N, answer repair, hidden retries, or
test-data-based checkpoint selection.

## Compression Handoff

After quality plateaus, export:

- the final model-only checkpoint;
- every resident tensor with shape, dtype, and compression grouping;
- training and evaluation hashes;
- public and held-out executable suites;
- fixed-depth and fixed-compute controls.

Tensor-ring, low-rank, quantized, and hybrid compression are evaluated on the frozen
checkpoint. A compressed result must rerun the same capability gates and report any
quality loss before it can enter the primary smallest-model claim.
