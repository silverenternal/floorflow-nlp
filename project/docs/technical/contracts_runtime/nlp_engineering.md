# NLP Engineering Layer v1

Status date: 2026-11-01.

The engineering layer is a thin nlp_flow adapter over proven floorplan and CHIP
training infrastructure.

## Reused designs

From the floorplan line:

- `RuntimeConfig` precision/autocast/compile/channels-last controls;
- conservative CPU fallback and Triton status reporting;
- metric accumulators and training metric conventions;
- deterministic seeding patterns.

From the CHIP line:

- atomic Torch and JSON writes;
- full Python/NumPy/Torch/CUDA RNG checkpoint state;
- parameter-tier and no-decay optimizer grouping patterns;
- latent/parameter probe and gradient-health conventions;
- frozen data/cache manifest concepts.

The nlp_flow layer adds only domain-specific pieces:

- canonical experiment spec and content hash;
- exact unique-trainable parameter accounting;
- 20:1 corpus-token/parameter audit;
- JSONL split manifest with SHA-256 and row counts;
- checkpoint provenance for spec, tokenizer, and data manifest;
- one runtime object that training and evaluation entry points share.

## Components

```text
src/newera/nlp_flow/engineering/
  spec.py        experiment JSON validation and hash
  budget.py      unique trainable parameters and 20:1 audit
  artifacts.py   JSONL manifest and file hashing
  runtime.py     device/precision/determinism wrapper
  checkpoint.py  atomic resumable checkpoint with full RNG/provenance
```

The initial spec is
`project/configs/other/contracts_runtime/contracts_control_engineering/nlp_engineering_p2m_control.json`. It is deliberately a template:
the tokenizer and benchmark hashes remain `null`, and GPU is not authorized.

## Contract

An experiment cannot enter the primary run unless:

1. the spec validates and has a stable SHA-256;
2. all train/validation/test JSONL files exist, contain `id` and `text`, and
   have distinct hashes;
3. unique trainable parameters equal planned corpus tokens / 20 within 2%;
4. tokenizer and benchmark SHA-256 values are present;
5. the checkpoint stores full resume RNG and provenance;
6. the active runtime is recorded in its manifest.

The template currently stops before step 4. That is intentional and prevents an
accidental pre-tokenizer GPU run.

Audit:

```bash
uv run python project/scripts/audit/engineering/runtime/audit_nlp_engineering.py \
  --config project/configs/other/contracts_runtime/contracts_control_engineering/nlp_engineering_p2m_control.json \
  --output outputs/reports/nlp_engineering_audit_v1.json
uv run pytest project/tests/other/contracts_runtime/engineering_compute_executors/test_nlp_engineering.py -q
```
