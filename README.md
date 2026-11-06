# FloorFlow NLP

A standalone extraction of the NLP and structured code-generation research package from [FloorFlow](https://github.com/silverenternal/floorflow). It contains grammar-aware AST prediction, structured JSON/output contracts, training and evaluation runners, experiment configurations, and focused tests.

This is research software. It does not include datasets, model weights, execution sandboxes, or generated reports, and it makes no capability or benchmark qualification claim. The historical strict AST result remains 0/60; recent matched experiments have not established a qualified generation model.

## Repository layout

- `src/floorflow/nlp_flow/` — reusable NLP and structured generation modules
- `scripts/` — NLP training, evaluation, data, and audit workflows
- `configs/` — grammar, task, and experiment contracts
- `tests/` — NLP unit and contract tests
- `docs/` — design notes and research summaries

## Setup

Python 3.11 or newer is required. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
# Optional pretrained-model runners:
uv sync --extra dev --extra pretrained
```

The installed package is imported as `floorflow.nlp_flow`.

## Data and checkpoints

No datasets or model checkpoints are included. Some runners expect local files under `.data/nlp/` and write generated results under `outputs/`; those locations are ignored by Git. Follow the relevant design note and configuration contract before providing data or launching a run. Never use test-split content for training, selection, or debugging.

Several data-backed tests also require local corpus fixtures or registered task artifacts. The test files document these requirements; a clean checkout does not include the private or locally mounted data they expect.

## Development

```bash
uv run pytest
uv run ruff check .
python -m compileall -q src scripts tests
```

GPU workflows require a separately reviewed configuration, a compatible CUDA PyTorch build, exclusive device availability, and their registered compute budgets. CPU preflight should pass before any GPU run.

## Project status

The code preserves strict historical evaluation gates and negative results. A public code snapshot should not be read as evidence that a model passed those gates. Experiment registrations and local outputs describe research state, not a supported production interface.

## License

The source project did not contain a license file, so no license is granted by this extraction. Contact the repository owner before reusing or redistributing the code.
