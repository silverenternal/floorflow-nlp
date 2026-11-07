# NewEra

NewEra is a research project exploring learnable, composable neuron-like building blocks toward broadly capable models, with NLP and structured code generation as research domains. It contains grammar-aware AST prediction, structured output contracts, training and evaluation runners, experiment configurations, and focused tests. See the [source notice](docs/research/NOTICE.md) for provenance and licensing information.

This is research software. It does not include datasets, model weights, execution sandboxes, or generated reports, and it makes no capability or benchmark qualification claim. The historical strict AST result remains 0/60; recent matched experiments have not established a qualified generation model.

## Repository layout

- `src/newera/nlp_flow/` — reusable NLP and structured generation modules
- `project/scripts/` — runnable workflows grouped into `data`, `experiments`, `evaluation`, `audit`, and `applications`
- `project/configs/` — configuration contracts grouped by domain
- `project/tests/` — unit and contract tests grouped by domain
- `project/docs/` — design notes grouped by domain; see the [documentation index](docs/README.md)

Files are grouped by their purpose, with at most five direct entries per formal directory; exploratory model/component directories warn on overflow. Core modules under `src/newera/nlp_flow/components/` are organized into `ast`, `generation`, `research`, and `platform`; the model package separates architecture, experts, task interfaces, and training. For example, test and configuration directories distinguish structured generation, data, evaluation, and training. Data scripts distinguish AST, corpus, sources, generation, and benchmarks; experiment scripts distinguish training, replay, validation, probes, and research. Script paths include their workflow group, e.g. `python project/scripts/audit/engineering/maintenance/audit_nlp_code_sizes.py`. Run scripts from the repository root after installation. Some experiment configurations refer to external data or artifacts that are not included.

The [research direction](docs/technical/governance/research_direction.md) distinguishes this long-term hypothesis from demonstrated model capability. The [development roadmap](docs/research/todo.json) tracks the next steps and their acceptance criteria.

## Engineering protocol

The proposed [model-training project protocol](docs/technical/governance/training_project_protocol.md) defines the intended data, experiment, training, evaluation, and release practices. The [issue response matrix](docs/technical/governance/issue_response_matrix.md) records the disposition and implementation status of all 30 review items. It is a draft for review and does not claim that all controls are implemented.

## Setup

Python 3.11 or newer is required. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
# Optional pretrained-model runners:
uv pip install -e ".[dev,pretrained]"
```

The installed package is imported as `newera.nlp_flow`.

## Data and checkpoints

No datasets or model checkpoints are included. Some runners expect local files under `.data/nlp/` and write generated results under `outputs/`; those locations are ignored by Git. Follow the relevant design note and configuration contract before providing data or launching a run. Never use test-split content for training, selection, or debugging.

Several data-backed tests also require local corpus fixtures or registered task artifacts. The test files document these requirements; a clean checkout does not include the private or locally mounted data they expect.

## Policy checks

Run the same dependency-free checks used by GitHub Actions:

```bash
PYTHONPATH=src python -m newera.gates --stage explore
PYTHONPATH=src python -m newera.gates --stage research --evidence path/to/run.json
PYTHONPATH=src python -m newera.gates --stage release --evidence path/to/release.json
# Optionally run a command after checks (no shell is used):
PYTHONPATH=src python -m newera.gates --stage research --evidence path/to/run.json --execute python project/scripts/experiments/training/models/train_nlp_compact.py --help
```

After installing the package, `newera-gate` is an equivalent entry point. Results are JSON: formal directory or syntax errors fail; exploratory directory overflow and missing research/release evidence warn without blocking. Warnings do not establish qualification, authorize training or release, or bypass existing specialized gates. For research/release, `qualified` is always `false`: this tool checks presence only and cannot verify human judgments or legal compliance.

A research evidence JSON may contain `run_id`, `config` (file path), `data_version`, `data_fingerprint`, `code_revision`, `environment`, `started_at`, integer `seed`, `budget`, `stop_conditions`, and `evaluation_protocol` (file path). A release manifest also names `model_card`, `evaluation_report`, `security_review`, `fairness_review`, and `legal_review` files. For each file field, supply a corresponding `<field>_sha256`; missing or mismatched digests are warnings. Relative evidence file paths are resolved from the repository root. These fields are minimum presence checks, not the complete protocol.

The GitHub Actions workflow runs the full CPU test suite, gate unit tests, targeted Ruff checks, repository structure checks, Gitleaks history scanning and a pip-audit dependency audit. The CI test job uses Python 3.13; data-dependent tests requiring local legacy artifacts remain explicitly skipped. The narrow Gitleaks false-positive allowlist is in `.github/gitleaks.toml` and covers only known tokenizer SHA-256 fields and one import line; it does not allowlist whole files. The action versions are pinned, but the workflow has not yet been verified on GitHub. The dependency audit resolves declared core dependencies and does not establish coverage of optional extras or a separately installed environment. GPU workflows are not run in CI. The optional `--execute` wrapper runs a command only after structural checks pass, but direct calls to existing training scripts do not automatically invoke it; existing specialized gates remain in effect.

## Development

```bash
.venv/bin/pytest
.venv/bin/ruff check .
python -m compileall -q src project
```

GPU workflows require a separately reviewed configuration, a compatible CUDA PyTorch build, exclusive device availability, and their registered compute budgets. CPU preflight should pass before any GPU run.

## Project status

The code preserves strict historical evaluation gates and negative results. A public code snapshot should not be read as evidence that a model passed those gates. Experiment registrations and local outputs describe research state, not a supported production interface.

## License

The source project did not contain a license file, so no license is granted by this extraction. Contact the repository owner before reusing or redistributing the code.
