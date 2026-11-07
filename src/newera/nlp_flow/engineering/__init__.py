"""Engineering substrate for nlp_flow experiments."""

import importlib

from newera.nlp_flow.engineering.components.acceleration_artifacts_budget.artifacts import (
    build_jsonl_manifest,
    sha256_file,
    write_json_atomic,
)
from newera.nlp_flow.engineering.components.acceleration_artifacts_budget.budget import (
    BudgetReport,
    audit_parameter_budget,
    count_unique_trainable_parameters,
    parameter_breakdown,
)
from newera.nlp_flow.engineering.components.checkpoint_reproducibility_runtime.checkpoint import (
    load_nlp_checkpoint,
    restore_nlp_checkpoint,
    save_nlp_checkpoint,
)
from newera.nlp_flow.engineering.components.free_schedule_spec.schedule_free import ScheduleFreeAdamW
from newera.nlp_flow.engineering.components.free_schedule_spec.spec import ExperimentSpec


def __getattr__(name: str):
    """Load the training runtime only when a caller actually requests it."""

    if name in {"NLPRuntime", "build_runtime"}:
        runtime = importlib.import_module(
            "newera.nlp_flow.engineering.components.checkpoint_reproducibility_runtime.runtime"
        )
        return getattr(runtime, name)
    raise AttributeError(name)

__all__ = [
    "BudgetReport",
    "ExperimentSpec",
    "NLPRuntime",
    "ScheduleFreeAdamW",
    "audit_parameter_budget",
    "build_jsonl_manifest",
    "build_runtime",
    "count_unique_trainable_parameters",
    "load_nlp_checkpoint",
    "parameter_breakdown",
    "restore_nlp_checkpoint",
    "save_nlp_checkpoint",
    "sha256_file",
    "write_json_atomic",
]
