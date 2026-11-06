"""Select source and test artifacts whose hashes identify an NLP experiment."""

from __future__ import annotations

from collections.abc import Mapping

LEGACY_IMPLEMENTATION_ARTIFACTS = (
    "scripts/run_nlp_structure_ablation.py",
    "src/floorflow/nlp_flow/structure_ablation_runner.py",
    "src/floorflow/nlp_flow/experiment_deadline.py",
    "src/floorflow/nlp_flow/experiment_fingerprint.py",
    "src/floorflow/nlp_flow/prefix_tree_runtime.py",
    "src/floorflow/nlp_flow/prefix_tree_policy.py",
    "src/floorflow/nlp_flow/prefix_tree_training.py",
    "src/floorflow/nlp_flow/tree_action_selection.py",
    "src/floorflow/nlp_flow/tree_completion_runtime.py",
    "src/floorflow/nlp_flow/ast_completion_contract.py",
    "src/floorflow/nlp_flow/ast_grammar.py",
    "configs/nlp_structure_matched_v12.json",
    "configs/nlp_gpu_experiment_queue_v13.json",
)


def implementation_artifact_paths(config: Mapping[str, object]) -> tuple[str, ...]:
    configured = config.get("implementation_artifacts")
    if configured is None:
        return LEGACY_IMPLEMENTATION_ARTIFACTS
    if not isinstance(configured, list) or not all(isinstance(path, str) for path in configured):
        raise TypeError("implementation_artifacts must be a list of repository-relative paths")
    paths = tuple(configured)
    if len(paths) != len(set(paths)):
        raise ValueError("implementation_artifacts must not contain duplicate paths")
    return paths


__all__ = ["implementation_artifact_paths"]
