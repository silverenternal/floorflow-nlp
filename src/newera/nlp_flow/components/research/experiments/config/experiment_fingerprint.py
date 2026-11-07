"""Select source and test artifacts whose hashes identify an NLP experiment."""

from __future__ import annotations

from collections.abc import Mapping

LEGACY_IMPLEMENTATION_ARTIFACTS = (
    "project/scripts/experiments/research/run_nlp_structure_ablation.py",
    "src/newera/nlp_flow/components/research/experiments/studies/structure_ablation_runner.py",
    "src/newera/nlp_flow/components/research/experiments/config/experiment_deadline.py",
    "src/newera/nlp_flow/components/research/experiments/config/experiment_fingerprint.py",
    "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_runtime.py",
    "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_policy.py",
    "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_training.py",
    "src/newera/nlp_flow/components/generation/trees/actions/tree_action_selection.py",
    "src/newera/nlp_flow/components/generation/trees/runtime/tree_completion_runtime.py",
    "src/newera/nlp_flow/components/ast/diagnostics/validation/ast_completion_contract.py",
    "src/newera/nlp_flow/components/ast/grammar/core/ast_grammar.py",
    "project/configs/structured_generation/structure_matched_strict/matched_structure/nlp_structure_matched_v12.json",
    "project/configs/other/general/gpu_queue/nlp_gpu_experiment_queue_v13.json",
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
