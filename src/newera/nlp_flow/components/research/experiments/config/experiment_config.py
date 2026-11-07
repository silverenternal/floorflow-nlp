"""Experiment-spec adapters for compact NLP model construction."""

from __future__ import annotations

from newera.nlp_flow.engineering import ExperimentSpec
from newera.nlp_flow.model import NLPModelConfig

DEFAULT_GAP_NAMES = (
    "json_validity",
    "ast_structure",
    "python_execution",
    "rust_execution",
    "cpp_execution",
    "engineering_acceleration",
    "model_architecture",
    "language_retention",
    "compositional_emergence",
)


def candidate_config(spec: ExperimentSpec) -> NLPModelConfig:
    model = spec.payload["model"]
    target_parameters = int(model["target_parameters"])
    unified_competence = bool(
        model.get("enable_unified_competence", False)
    )
    return NLPModelConfig(
        vocab_size=int(model["vocab_size"]),
        hidden_size=int(model.get("hidden_size", 160)),
        num_layers=int(model.get("num_layers", 3)),
        intermediate_size=int(model.get("intermediate_size", 400)),
        max_seq_len=int(model["max_context_tokens"]),
        parameter_cap=min(23_000_000, int(target_parameters * 1.05)),
        enable_typed_channels=spec.arm == "candidate",
        enable_typed_fusion=spec.arm == "candidate",
        enable_two_stage_cfm=spec.arm == "candidate",
        enable_json_action_head=spec.arm == "candidate",
        enable_json_schema_key_head=bool(
            model.get("enable_json_schema_key_head", False)
        ),
        enable_expert_factory=(
            spec.arm == "candidate" and not unified_competence
        ),
        enable_emergence_objective=bool(
            model.get("enable_emergence_objective", False)
        ),
        emergence_basis_count=int(model.get("emergence_basis_count", 32)),
        emergence_active_basis=int(model.get("emergence_active_basis", 4)),
        emergence_reward_weight=float(
            model.get("emergence_reward_weight", 0.1)
        ),
        emergence_norm_envelope_ratio=float(
            model.get("emergence_norm_envelope_ratio", 1.5)
        ),
        enable_dynamic_stack=bool(model.get("enable_dynamic_stack", False)),
        dynamic_max_steps=int(model.get("dynamic_max_steps", 4)),
        dynamic_min_steps=int(model.get("dynamic_min_steps", 1)),
        dynamic_parallel_lanes=int(model.get("dynamic_parallel_lanes", 2)),
        dynamic_parallel_top_k=int(model.get("dynamic_parallel_top_k", 1)),
        dynamic_lane_rank=int(model.get("dynamic_lane_rank", 8)),
        enable_gap_ensemble=bool(model.get("enable_gap_ensemble", False)),
        gap_adapter_rank=int(model.get("gap_adapter_rank", 8)),
        gap_fusion_dim=int(model.get("gap_fusion_dim", 64)),
        gap_heads=int(model.get("gap_heads", 4)),
        gap_router_enabled=bool(model.get("gap_router_enabled", True)),
        gap_names=tuple(model.get("gap_names", DEFAULT_GAP_NAMES)),
        enable_benchmark_proxy=bool(
            model.get("enable_benchmark_proxy", False)
        ),
        proxy_test_output_classes=int(
            model.get("proxy_test_output_classes", 64)
        ),
        proxy_max_constraints=int(
            model.get("proxy_max_constraints", 16)
        ),
        proxy_retrieval_dim=int(model.get("proxy_retrieval_dim", 32)),
        enable_turn_embeddings=bool(
            model.get("enable_turn_embeddings", False)
        ),
        turn_type_count=int(model.get("turn_type_count", 5)),
        enable_task_metadata=bool(model.get("enable_task_metadata", False)),
        task_family_count=int(model.get("task_family_count", 4)),
        task_language_count=int(model.get("task_language_count", 4)),
        enable_conversation_control=bool(
            model.get("enable_conversation_control", False)
        ),
        use_learned_stop_at_inference=bool(
            model.get("use_learned_stop_at_inference", False)
        ),
        enable_expert_bank=bool(model.get("enable_expert_bank", False)),
        expert_bank_mode=str(model.get("expert_bank_mode", "generated")),
        expert_count=int(model.get("expert_count", 4)),
        expert_top_k=int(model.get("expert_top_k", 2)),
        enable_neuron_graph=bool(model.get("enable_neuron_graph", False)),
        neuron_graph_link_types=int(model.get("neuron_graph_link_types", 4)),
        neuron_graph_max_edges=(
            None
            if model.get("neuron_graph_max_edges") is None
            else int(model["neuron_graph_max_edges"])
        ),
        neuron_graph_candidate_budget=(
            None
            if model.get("neuron_graph_candidate_budget", 256) is None
            else int(model.get("neuron_graph_candidate_budget", 256))
        ),
        enable_neuron_module=bool(
            model.get("enable_neuron_module", False)
        ),
        neuron_local_size=int(model.get("neuron_local_size", 32)),
        enable_unified_competence=unified_competence,
        unified_cell_rank=int(model.get("unified_cell_rank", 8)),
        unified_top_k=int(model.get("unified_top_k", 3)),
        unified_planner_dim=int(model.get("unified_planner_dim", 64)),
        unified_generated_rank=int(model.get("unified_generated_rank", 16)),
        unified_generated_basis_count=int(
            model.get("unified_generated_basis_count", 4)
        ),
        unified_generated_expert_mode=str(
            model.get("unified_generated_expert_mode", "online")
        ),
        enable_runtime_competence=bool(
            model.get("enable_runtime_competence", False)
        ),
        runtime_competence_capacity=int(
            model.get("runtime_competence_capacity", 8)
        ),
        runtime_trigger_threshold=float(
            model.get("runtime_trigger_threshold", 0.5)
        ),
        runtime_promotion_trials=int(
            model.get("runtime_promotion_trials", 2)
        ),
        runtime_promotion_success_rate=float(
            model.get("runtime_promotion_success_rate", 0.75)
        ),
    )


__all__ = ["DEFAULT_GAP_NAMES", "candidate_config"]
