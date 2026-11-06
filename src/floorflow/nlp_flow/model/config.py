"""Validated compact decoder configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NLPModelConfig:
    """Small GPT-style mechanics without GPT-scale parameter growth."""

    vocab_size: int = 8192
    hidden_size: int = 160
    num_layers: int = 3
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 400
    max_seq_len: int = 512
    channel_size: int = 32
    parameter_cap: int = 2_300_000
    rope_base: float = 10_000.0
    norm_eps: float = 1.0e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    enable_typed_channels: bool = False
    enable_typed_fusion: bool = False
    enable_two_stage_cfm: bool = False
    enable_json_action_head: bool = False
    enable_json_schema_key_head: bool = False
    enable_expert_factory: bool = False
    enable_emergence_objective: bool = False
    emergence_basis_count: int = 32
    emergence_active_basis: int = 4
    emergence_reward_weight: float = 0.1
    emergence_norm_envelope_ratio: float = 1.5
    enable_dynamic_stack: bool = False
    dynamic_max_steps: int = 4
    dynamic_min_steps: int = 1
    dynamic_parallel_lanes: int = 2
    dynamic_parallel_top_k: int = 1
    dynamic_lane_rank: int = 8
    enable_gap_ensemble: bool = False
    gap_adapter_rank: int = 8
    gap_fusion_dim: int = 64
    gap_heads: int = 4
    gap_router_enabled: bool = True
    gap_names: tuple[str, ...] = (
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
    enable_benchmark_proxy: bool = False
    proxy_test_output_classes: int = 64
    proxy_max_constraints: int = 16
    proxy_retrieval_dim: int = 32
    enable_turn_embeddings: bool = False
    turn_type_count: int = 5
    enable_task_metadata: bool = False
    task_family_count: int = 4
    task_language_count: int = 4
    enable_conversation_control: bool = False
    use_learned_stop_at_inference: bool = False
    enable_expert_bank: bool = False
    expert_bank_mode: str = "generated"
    expert_count: int = 4
    expert_top_k: int = 2
    enable_neuron_module: bool = False
    neuron_local_size: int = 32
    enable_unified_competence: bool = False
    unified_cell_rank: int = 8
    unified_top_k: int = 3
    unified_planner_dim: int = 64
    unified_generated_rank: int = 16
    unified_generated_basis_count: int = 4
    unified_generated_expert_mode: str = "online"
    enable_runtime_competence: bool = False
    runtime_competence_capacity: int = 8
    runtime_trigger_threshold: float = 0.5
    runtime_promotion_trials: int = 2
    runtime_promotion_success_rate: float = 0.75

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "intermediate_size",
            "max_seq_len",
            "channel_size",
            "parameter_cap",
        )
        for field in positive:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.rope_base <= 0.0:
            raise ValueError("rope_base must be positive")
        if not 0.0 <= self.norm_eps < 1.0e-2:
            raise ValueError("norm_eps is outside the supported range")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.parameter_cap > 23_000_000:
            raise ValueError("primary model cap cannot exceed 23M parameters")
        if not isinstance(self.enable_typed_channels, bool):
            raise TypeError("enable_typed_channels must be bool")
        if not isinstance(self.enable_typed_fusion, bool):
            raise TypeError("enable_typed_fusion must be bool")
        if not isinstance(self.enable_two_stage_cfm, bool):
            raise TypeError("enable_two_stage_cfm must be bool")
        if not isinstance(self.enable_json_action_head, bool):
            raise TypeError("enable_json_action_head must be bool")
        if not isinstance(self.enable_json_schema_key_head, bool):
            raise TypeError("enable_json_schema_key_head must be bool")
        if not isinstance(self.enable_expert_factory, bool):
            raise TypeError("enable_expert_factory must be bool")
        if not isinstance(self.enable_emergence_objective, bool):
            raise TypeError("enable_emergence_objective must be bool")
        if not isinstance(self.enable_dynamic_stack, bool):
            raise TypeError("enable_dynamic_stack must be bool")
        if not isinstance(self.enable_gap_ensemble, bool):
            raise TypeError("enable_gap_ensemble must be bool")
        if not isinstance(self.enable_benchmark_proxy, bool):
            raise TypeError("enable_benchmark_proxy must be bool")
        for field in (
            "enable_turn_embeddings",
            "enable_task_metadata",
            "enable_conversation_control",
            "enable_expert_bank",
            "enable_neuron_module",
            "enable_unified_competence",
            "enable_runtime_competence",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be bool")
        if not isinstance(self.use_learned_stop_at_inference, bool):
            raise TypeError("use_learned_stop_at_inference must be bool")
        if self.use_learned_stop_at_inference and not self.enable_conversation_control:
            raise ValueError(
                "learned stopping requires the conversation control head"
            )
        if self.enable_typed_fusion and not self.enable_typed_channels:
            raise ValueError("typed fusion requires typed channels")
        if self.enable_two_stage_cfm and not self.enable_typed_channels:
            raise ValueError("two-stage CFM requires typed channels")
        if self.enable_json_schema_key_head and not self.enable_json_action_head:
            raise ValueError("schema key head requires the JSON action head")
        if self.enable_expert_factory and not self.enable_typed_channels:
            raise ValueError("expert factory requires typed channels")
        if self.emergence_basis_count <= 0:
            raise ValueError("emergence_basis_count must be positive")
        if not 0 < self.emergence_active_basis <= self.emergence_basis_count:
            raise ValueError("emergence_active_basis must be within the basis bank")
        if self.emergence_reward_weight < 0.0:
            raise ValueError("emergence_reward_weight must be non-negative")
        if self.emergence_norm_envelope_ratio <= 0.0:
            raise ValueError("emergence_norm_envelope_ratio must be positive")
        if self.dynamic_max_steps <= 0:
            raise ValueError("dynamic_max_steps must be positive")
        if self.dynamic_min_steps <= 0:
            raise ValueError("dynamic_min_steps must be positive")
        if self.dynamic_min_steps > self.dynamic_max_steps:
            raise ValueError("dynamic_min_steps cannot exceed dynamic_max_steps")
        if self.dynamic_parallel_lanes <= 0:
            raise ValueError("dynamic_parallel_lanes must be positive")
        if not 0 < self.dynamic_parallel_top_k <= self.dynamic_parallel_lanes:
            raise ValueError(
                "dynamic_parallel_top_k must be within the configured lanes"
            )
        if self.dynamic_lane_rank <= 0:
            raise ValueError("dynamic_lane_rank must be positive")
        if self.gap_adapter_rank <= 0:
            raise ValueError("gap_adapter_rank must be positive")
        if self.gap_fusion_dim <= 0 or self.gap_heads <= 0:
            raise ValueError("gap fusion dimensions must be positive")
        if self.gap_fusion_dim % self.gap_heads:
            raise ValueError("gap_fusion_dim must be divisible by gap_heads")
        if not isinstance(self.gap_router_enabled, bool):
            raise TypeError("gap_router_enabled must be bool")
        if not self.gap_names or len(set(self.gap_names)) != len(self.gap_names):
            raise ValueError("gap_names must be non-empty and unique")
        if self.proxy_test_output_classes <= 0:
            raise ValueError("proxy_test_output_classes must be positive")
        if self.proxy_max_constraints <= 0:
            raise ValueError("proxy_max_constraints must be positive")
        if self.proxy_retrieval_dim <= 0:
            raise ValueError("proxy_retrieval_dim must be positive")
        if self.turn_type_count <= 0:
            raise ValueError("turn_type_count must be positive")
        if self.task_family_count <= 0 or self.task_language_count <= 0:
            raise ValueError("task metadata dimensions must be positive")
        if self.expert_bank_mode not in {
            "generated",
            "fixed_pool",
            "random_frozen",
            "disabled",
        }:
            raise ValueError("unsupported expert_bank_mode")
        if self.expert_count <= 0 or not 0 < self.expert_top_k <= self.expert_count:
            raise ValueError("expert routing dimensions are invalid")
        if self.neuron_local_size <= 0:
            raise ValueError("neuron_local_size must be positive")
        if self.unified_cell_rank <= 0 or self.unified_planner_dim <= 0:
            raise ValueError("unified competence dimensions must be positive")
        if self.unified_top_k <= 0:
            raise ValueError("unified_top_k must be positive")
        if self.unified_top_k > len(self.gap_names):
            raise ValueError("unified_top_k cannot exceed gap_names")
        if self.unified_generated_rank <= 0 or self.unified_generated_basis_count <= 0:
            raise ValueError("unified generated-expert dimensions must be positive")
        if self.unified_generated_expert_mode not in {"online", "disabled"}:
            raise ValueError("unsupported unified_generated_expert_mode")
        if self.runtime_competence_capacity <= 0:
            raise ValueError("runtime_competence_capacity must be positive")
        if not 0.0 <= self.runtime_trigger_threshold <= 1.0:
            raise ValueError("runtime trigger threshold must be in [0, 1]")
        if self.runtime_promotion_trials <= 0:
            raise ValueError("runtime promotion trials must be positive")
        if not 0.0 <= self.runtime_promotion_success_rate <= 1.0:
            raise ValueError("runtime promotion success rate must be in [0, 1]")
        if self.enable_runtime_competence and not (
            self.enable_unified_competence or self.enable_expert_factory
        ):
            raise ValueError(
                "runtime competence requires a generated expert factory"
            )
        if (
            self.enable_runtime_competence
            and self.enable_unified_competence
            and self.unified_generated_expert_mode == "disabled"
        ):
            raise ValueError(
                "runtime competence requires online generated experts"
            )
        if self.enable_unified_competence and any(
            (
                self.enable_gap_ensemble,
                self.enable_expert_bank,
                self.enable_neuron_module,
            )
        ):
            raise ValueError(
                "unified competence cannot be combined with legacy "
                "gap/expert/neuron modules"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["NLPModelConfig"]
