"""Compact hierarchical decoder skeleton with tied output and typed heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from floorflow.nlp_flow.model.benchmark_proxy import (
    BenchmarkProxyConfig,
    BenchmarkProxyObjective,
    BenchmarkProxyPrediction,
)
from floorflow.nlp_flow.model.blocks import CompactDecoderBlock, RMSNorm
from floorflow.nlp_flow.model.cfm import TwoStageASTCFM
from floorflow.nlp_flow.model.channels import TypedChannelMemory, TypedChannelMemoryEncoder
from floorflow.nlp_flow.model.competence import (
    UnifiedCompetenceBank,
    UnifiedCompetenceConfig,
)
from floorflow.nlp_flow.model.config import NLPModelConfig
from floorflow.nlp_flow.model.conversation import (
    ConversationControlHead,
    TurnSegmentEmbedding,
)
from floorflow.nlp_flow.model.dynamic_stack import (
    DynamicStackConfig,
    DynamicStackDecoder,
)
from floorflow.nlp_flow.model.emergence import (
    EmergenceConfig,
    EmergentSemanticObjective,
)
from floorflow.nlp_flow.model.expert_bank import (
    ExpertBankConfig,
    ExpertResidualBank,
)
from floorflow.nlp_flow.model.expert_factory import (
    ContextConditionedExpertFactory,
    GeneratedExpertState,
    MaterializedExpert,
)
from floorflow.nlp_flow.model.fusion import TypedChannelFusion
from floorflow.nlp_flow.model.gap_ensemble import GapEnsemble, GapEnsembleConfig
from floorflow.nlp_flow.model.json_grammar import (
    JSONActionHead,
    JSONGrammarState,
    JSONSchemaKeyHead,
    JSONTypedHeads,
)
from floorflow.nlp_flow.model.neuron_module import NeuronModule
from floorflow.nlp_flow.model.runtime_competence import (
    RuntimeCompetenceConfig,
    RuntimeCompetenceManager,
)
from floorflow.nlp_flow.model.runtime_model_api import RuntimeModelMixin


@dataclass(frozen=True)
class NLPModelOutput:
    hidden_states: torch.Tensor
    token_logits: torch.Tensor
    typed_channels: dict[str, torch.Tensor]
    typed_memory: TypedChannelMemory | None = None
    fusion_diagnostics: dict[str, torch.Tensor] | None = None
    dynamic_diagnostics: dict[str, torch.Tensor | int] | None = None
    gap_diagnostics: dict[str, torch.Tensor] | None = None
    gap_logits: dict[str, torch.Tensor] | None = None
    gap_states: dict[str, torch.Tensor] | None = None
    benchmark_proxy: BenchmarkProxyPrediction | None = None
    conversation_logits: dict[str, torch.Tensor] | None = None
    growth_diagnostics: dict[str, torch.Tensor | str] | None = None
    tree_mode_logits: torch.Tensor | None = None
    runtime_synthesis_score: torch.Tensor | None = None


class CompactNLPDecoder(RuntimeModelMixin, nn.Module):
    """Modern compact token core; CFM and typed fusion attach as residuals later."""

    def __init__(self, config: NLPModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.turn_embedding = (
            TurnSegmentEmbedding(config.hidden_size, config.turn_type_count)
            if config.enable_turn_embeddings
            else None
        )
        self.task_family_embedding = (
            nn.Embedding(config.task_family_count, config.hidden_size)
            if config.enable_task_metadata
            else None
        )
        self.task_language_embedding = (
            nn.Embedding(config.task_language_count, config.hidden_size)
            if config.enable_task_metadata
            else None
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = (
            nn.ModuleList()
            if config.enable_dynamic_stack
            else nn.ModuleList(
                [CompactDecoderBlock(config) for _ in range(config.num_layers)]
            )
        )
        self.dynamic_stack = (
            DynamicStackDecoder(
                DynamicStackConfig(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    intermediate_size=config.intermediate_size,
                    max_seq_len=config.max_seq_len,
                    max_serial_steps=config.dynamic_max_steps,
                    min_serial_steps=config.dynamic_min_steps,
                    parallel_lanes=config.dynamic_parallel_lanes,
                    parallel_top_k=config.dynamic_parallel_top_k,
                    lane_rank=config.dynamic_lane_rank,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                )
            )
            if config.enable_dynamic_stack
            else None
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.typed_memory = (
            TypedChannelMemoryEncoder(config.hidden_size, config.channel_size)
            if config.enable_typed_channels
            else None
        )
        self.typed_fusion = (
            TypedChannelFusion(config.hidden_size)
            if config.enable_typed_fusion
            else None
        )
        self.cfm = TwoStageASTCFM(config.hidden_size) if config.enable_two_stage_cfm else None
        self.json_head = (
            JSONActionHead(config.hidden_size)
            if config.enable_json_action_head
            else None
        )
        self.json_typed_heads = (
            JSONTypedHeads(config.hidden_size)
            if config.enable_json_action_head
            else None
        )
        self.json_schema_key_head = (
            JSONSchemaKeyHead(config.hidden_size)
            if config.enable_json_schema_key_head
            else None
        )
        self.expert_factory = (
            ContextConditionedExpertFactory(config.hidden_size)
            if config.enable_expert_factory and not config.enable_expert_bank
            else None
        )
        self.expert_bank = (
            ExpertResidualBank(
                ExpertBankConfig(
                    hidden_size=config.hidden_size,
                    mode=config.expert_bank_mode,
                    expert_count=config.expert_count,
                    top_k=config.expert_top_k,
                )
            )
            if config.enable_expert_bank
            else None
        )
        self.neuron = (
            NeuronModule(config.hidden_size, config.neuron_local_size)
            if config.enable_neuron_module
            else None
        )
        self.unified_competence = (
            UnifiedCompetenceBank(
                UnifiedCompetenceConfig(
                    hidden_size=config.hidden_size,
                    competence_names=config.gap_names,
                    cell_rank=config.unified_cell_rank,
                    top_k=config.unified_top_k,
                    planner_dim=config.unified_planner_dim,
                    generated_rank=config.unified_generated_rank,
                    generated_basis_count=(
                        config.unified_generated_basis_count
                    ),
                    max_depth_steps=config.dynamic_max_steps,
                    parallel_lanes=config.dynamic_parallel_lanes,
                    generated_expert_mode=config.unified_generated_expert_mode,
                )
            )
            if config.enable_unified_competence
            else None
        )
        runtime_factory = (
            self.unified_competence.generated_expert
            if (
                self.unified_competence is not None
                and self.unified_competence.generated_expert is not None
            )
            else self.expert_factory
        )
        if config.enable_runtime_competence and runtime_factory is None:
            raise RuntimeError("runtime competence requires an expert factory")
        self.runtime_competence = (
            RuntimeCompetenceManager(
                runtime_factory,
                RuntimeCompetenceConfig(
                    hidden_size=config.hidden_size,
                    capacity=config.runtime_competence_capacity,
                    trigger_threshold=config.runtime_trigger_threshold,
                    promotion_trials=config.runtime_promotion_trials,
                    promotion_success_rate=(
                        config.runtime_promotion_success_rate
                    ),
                ),
            )
            if config.enable_runtime_competence and runtime_factory is not None
            else None
        )
        self.emergence = (
            EmergentSemanticObjective(
                EmergenceConfig(
                    hidden_size=config.hidden_size,
                    basis_count=config.emergence_basis_count,
                    active_basis=config.emergence_active_basis,
                    reward_weight=config.emergence_reward_weight,
                    norm_envelope_ratio=config.emergence_norm_envelope_ratio,
                )
            )
            if config.enable_emergence_objective
            else None
        )
        self.gap_ensemble = (
            GapEnsemble(
                GapEnsembleConfig(
                    hidden_size=config.hidden_size,
                    enabled_gaps=config.gap_names,
                    adapter_rank=config.gap_adapter_rank,
                    fusion_dim=config.gap_fusion_dim,
                    heads=config.gap_heads,
                )
            )
            if config.enable_gap_ensemble
            else None
        )
        self.benchmark_proxy = (
            BenchmarkProxyObjective(
                BenchmarkProxyConfig(
                    hidden_size=config.hidden_size,
                    test_output_classes=config.proxy_test_output_classes,
                    max_constraints=config.proxy_max_constraints,
                    retrieval_dim=config.proxy_retrieval_dim,
                )
            )
            if config.enable_benchmark_proxy
            else None
        )
        self.conversation_head = (
            ConversationControlHead(config.hidden_size)
            if config.enable_conversation_control
            else None
        )
        self.output = None if config.tie_embeddings else nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if self.output is not None:
            nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        channel_masks: dict[str, torch.Tensor] | None = None,
        turn_ids: torch.Tensor | None = None,
        family_ids: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> NLPModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured context length")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain an out-of-vocabulary id")
        hidden = self.embedding_dropout(self.embedding(input_ids))
        if turn_ids is not None:
            if turn_ids.shape != input_ids.shape:
                raise ValueError("turn_ids must match input_ids")
            if self.turn_embedding is None:
                raise RuntimeError("turn ids were provided but turn embeddings are disabled")
            if turn_ids.min() < 0 or turn_ids.max() >= self.config.turn_type_count:
                raise ValueError("turn_ids contain an out-of-range type")
            hidden = hidden + self.turn_embedding(turn_ids)
        if family_ids is not None or language_ids is not None:
            if family_ids is None or language_ids is None:
                raise ValueError(
                    "family_ids and language_ids must be provided together"
                )
            if self.task_family_embedding is None or self.task_language_embedding is None:
                raise RuntimeError("task metadata embeddings are disabled")
            if family_ids.shape != input_ids.shape or language_ids.shape != input_ids.shape:
                raise ValueError("task metadata ids must match input_ids")
            hidden = (
                hidden
                + self.task_family_embedding(family_ids.to(input_ids.device))
                + self.task_language_embedding(language_ids.to(input_ids.device))
            )
        unified_depth_bias = (
            self.unified_competence.depth_bias(hidden)
            if self.unified_competence is not None
            else None
        )
        unified_lane_bias = (
            self.unified_competence.lane_bias(hidden)
            if self.unified_competence is not None
            else None
        )
        dynamic_diagnostics = None
        if self.dynamic_stack is not None:
            dynamic = self.dynamic_stack(
                hidden,
                attention_mask=attention_mask,
                halt_bias=unified_depth_bias,
                lane_bias=unified_lane_bias,
            )
            hidden = dynamic.hidden_states
            dynamic_diagnostics = {
                "mean_depth": dynamic.mean_depth,
                "max_depth": dynamic.max_depth,
                "ponder_cost": dynamic.ponder_cost,
                "lane_balance_loss": dynamic.lane_balance_loss,
                "serial_token_steps": dynamic.serial_token_steps,
                "parallel_token_lane_calls": dynamic.parallel_token_lane_calls,
                "selected_lane_counts": dynamic.selected_lane_counts,
                "executed_steps": dynamic.executed_steps,
                "dense_serial_token_calls": dynamic.dense_serial_token_calls,
            }
        else:
            for block in self.blocks:
                hidden = block(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        hidden = self.final_norm(hidden)
        typed_memory = (
            self.typed_memory(hidden, channel_masks)
            if self.typed_memory is not None
            else None
        )
        diagnostics = None
        if typed_memory is not None and self.typed_fusion is not None:
            fused = self.typed_fusion(hidden, typed_memory)
            hidden = fused.hidden_states
            diagnostics = fused.diagnostics
        growth_diagnostics: dict[str, torch.Tensor | str] | None = None
        unified = None
        if self.unified_competence is not None:
            unified = self.unified_competence(
                hidden,
                attention_mask=attention_mask,
            )
            hidden = unified.hidden_states
            growth_diagnostics = {
                f"unified_{name}": value
                for name, value in unified.diagnostics.items()
            }
        if self.neuron is not None:
            neuron = self.neuron(hidden)
            hidden = neuron.hidden_states
            growth_diagnostics = {
                f"neuron_{name}": value
                for name, value in neuron.diagnostics.items()
            }
        if self.expert_bank is not None:
            expert = self.expert_bank(hidden, attention_mask)
            hidden = hidden + expert.residual
            growth_diagnostics = {
                **(growth_diagnostics or {}),
                **{
                    f"expert_{name}": value
                    for name, value in expert.diagnostics.items()
                },
            }
        typed = {} if typed_memory is None else typed_memory.token_states
        gap_diagnostics = None
        gap_logits = None
        gap_states = None
        if self.gap_ensemble is not None:
            gap_output = self.gap_ensemble(
                hidden,
                router_enabled=self.config.gap_router_enabled,
            )
            hidden = gap_output.hidden_states
            gap_diagnostics = gap_output.diagnostics
            gap_logits = gap_output.gap_logits
            gap_states = gap_output.gap_states
        elif unified is not None:
            gap_diagnostics = unified.diagnostics
            gap_logits = {
                name: unified.planner.competence_logits[..., index]
                for index, name in enumerate(self.config.gap_names)
            }
            gap_states = unified.competence_states
        benchmark_proxy = (
            self.benchmark_proxy.predict(hidden, attention_mask)
            if self.benchmark_proxy is not None
            else None
        )
        conversation_logits = (
            self.conversation_head(hidden)
            if self.conversation_head is not None
            else None
        )
        tree_mode_logits = (
            unified.planner.mode_logits
            if unified is not None
            else None
        )
        runtime_synthesis_score = None
        if self.runtime_competence is not None:
            mask = (
                torch.ones_like(input_ids, dtype=torch.bool)
                if attention_mask is None
                else attention_mask.to(device=hidden.device, dtype=torch.bool)
            )
            weights = mask.to(hidden.dtype)[..., None]
            pooled = (hidden * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
            if gap_logits:
                gap_values = torch.stack(
                    list(gap_logits.values()),
                    dim=-1,
                ).mean(dim=1)
                probabilities = torch.softmax(gap_values, dim=-1)
                router_entropy = -(
                    probabilities.clamp_min(1.0e-8).log() * probabilities
                ).sum(dim=-1) / torch.log(
                    pooled.new_tensor(float(max(2, probabilities.shape[-1])))
                )
            else:
                router_entropy = pooled.new_zeros(pooled.shape[0])
            evidence = torch.stack(
                (
                    pooled.new_zeros(pooled.shape[0]),
                    router_entropy,
                    pooled.new_full((pooled.shape[0],), 0.5),
                    pooled.new_zeros(pooled.shape[0]),
                    pooled.new_zeros(pooled.shape[0]),
                ),
                dim=-1,
            )
            runtime_synthesis_score = self.runtime_competence.detect(
                pooled,
                evidence,
            )
        weight = self.embedding.weight if self.output is None else self.output.weight
        logits = hidden @ weight.transpose(0, 1)
        return NLPModelOutput(
            hidden_states=hidden,
            token_logits=logits,
            typed_channels=typed,
            typed_memory=typed_memory,
            fusion_diagnostics=diagnostics,
            dynamic_diagnostics=dynamic_diagnostics,
            gap_diagnostics=gap_diagnostics,
            gap_logits=gap_logits,
            gap_states=gap_states,
            benchmark_proxy=benchmark_proxy,
            conversation_logits=conversation_logits,
            growth_diagnostics=growth_diagnostics,
            tree_mode_logits=tree_mode_logits,
            runtime_synthesis_score=runtime_synthesis_score,
        )

    def json_action_logits(
        self,
        hidden: torch.Tensor,
        states: list[JSONGrammarState],
        step_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.json_head is None:
            raise RuntimeError("JSON action head is disabled")
        return self.json_head(hidden, states, step_ids)

    def json_typed_logits(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.json_typed_heads is None:
            raise RuntimeError("JSON typed heads are disabled")
        return self.json_typed_heads(hidden)

    def json_schema_key_logits(self, hidden: torch.Tensor, states):
        if self.json_schema_key_head is None:
            raise RuntimeError("JSON schema key head is disabled")
        return self.json_schema_key_head(hidden, states)

    def synthesize_expert(
        self,
        parent_hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> GeneratedExpertState:
        if self.expert_factory is None:
            if (
                self.unified_competence is not None
                and self.unified_competence.generated_expert is not None
            ):
                return self.unified_competence.generated_expert(
                    parent_hidden,
                    context,
                )
            if self.expert_bank is None or self.expert_bank.factory is None:
                raise RuntimeError("expert factory is disabled")
            return self.expert_bank.factory(parent_hidden, context)
        return self.expert_factory(parent_hidden, context)

    def materialize_expert(self, context: torch.Tensor) -> MaterializedExpert:
        if (
            self.unified_competence is not None
            and self.unified_competence.generated_expert is not None
        ):
            return self.unified_competence.generated_expert.materialize(context)
        if self.expert_factory is not None:
            return self.expert_factory.materialize(context)
        raise RuntimeError("generated expert factory is disabled")

    def apply_materialized_expert(
        self,
        hidden: torch.Tensor,
        expert: MaterializedExpert,
    ) -> torch.Tensor:
        if (
            self.unified_competence is not None
            and self.unified_competence.generated_expert is not None
        ):
            return self.unified_competence.apply_materialized_generated_expert(
                hidden,
                expert,
            )
        if self.expert_factory is not None:
            return self.expert_factory.apply_materialized(hidden, expert)
        raise RuntimeError("generated expert factory is disabled")

    def freeze_growth(self) -> None:
        if self.expert_bank is not None:
            self.expert_bank.freeze_growth()
        if self.neuron is not None:
            for parameter in self.neuron.parameters():
                parameter.requires_grad_(False)
        if self.unified_competence is not None:
            self.unified_competence.freeze_generated_expert()


__all__ = ["CompactNLPDecoder", "NLPModelOutput"]
