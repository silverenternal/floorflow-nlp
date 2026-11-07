"""Compact hierarchical decoder skeleton with tied output and typed heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from newera.neuron.graph import (
    DynamicNeuronGraphLayer,
    LinkMutation,
    LinkProposalPolicy,
    NeuronGraphTopology,
    TopologyController,
    gather_rows,
)
from newera.nlp_flow.model.components.architecture.foundation.blocks import CompactDecoderBlock, RMSNorm
from newera.nlp_flow.model.components.architecture.foundation.config import NLPModelConfig
from newera.nlp_flow.model.components.architecture.representations.channels import (
    TypedChannelMemory,
    TypedChannelMemoryEncoder,
)
from newera.nlp_flow.model.components.architecture.representations.fusion import TypedChannelFusion
from newera.nlp_flow.model.components.architecture.state.dynamic_stack import (
    DynamicStackConfig,
    DynamicStackDecoder,
)
from newera.nlp_flow.model.components.experts.banks.expert_bank import (
    ExpertBankConfig,
    ExpertResidualBank,
)
from newera.nlp_flow.model.components.experts.banks.expert_factory import (
    ContextConditionedExpertFactory,
    GeneratedExpertState,
    MaterializedExpert,
)
from newera.nlp_flow.model.components.experts.banks.neuron_module import NeuronModule
from newera.nlp_flow.model.components.experts.competence.competence import (
    UnifiedCompetenceBank,
    UnifiedCompetenceConfig,
)
from newera.nlp_flow.model.components.experts.competence.emergence import (
    EmergenceConfig,
    EmergentSemanticObjective,
)
from newera.nlp_flow.model.components.experts.competence.runtime_competence import (
    RuntimeCompetenceConfig,
    RuntimeCompetenceManager,
)
from newera.nlp_flow.model.components.experts.ensemble.gap_ensemble import GapEnsemble, GapEnsembleConfig
from newera.nlp_flow.model.components.tasks.conversation.conversation import (
    ConversationControlHead,
    TurnSegmentEmbedding,
)
from newera.nlp_flow.model.components.tasks.output.benchmark_proxy import (
    BenchmarkProxyConfig,
    BenchmarkProxyObjective,
    BenchmarkProxyPrediction,
)
from newera.nlp_flow.model.components.tasks.output.json_grammar import (
    JSONActionHead,
    JSONGrammarState,
    JSONSchemaKeyHead,
    JSONTypedHeads,
)
from newera.nlp_flow.model.components.tasks.runtime.runtime_model_api import RuntimeModelMixin
from newera.nlp_flow.model.components.training.objectives.cfm import TwoStageASTCFM


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
    neuron_topology: NeuronGraphTopology | None = None
    neuron_graph_diagnostics: dict[str, torch.Tensor | int] | None = None
    neuron_candidate_edges: torch.Tensor | None = None
    neuron_link_logits: torch.Tensor | None = None
    neuron_link_strengths: torch.Tensor | None = None
    neuron_link_loss: torch.Tensor | None = None


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
        self.neuron_graph = (
            DynamicNeuronGraphLayer(
                config.hidden_size,
                config.neuron_graph_link_types,
                causal=True,
            )
            if config.enable_neuron_graph
            else None
        )
        self.neuron_link_policy = (
            LinkProposalPolicy(
                config.hidden_size,
                config.neuron_graph_link_types,
                initial_keep_bias=config.neuron_graph_initial_keep_bias,
                predict_strengths=config.neuron_graph_predict_strengths,
            )
            if config.enable_neuron_graph
            else None
        )
        self.neuron_topology_controller = (
            TopologyController(max_edges=config.neuron_graph_max_edges, causal=True)
            if config.enable_neuron_graph
            else None
        )
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
        neuron_topology: NeuronGraphTopology | None = None,
        neuron_candidate_edges: torch.Tensor | None = None,
        neuron_mutations: tuple[LinkMutation, ...] = (),
        neuron_link_threshold: float = 0.8,
        segment_ids: torch.Tensor | None = None,
        neuron_link_targets: torch.Tensor | None = None,
        update_neuron_topology: bool = False,
        neuron_candidate_offset: int = 0,
        neuron_node_ids: torch.Tensor | None = None,
    ) -> NLPModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured context length")
        if self.neuron_graph is not None and self.training and (
            update_neuron_topology or neuron_mutations
        ):
            raise ValueError(
                "hard topology updates require eval mode; apply them between training steps"
            )
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain an out-of-vocabulary id")
        if torch.are_deterministic_algorithms_enabled():
            token_embeddings = gather_rows(
                self.embedding.weight,
                input_ids.reshape(-1),
            ).reshape(*input_ids.shape, self.config.hidden_size)
        else:
            token_embeddings = self.embedding(input_ids)
        hidden = self.embedding_dropout(token_embeddings)
        if turn_ids is not None:
            if turn_ids.shape != input_ids.shape:
                raise ValueError("turn_ids must match input_ids")
            if self.turn_embedding is None:
                raise RuntimeError("turn ids were provided but turn embeddings are disabled")
            if turn_ids.min() < 0 or turn_ids.max() >= self.config.turn_type_count:
                raise ValueError("turn_ids contain an out-of-range type")
            turn_values = (
                gather_rows(self.turn_embedding.weight, turn_ids.reshape(-1)).reshape(
                    *turn_ids.shape, self.config.hidden_size
                )
                if torch.are_deterministic_algorithms_enabled()
                else self.turn_embedding(turn_ids)
            )
            hidden = hidden + turn_values
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
                + (
                    gather_rows(self.task_family_embedding.weight, family_ids.to(input_ids.device).reshape(-1)).reshape(
                        *family_ids.shape, self.config.hidden_size
                    )
                    if torch.are_deterministic_algorithms_enabled()
                    else self.task_family_embedding(family_ids.to(input_ids.device))
                )
                + (
                    gather_rows(self.task_language_embedding.weight, language_ids.to(input_ids.device).reshape(-1)).reshape(
                        *language_ids.shape, self.config.hidden_size
                    )
                    if torch.are_deterministic_algorithms_enabled()
                    else self.task_language_embedding(language_ids.to(input_ids.device))
                )
            )
        neuron_graph_diagnostics = None
        neuron_link_logits = None
        neuron_link_strengths = None
        neuron_link_loss = None
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must match input_ids")
        if neuron_node_ids is not None and (
            neuron_node_ids.dtype != torch.long
            or neuron_node_ids.shape != input_ids.shape
            or neuron_node_ids.device != hidden.device
        ):
            raise ValueError("neuron_node_ids must be long [batch, sequence] on the model device")
        if neuron_node_ids is not None:
            valid_node_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is not None:
                valid_node_mask &= attention_mask.to(torch.bool)
            if segment_ids is not None:
                valid_node_mask &= segment_ids.ge(0)
            valid_node_ids = neuron_node_ids[valid_node_mask]
            invalid_node_ids = neuron_node_ids[~valid_node_mask]
            if torch.any(valid_node_ids < 0) or torch.any(invalid_node_ids != -1):
                raise ValueError("valid nodes need non-negative IDs; padding nodes must use -1")
            if torch.unique(valid_node_ids).numel() != valid_node_ids.numel():
                raise ValueError("neuron_node_ids must be unique across the graph batch")
        if self.neuron_graph is not None:
            if neuron_topology is not None:
                neuron_topology.validate(
                    batch_size=input_ids.shape[0],
                    nodes_per_graph=input_ids.shape[1],
                    num_edge_types=self.config.neuron_graph_link_types,
                    device=hidden.device,
                    causal=True,
                )
                if neuron_topology.node_ids is None or neuron_node_ids is None:
                    raise ValueError(
                        "reusing neuron_topology requires stored and supplied neuron_node_ids"
                    )
                if not torch.equal(neuron_topology.node_ids, neuron_node_ids):
                    raise ValueError("neuron_topology node identities do not match the current nodes")
            graph_topology = neuron_topology or NeuronGraphTopology.causal_chain(
                input_ids.shape[0],
                input_ids.shape[1],
                device=hidden.device,
                dtype=hidden.dtype,
                node_ids=neuron_node_ids,
            )
            graph_topology.validate(
                batch_size=input_ids.shape[0],
                nodes_per_graph=input_ids.shape[1],
                num_edge_types=self.config.neuron_graph_link_types,
                device=hidden.device,
                causal=True,
            )
            graph_topology = graph_topology.filter_context(
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                num_edge_types=self.config.neuron_graph_link_types,
            )
            if (
                self.config.neuron_graph_max_edges is not None
                and graph_topology.edge_count > self.config.neuron_graph_max_edges
            ):
                raise ValueError("topology exceeds the configured global edge budget")
            assert self.neuron_link_policy is not None
            assert self.neuron_topology_controller is not None
            if neuron_candidate_edges is None:
                neuron_candidate_edges = NeuronGraphTopology.causal_candidates(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    device=hidden.device,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    max_candidates_per_graph=self.config.neuron_graph_candidate_budget,
                    randomize=self.training,
                    candidate_offset=neuron_candidate_offset,
                )
                check_candidates = False
            else:
                check_candidates = True
            if neuron_link_targets is not None:
                if (
                    neuron_link_targets.shape != (neuron_candidate_edges.shape[1],)
                    or neuron_link_targets.dtype != torch.long
                    or neuron_link_targets.device != hidden.device
                ):
                    raise ValueError("neuron_link_targets must align with candidate edges on the model device")
                target_mask = NeuronGraphTopology.candidate_context_mask(
                    neuron_candidate_edges,
                    batch_size=input_ids.shape[0],
                    nodes_per_graph=input_ids.shape[1],
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
                neuron_link_targets = neuron_link_targets[target_mask]
            neuron_candidate_edges = NeuronGraphTopology.filter_candidates(
                neuron_candidate_edges,
                batch_size=input_ids.shape[0],
                nodes_per_graph=input_ids.shape[1],
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )
            if check_candidates and neuron_candidate_edges.shape[1]:
                candidate_count = neuron_candidate_edges.shape[1]
                node_count = input_ids.shape[0] * input_ids.shape[1]
                candidate_keys = neuron_candidate_edges[0] * node_count + neuron_candidate_edges[1]
                if torch.unique(candidate_keys).numel() != candidate_count:
                    raise ValueError("neuron_candidate_edges cannot contain duplicate directed edges")
            if neuron_link_targets is not None:
                node_count = input_ids.shape[0] * input_ids.shape[1]
                existing_count = graph_topology.edge_count
                combined = torch.cat((graph_topology.edge_index, neuron_candidate_edges), dim=1)
                combined_keys = combined[0] * node_count + combined[1]
                unique_keys, inverse = torch.unique(
                    combined_keys, sorted=True, return_inverse=True
                )
                neuron_candidate_edges = torch.stack(
                    (unique_keys // node_count, unique_keys % node_count)
                )
                aligned_targets = torch.full(
                    (unique_keys.numel(),), -100, dtype=torch.long, device=hidden.device
                )
                aligned_targets[inverse[existing_count:]] = neuron_link_targets
                neuron_link_targets = aligned_targets
            else:
                node_count = input_ids.shape[0] * input_ids.shape[1]
                combined = torch.cat((graph_topology.edge_index, neuron_candidate_edges), dim=1)
                unique_keys = torch.unique(
                    combined[0] * node_count + combined[1], sorted=True
                )
                neuron_candidate_edges = torch.stack(
                    (unique_keys // node_count, unique_keys % node_count)
                )
            neuron_link_logits = self.neuron_link_policy(
                hidden,
                neuron_candidate_edges,
                graph_topology,
                check_values=check_candidates,
            )
            if self.config.neuron_graph_predict_strengths:
                neuron_link_strengths = self.neuron_link_policy.predict_strengths(
                    hidden,
                    neuron_candidate_edges,
                    graph_topology,
                    check_values=check_candidates,
                )
            if neuron_link_targets is not None:
                neuron_link_loss = self.neuron_link_policy.supervised_loss(
                    neuron_link_logits,
                    neuron_link_targets,
                )
            if update_neuron_topology or neuron_mutations:
                if self.config.neuron_graph_max_edges is None:
                    raise ValueError(
                        "hard topology updates require a finite neuron_graph_max_edges budget"
                    )
                policy_mutations = (
                    self.neuron_link_policy.propose(
                        neuron_link_logits,
                        neuron_candidate_edges,
                        graph_topology,
                        threshold=neuron_link_threshold,
                        candidate_strengths=neuron_link_strengths,
                    )
                    if update_neuron_topology
                    else ()
                )
                graph_topology = self.neuron_topology_controller.apply(
                    graph_topology,
                    (*neuron_mutations, *policy_mutations),
                    num_edge_types=self.config.neuron_graph_link_types,
                )
                context_topology = graph_topology.filter_context(
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    num_edge_types=self.config.neuron_graph_link_types,
                )
                if context_topology.edge_count != graph_topology.edge_count:
                    raise ValueError(
                        "hard topology updates cannot connect padding or different segments"
                    )
                graph_topology = context_topology
            if self.config.neuron_graph_max_edges is not None and graph_topology.edge_count > self.config.neuron_graph_max_edges:
                raise ValueError("topology exceeds the configured global edge budget")
            candidate_type_weights = (
                self.neuron_link_policy.expected_type_weights(
                    neuron_link_logits,
                    neuron_candidate_edges,
                    graph_topology,
                    candidate_strengths=neuron_link_strengths,
                )
                if self.training
                else None
            )
            if candidate_type_weights is not None:
                graph_result = self.neuron_graph(
                    hidden,
                    graph_topology,
                    candidate_edges=neuron_candidate_edges,
                    candidate_type_weights=candidate_type_weights,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
            else:
                graph_result = self.neuron_graph(
                    hidden,
                    graph_topology,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
            hidden = graph_result.hidden_states
            neuron_graph_diagnostics = graph_result.diagnostics
            neuron_topology = graph_topology
        elif (
            neuron_topology is not None
            or neuron_candidate_edges is not None
            or neuron_mutations
            or neuron_link_targets is not None
            or neuron_node_ids is not None
        ):
            raise RuntimeError("neuron topology inputs were provided but neuron graph is disabled")
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
                segment_ids=segment_ids,
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
                    segment_ids=segment_ids,
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
            neuron_topology=neuron_topology,
            neuron_graph_diagnostics=neuron_graph_diagnostics,
            neuron_candidate_edges=neuron_candidate_edges,
            neuron_link_logits=neuron_link_logits,
            neuron_link_strengths=neuron_link_strengths,
            neuron_link_loss=neuron_link_loss,
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
