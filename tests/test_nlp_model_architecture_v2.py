from __future__ import annotations

import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_quality_gate import ASTHandoffEvidence
from floorflow.nlp_flow.model import (
    CompactNLPDecoder,
    NLPModelConfig,
    ScopeController,
)

ROOT = Path(__file__).resolve().parents[1]


def _base_config(**overrides) -> NLPModelConfig:
    values = {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 64,
        "max_seq_len": 32,
        "parameter_cap": 2_000_000,
    }
    values.update(overrides)
    return NLPModelConfig(**values)


def test_optional_growth_proxy_conversation_modules_are_zero_residual_identity() -> None:
    torch.manual_seed(211)
    control = CompactNLPDecoder(_base_config()).eval()
    torch.manual_seed(211)
    enhanced = CompactNLPDecoder(
        _base_config(
            enable_turn_embeddings=True,
            enable_conversation_control=True,
            enable_gap_ensemble=True,
            enable_benchmark_proxy=True,
            enable_expert_bank=True,
            expert_bank_mode="fixed_pool",
            expert_count=4,
            expert_top_k=2,
            enable_neuron_module=True,
            neuron_local_size=8,
        )
    ).eval()
    ids = torch.randint(0, 128, (2, 8))
    turns = torch.tensor([[0] * 5 + [1] * 3, [0] * 4 + [1] * 4])

    with torch.no_grad():
        left = control(ids)
        right = enhanced(ids, turn_ids=turns)

    assert torch.allclose(left.token_logits, right.token_logits)
    assert right.benchmark_proxy is not None
    assert right.conversation_logits is not None
    assert right.growth_diagnostics is not None
    assert set(right.conversation_logits) == {
        "continue_logits",
        "turn_boundary_logits",
        "reply_mode_logits",
        "language_logits",
        "validity_logits",
    }
    assert "expert_expert_gate" in right.growth_diagnostics
    assert "neuron_fast_gate_mean" in right.growth_diagnostics


def test_expanded_training_scopes_select_independent_modules() -> None:
    model = CompactNLPDecoder(
        _base_config(
            enable_conversation_control=True,
            enable_turn_embeddings=True,
            enable_gap_ensemble=True,
            enable_benchmark_proxy=True,
            enable_expert_bank=True,
        )
    )
    expectations = {
        "S1-gaps": "gap_ensemble",
        "S2-gap-router": "gap_ensemble",
        "S4-proxy": "benchmark_proxy",
        "S5-conversation": "conversation_head",
        "S6-growth": "expert_bank",
    }
    for scope_name, prefix in expectations.items():
        controller = ScopeController.from_name(scope_name)
        report = controller.apply(model)
        assert prefix in report["prefixes"]


def test_growth_freeze_marks_dynamic_children_static() -> None:
    model = CompactNLPDecoder(
        _base_config(
            enable_expert_bank=True,
            enable_neuron_module=True,
            neuron_local_size=8,
        )
    )
    model.freeze_growth()

    assert all(
        not parameter.requires_grad
        for parameter in model.expert_bank.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.neuron.parameters()
    )


def test_v2_architecture_config_builds_the_full_network() -> None:
    payload = json.loads(
        (ROOT / "configs/nlp_model_architecture_v2.json").read_text(
            encoding="utf-8"
        )
    )
    model = CompactNLPDecoder(NLPModelConfig(**payload["model"]))

    assert model.dynamic_stack is not None
    assert model.gap_ensemble is not None
    assert model.benchmark_proxy is not None
    assert model.turn_embedding is not None
    assert model.conversation_head is not None
    assert model.expert_bank is not None
    assert model.neuron is not None


def test_cfm_context_and_generated_ast_result_map_to_token_logits() -> None:
    model = CompactNLPDecoder(
        _base_config(
            enable_typed_channels=True,
            enable_two_stage_cfm=True,
        )
    ).eval()
    ids = torch.randint(0, 128, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, -2:] = False
    context = model.encode_cfm_context(ids, mask)
    cfm = model.sample_cfm(
        context,
        steps_a=2,
        steps_b=2,
        ast_noise=torch.randn(2, 32),
        result_noise=torch.randn(2, 8, 32),
    )
    logits = model.ast_result_token_logits(cfm.result_state)

    assert context.shape == (2, 32)
    assert logits.shape == (2, 8, 128)


def test_verified_cfm_requires_complete_ast_gate_evidence() -> None:
    model = CompactNLPDecoder(
        _base_config(
            enable_typed_channels=True,
            enable_two_stage_cfm=True,
        )
    ).eval()
    failed = ASTHandoffEvidence(
        source_kind="token_endpoint",
        status="fail",
        parse_validity=True,
        checks={"node_type_f1": False},
    )

    try:
        model.sample_verified_cfm(
            torch.randn(2, 32),
            failed,
            steps_a=1,
            steps_b=1,
        )
    except ValueError as error:
        assert "verified token-endpoint AST" in str(error)
    else:
        raise AssertionError("unverified AST reached the final CFM path")


def test_v3_unified_competence_replaces_separate_routers() -> None:
    payload = json.loads(
        (ROOT / "configs/nlp_model_architecture_v3.json").read_text(
            encoding="utf-8"
        )
    )
    model = CompactNLPDecoder(NLPModelConfig(**payload["model"])).eval()
    ids = torch.randint(0, 128, (2, 8))

    with torch.no_grad():
        output = model(ids)

    assert model.unified_competence is not None
    assert model.gap_ensemble is None
    assert model.expert_bank is None
    assert model.neuron is None
    assert output.gap_logits is not None
    assert output.gap_diagnostics is not None
    assert "unified_mean_birth_probability" in output.growth_diagnostics
    assert output.tree_mode_logits is not None
    assert output.tree_mode_logits.shape == (2, 8, 4)
    assert output.runtime_synthesis_score is not None
    assert output.runtime_synthesis_score.shape == (2,)
    materialized = model.materialize_expert(torch.randn(2, 320))
    applied = model.apply_materialized_expert(
        torch.randn(2, 8, 320),
        materialized,
    )
    assert applied.shape == (2, 8, 320)
    scope = ScopeController.from_name("S6-unified-competence").apply(model)
    assert scope["prefixes"] == ("unified_competence",)
    runtime_scope = ScopeController.from_name(
        "S9-runtime-competence"
    ).apply(model)
    assert runtime_scope["prefixes"] == ("runtime_competence",)


def test_v4_adds_native_schema_key_output_to_the_final_architecture() -> None:
    payload = json.loads(
        (ROOT / "configs/nlp_model_architecture_v4.json").read_text(
            encoding="utf-8"
        )
    )
    model = CompactNLPDecoder(NLPModelConfig(**payload["model"])).eval()
    ids = torch.randint(0, 128, (1, 8))

    with torch.no_grad():
        output = model(ids)

    assert model.json_schema_key_head is not None
    assert payload["inference_contract"]["single_deterministic_sample"] is True
    assert payload["inference_contract"]["parser_repair_allowed"] is False
    assert output.tree_mode_logits is not None


def test_task_metadata_embeddings_change_model_state() -> None:
    model = CompactNLPDecoder(
        _base_config(
            enable_task_metadata=True,
            task_family_count=4,
            task_language_count=4,
        )
    ).eval()
    ids = torch.randint(0, 128, (1, 8))
    family = torch.zeros(1, 8, dtype=torch.long)
    language = torch.zeros(1, 8, dtype=torch.long)

    with torch.no_grad():
        code = model(
            ids,
            family_ids=family,
            language_ids=language,
        ).hidden_states
        engineering = model(
            ids,
            family_ids=family + 3,
            language_ids=language + 1,
        ).hidden_states

    assert not torch.equal(code, engineering)


def test_planner_heads_can_be_isolated_for_causal_ablation() -> None:
    payload = json.loads(
        (ROOT / "configs/nlp_model_architecture_v3.json").read_text(
            encoding="utf-8"
        )
    )
    model = CompactNLPDecoder(NLPModelConfig(**payload["model"]))
    ScopeController.from_name("S8-planner-depth").apply(model)
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable
    assert all(
        ".head_adapters.depth." in name or ".planner.depth." in name
        for name in trainable
    )
