from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.research.experiments.studies.target import target_tiers, validate_target_plan

ROOT = Path(__file__).resolve().parents[5]


def load_target() -> dict:
    return json.loads((ROOT / "project/configs/other/general/chat_sft_canonical/nlp_coding_target.json").read_text())


def test_coding_target_obeys_one_to_twenty_budget() -> None:
    payload = load_target()

    assert validate_target_plan(payload) == ()
    assert [tier.parameters for tier in target_tiers(payload)] == [
        2_300_000,
        4_600_000,
        11_500_000,
        23_000_000,
        46_000_000,
    ]
    assert all(tier.corpus_tokens == tier.parameters * 20 for tier in target_tiers(payload))


def test_coding_target_requires_trition_and_linux_engineering_tracks() -> None:
    payload = load_target()
    engineering = next(
        track for track in payload["evaluation_tracks"] if track["id"] == "engineering_acceleration"
    )
    engineering["benchmarks"] = ["NLP-Triton-Kernel-Pilot"]

    errors = validate_target_plan(payload)

    assert "engineering_acceleration is missing Linux" in errors


def test_coding_target_rejects_best_of_n_primary_selection() -> None:
    payload = load_target()
    payload["benchmark_policy"]["generation_budget"]["best_of_n_allowed_in_primary"] = True

    errors = validate_target_plan(payload)

    assert "best-of-N cannot enter the primary endpoint" in errors


def test_coding_target_keeps_sota_threshold_unopened() -> None:
    payload = load_target()
    payload["sota_band"]["status"] = "frozen"

    errors = validate_target_plan(payload)

    assert any("SOTA-band thresholds" in error for error in errors)


def test_coding_target_rejects_large_model_stacking_claim() -> None:
    payload = load_target()
    payload["anti_scaling_contract"]["primary_claim_parameter_cap"] = 100_000_000

    errors = validate_target_plan(payload)

    assert "primary claim cap must not exceed 23M parameters" in errors


def test_coding_target_requires_efficiency_metrics() -> None:
    payload = load_target()
    payload["anti_scaling_contract"]["primary_efficiency_metrics"] = []

    errors = validate_target_plan(payload)

    assert any("efficiency are required" in error for error in errors)


def test_coding_target_requires_ast_first_structure_track() -> None:
    payload = load_target()
    payload["evaluation_tracks"] = [
        track for track in payload["evaluation_tracks"] if track["id"] != "ast_structure"
    ]

    errors = validate_target_plan(payload)

    assert any("evaluation tracks must be exactly" in error for error in errors)


def test_coding_target_treats_json_as_hard_guard() -> None:
    payload = load_target()
    payload["hard_guards"]["strict_json_parse_validity"] = 0.999
    payload["hard_guards"]["retry_count"] = 1

    errors = validate_target_plan(payload)

    assert "strict JSON parse validity must equal 1.0" in errors
    assert "JSON repair and retry counts must be zero" in errors
