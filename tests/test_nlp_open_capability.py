from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.engineering import ExperimentSpec

ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict:
    return json.loads(
        (ROOT / "configs/nlp_open_capability_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_open_capability_spec_is_valid_but_explicitly_exploratory() -> None:
    spec = ExperimentSpec.from_mapping(_payload())

    assert spec.tier == "OPEN"
    assert spec.payload["track"] == "open_capability_exploratory"
    assert spec.payload["model"]["parameter_budget_mode"] == (
        "open_capability_exploratory"
    )
    assert spec.validate() == ()


def test_open_capability_dialogue_spec_matches_pretrain_architecture() -> None:
    pretrain = ExperimentSpec.from_json(
        ROOT / "configs/nlp_open_capability_v1.json"
    )
    dialogue = ExperimentSpec.from_json(
        ROOT / "configs/nlp_open_capability_dialogue_sft_v1.json"
    )

    assert dialogue.payload["data"]["text_mode"] == "chat"
    assert dialogue.payload["model"]["hidden_size"] == (
        pretrain.payload["model"]["hidden_size"]
    )
    assert dialogue.payload["model"]["num_layers"] == (
        pretrain.payload["model"]["num_layers"]
    )
    assert dialogue.payload["optimizer"]["learning_rate"] < (
        pretrain.payload["optimizer"]["learning_rate"]
    )


def test_conversation_v2_uses_row_aligned_answer_supervision() -> None:
    spec = ExperimentSpec.from_json(
        ROOT / "configs/nlp_open_capability_conversation_sft_v3.json"
    )

    assert spec.payload["data"]["dataset"] == "row_aligned_instruction"
    assert spec.payload["data"]["manifest"] == (
        ".data/nlp/chat_conversation_v3/manifest.json"
    )


def test_non_exploratory_spec_cannot_enable_open_growth() -> None:
    payload = copy.deepcopy(_payload())
    payload["track"] = "primary"
    payload["tier"] = "P12M"
    payload["model"]["parameter_budget_mode"] = "matched_20_to_1"

    with pytest.raises(
        ValueError,
        match="net_parameter_growth_allowed",
    ):
        ExperimentSpec.from_mapping(payload)
