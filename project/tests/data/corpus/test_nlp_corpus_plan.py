from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import CODE_LANGUAGE_NAMES, CORPUS_FAMILY_NAMES
from newera.nlp_flow.components.platform.data.corpus.corpus import corpus_tiers, parameter_target, validate_corpus_plan

ROOT = Path(__file__).resolve().parents[4]


def load_plan() -> dict:
    return json.loads((ROOT / "project/configs/data/corpus/nlp_corpus_plan.json").read_text())


def test_corpus_plan_matches_rust_families_and_languages() -> None:
    payload = load_plan()
    families = {row["family"] for row in payload["mix"]}
    code_languages = {
        language["name"]
        for row in payload["mix"]
        if row["family"] == "code"
        for language in row["languages"]
    }

    assert validate_corpus_plan(payload) == ()
    assert families == set(CORPUS_FAMILY_NAMES)
    assert code_languages == set(CODE_LANGUAGE_NAMES)


def test_every_tier_obeys_one_to_twenty_parameter_budget() -> None:
    for tier in corpus_tiers(load_plan()):
        assert tier.total_tokens == tier.train_tokens + tier.validation_tokens + tier.test_tokens
        expected = parameter_target(tier.total_tokens, 20)
        assert expected == tier.target_parameters


def test_corpus_plan_rejects_wrong_parameter_ratio() -> None:
    payload = load_plan()
    payload["tiers"][0]["target_parameters"] += 100000

    errors = validate_corpus_plan(payload)

    assert any("1/20 ratio" in error for error in errors)


def test_corpus_plan_requires_python_rust_and_cpp() -> None:
    payload = load_plan()
    code = next(row for row in payload["mix"] if row["family"] == "code")
    code["languages"] = code["languages"][:-1]

    errors = validate_corpus_plan(payload)

    assert any("code languages" in error for error in errors)


def test_corpus_plan_requires_full_family_share() -> None:
    payload = load_plan()
    payload["mix"][0]["share_percent"] += 1

    errors = validate_corpus_plan(payload)

    assert "family share_percent values must sum to 100" in errors
