"""Corpus mixture and token-to-parameter budget contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import (
    CODE_LANGUAGE_NAMES,
    CORPUS_FAMILY_NAMES,
)


@dataclass(frozen=True)
class CorpusTier:
    tier_id: str
    total_tokens: int
    target_parameters: int
    max_context_tokens: int
    vocab_size: int
    train_tokens: int
    validation_tokens: int
    test_tokens: int

    def parameter_error(self, tokens_per_parameter: int) -> int:
        return self.target_parameters - self.total_tokens // tokens_per_parameter


def parameter_target(total_tokens: int, tokens_per_parameter: int = 20) -> int:
    if total_tokens <= 0 or tokens_per_parameter <= 0:
        raise ValueError("tokens and ratio denominator must be positive")
    return total_tokens // tokens_per_parameter


def _integer(row: Mapping[str, Any], field: str, errors: list[str]) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return 0
    return value


def _tiers(payload: Any, ratio: int, tolerance: float, errors: list[str]) -> tuple[CorpusTier, ...]:
    if not isinstance(payload, list) or not payload:
        errors.append("tiers must be a non-empty list")
        return ()
    tiers: list[CorpusTier] = []
    ids: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            errors.append(f"tiers[{index}] must be an object")
            continue
        tier_id = str(row.get("id", ""))
        if not tier_id or tier_id in ids:
            errors.append(f"tiers[{index}] has a missing or duplicate id")
        ids.add(tier_id)
        total = _integer(row, "total_tokens", errors)
        target = _integer(row, "target_parameters", errors)
        context = _integer(row, "max_context_tokens", errors)
        vocab = _integer(row, "vocab_size", errors)
        splits = row.get("splits")
        if not isinstance(splits, Mapping):
            errors.append(f"tiers[{index}].splits must be an object")
            splits = {}
        train = _integer(splits, "train", errors)
        validation = _integer(splits, "validation", errors)
        test = _integer(splits, "test", errors)
        if total and sum((train, validation, test)) != total:
            errors.append(f"{tier_id} split tokens do not sum to total_tokens")
        expected = parameter_target(total, ratio) if total else 0
        if expected and abs(target - expected) / expected > tolerance:
            errors.append(f"{tier_id} target_parameters violates the 1/{ratio} ratio")
        tiers.append(
            CorpusTier(
                tier_id,
                total,
                target,
                context,
                vocab,
                train,
                validation,
                test,
            )
        )
    return tuple(tiers)


def _mix(payload: Any, errors: list[str]) -> frozenset[str]:
    if not isinstance(payload, list) or not payload:
        errors.append("mix must be a non-empty list")
        return frozenset()
    families: list[str] = []
    shares: list[int] = []
    code_languages: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            errors.append(f"mix[{index}] must be an object")
            continue
        family = str(row.get("family", ""))
        share = row.get("share_percent")
        if not isinstance(share, int) or isinstance(share, bool) or not 0 < share <= 100:
            errors.append(f"mix[{index}].share_percent must be in (0, 100]")
            continue
        families.append(family)
        shares.append(share)
        if family == "code":
            languages = row.get("languages")
            if not isinstance(languages, list):
                errors.append("code mix must declare languages")
                continue
            for language in languages:
                if isinstance(language, Mapping):
                    code_languages.add(str(language.get("name", "")))
            language_total = sum(
                int(language.get("share_percent", 0))
                for language in languages
                if isinstance(language, Mapping)
            )
            if language_total != share:
                errors.append("code language shares must sum to the code family share")
    if sum(shares) != 100:
        errors.append("family share_percent values must sum to 100")
    expected_families = set(CORPUS_FAMILY_NAMES)
    if set(families) != expected_families or len(families) != len(set(families)):
        errors.append(f"mix families must be exactly {sorted(expected_families)}")
    if code_languages != set(CODE_LANGUAGE_NAMES):
        errors.append(f"code languages must be exactly {list(CODE_LANGUAGE_NAMES)}")
    return frozenset(families)


def validate_corpus_plan(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every corpus/ratio contract violation."""

    errors: list[str] = []
    ratio = payload.get("ratio_contract")
    if not isinstance(ratio, Mapping):
        errors.append("ratio_contract must be an object")
        ratio = {}
    tokens_per_parameter = ratio.get("tokens_per_parameter")
    if tokens_per_parameter != 20:
        errors.append("ratio_contract.tokens_per_parameter must equal 20")
        tokens_per_parameter = 20
    tolerance = ratio.get("tolerance_relative")
    if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 0.05:
        errors.append("ratio_contract.tolerance_relative must be in [0, 0.05]")
        tolerance = 0.02
    _tiers(payload.get("tiers"), int(tokens_per_parameter), float(tolerance), errors)
    _mix(payload.get("mix"), errors)

    split = payload.get("split_policy")
    if not isinstance(split, Mapping):
        errors.append("split_policy must be an object")
    else:
        if split.get("random_line_split_allowed") is not False:
            errors.append("random line splitting must be disabled")
        if split.get("tokenizer_fit_split") != "train":
            errors.append("tokenizer must be fitted on train only")
    gates = payload.get("promotion_gates")
    if not isinstance(gates, Mapping) or len(gates) < 6:
        errors.append("promotion_gates is incomplete")
    return tuple(errors)


def corpus_tiers(payload: Mapping[str, Any]) -> tuple[CorpusTier, ...]:
    errors = validate_corpus_plan(payload)
    if errors:
        raise ValueError("; ".join(errors))
    ratio = payload["ratio_contract"]
    return _tiers(
        payload["tiers"],
        int(ratio["tokens_per_parameter"]),
        float(ratio["tolerance_relative"]),
        [],
    )


__all__ = [
    "CorpusTier",
    "corpus_tiers",
    "parameter_target",
    "validate_corpus_plan",
]
