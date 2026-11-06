"""Validation-only alignment checks for joined internal proxies and external metrics."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True))
    left_norm = sum((value - left_mean) ** 2 for value in left_ranks) ** 0.5
    right_norm = sum((value - right_mean) ** 2 for value in right_ranks) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _sign_agreement(left: list[float], right: list[float]) -> float:
    agrees = 0
    total = 0
    for first, second in itertools.combinations(range(len(left)), 2):
        proxy_delta = left[second] - left[first]
        external_delta = right[second] - right[first]
        if proxy_delta == 0 or external_delta == 0:
            continue
        total += 1
        agrees += int((proxy_delta > 0) == (external_delta > 0))
    return agrees / total if total else 0.0


def _interval(values: list[float], confidence: float) -> list[float]:
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    low = min(len(ordered) - 1, max(0, int(tail * len(ordered))))
    high = min(len(ordered) - 1, max(0, int((1.0 - tail) * len(ordered)) - 1))
    return [ordered[low], ordered[high]]


def align_proxy_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    minimum_checkpoints: int = 5,
    minimum_cases: int = 30,
    minimum_rho: float = 0.7,
    minimum_sign_agreement: float = 0.75,
    bootstrap_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Report alignment per family; insufficient support stays unqualified."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("proxy alignment rejects final-test, train and fixture rows")
        if row.get("oracle_only") or row.get("fixture_execution"):
            raise ValueError("proxy alignment rejects oracle and fixture labels")
        required = ("family", "checkpoint_id", "case_id", "proxy_value", "external_value", "evidence_sha256")
        if any(key not in row for key in required):
            raise ValueError("proxy alignment row is missing joined identity or evidence")
        key = (str(row["family"]), str(row["checkpoint_id"]), str(row["case_id"]))
        if key in seen:
            raise ValueError("proxy alignment contains duplicate checkpoint/case joins")
        seen.add(key)
        grouped[str(row["family"])].append(row)

    results = {}
    rng = random.Random(seed)
    for family, family_rows in sorted(grouped.items()):
        checkpoints = {row["checkpoint_id"] for row in family_rows}
        cases = {row["case_id"] for row in family_rows}
        if len(checkpoints) < minimum_checkpoints or len(cases) < minimum_cases:
            results[family] = {
                "qualified": False,
                "status": "unqualified_insufficient_support",
                "checkpoint_count": len(checkpoints),
                "case_count": len(cases),
                "rho": None,
                "sign_agreement": None,
            }
            continue
        proxy = [float(row["proxy_value"]) for row in family_rows]
        external = [float(row["external_value"]) for row in family_rows]
        rho = _correlation(proxy, external)
        sign = _sign_agreement(proxy, external)
        boot_rho = []
        boot_sign = []
        for _ in range(bootstrap_resamples):
            indices = [rng.randrange(len(family_rows)) for _ in family_rows]
            sample_proxy = [proxy[index] for index in indices]
            sample_external = [external[index] for index in indices]
            boot_rho.append(_correlation(sample_proxy, sample_external))
            boot_sign.append(_sign_agreement(sample_proxy, sample_external))
        rho_interval = _interval(boot_rho, confidence) if boot_rho else [rho, rho]
        sign_interval = _interval(boot_sign, confidence) if boot_sign else [sign, sign]
        qualified = rho >= minimum_rho and sign >= minimum_sign_agreement
        results[family] = {
            "qualified": qualified,
            "status": "qualified" if qualified else "unqualified_alignment_below_threshold",
            "checkpoint_count": len(checkpoints),
            "case_count": len(cases),
            "support_count": len(family_rows),
            "rho": rho,
            "rho_interval": rho_interval,
            "sign_agreement": sign,
            "sign_agreement_interval": sign_interval,
            "label_provenance": "joined_validation_verifier_evidence",
        }
    return {
        "status": "unqualified_no_families" if not results else "alignment_reported",
        "split": "validation",
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "confidence": confidence,
        "families": results,
    }


__all__ = ["align_proxy_metrics"]
