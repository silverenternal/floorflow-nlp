#!/usr/bin/env python3
"""Freeze tokenizer hash and the P2M/P5M corpus-to-parameter ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / ".data/nlp/p5m_v2/manifest.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / ".data/nlp/tokenizer_v1/tokenizer.json",
    )
    parser.add_argument("--tier", choices=("P2M", "P5M", "P12M"), default="P5M")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_tokenizer_budget_v1.json",
    )
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.output = args.output.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    combined = manifest["combined"]
    total_tokens = int(manifest["combined_total_tokens"])
    split_tokens = {
        split: int(combined[split]["tokens"])
        for split in ("train", "validation", "test")
    }
    tokenizer_sha = _sha256(args.tokenizer)
    tiers = {}
    all_tiers = (
        ("P2M", 46_000_000, 2_300_000),
        ("P5M", 92_000_000, 4_600_000),
        ("P12M", 230_000_000, 11_500_000),
    )
    for tier, planned_tokens, target_parameters in all_tiers:
        tiers[tier] = {
            "planned_tokens": planned_tokens,
            "target_parameters": target_parameters,
            "available_tokens": total_tokens,
            "seat_available": total_tokens >= planned_tokens,
            "actual_tokens_per_planned_parameter": total_tokens / target_parameters,
            "parameter_target_from_actual_tokens_over_20": total_tokens // 20,
        }
    checks = {
        "manifest_pass": manifest["status"] == "pass",
        "tokenizer_hash_matches_manifest": (
            tokenizer_sha == manifest["base"]["tokenizer_sha256"]
        ),
        "split_tokens_sum_to_total": sum(split_tokens.values()) == total_tokens,
        "p2m_seat_available": tiers["P2M"]["seat_available"],
        f"{args.tier.lower()}_seat_available": tiers[args.tier]["seat_available"],
        "ratio_is_twenty_to_one": (
            tiers[args.tier]["parameter_target_from_actual_tokens_over_20"]
            * 20
            <= total_tokens
            < (
                tiers[args.tier]["parameter_target_from_actual_tokens_over_20"]
                + 1
            )
            * 20
        ),
    }
    report = {
        "version": "nlp-tokenizer-budget-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tokenizer": str(args.tokenizer.relative_to(ROOT)),
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": 8192,
        "manifest": str(args.manifest.relative_to(ROOT)),
        "split_tokens": split_tokens,
        "total_tokens": total_tokens,
        "tiers": tiers,
        "selected_tier": args.tier,
        "evidence_boundary": "Tokenizer/token ledger only; model quality is untested.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
