#!/usr/bin/env python3
"""CPU corpus and sampling probe for gap-aware data-efficient NLP learning."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from floorflow.nlp_flow.data_efficiency import (
    GAP_NAMES,
    CorpusExample,
    EfficientCorpusSampler,
    ExposureDeduplicator,
    HardExampleMemory,
    gap_training_examples,
    load_balanced_jsonl_examples,
    selection_metrics,
    token_efficiency_gain,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_data_efficient_learning_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/nlp_data_efficient_learning_probe_v1.json"
        ),
    )
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_balanced(
    paths: list[str],
    *,
    rows_per_family: int,
    max_scan_rows: int,
    code_language_quotas: dict[str, int] | None = None,
) -> tuple[CorpusExample, ...]:
    return load_balanced_jsonl_examples(
        tuple(_resolve(path) for path in paths),
        max_per_family=rows_per_family,
        max_scan_rows=max_scan_rows,
        max_per_code_language=code_language_quotas,
    )


def _byte_bigram_nll(
    training: list[CorpusExample],
    validation: list[CorpusExample],
) -> tuple[float, dict[str, float]]:
    transition = Counter()
    previous = Counter()
    for example in training:
        payload = example.text.encode("utf-8")[:4096]
        sequence = [256, *payload, 257]
        transition.update(itertools.pairwise(sequence))
        previous.update(sequence[:-1])
    vocabulary = 258
    alpha = 0.1
    family_nll: list[float] = []
    by_family: dict[str, list[float]] = {}
    for example in validation:
        payload = example.text.encode("utf-8")[:4096]
        sequence = [256, *payload, 257]
        losses = [
            -math.log(
                (transition[(left, right)] + alpha)
                / (previous[left] + alpha * vocabulary)
            )
            for left, right in itertools.pairwise(sequence)
        ]
        value = sum(losses) / max(1, len(losses))
        family_nll.append(value)
        by_family.setdefault(example.family, []).append(value)
    return (
        sum(family_nll) / max(1, len(family_nll)),
        {
            family: sum(values) / len(values)
            for family, values in by_family.items()
        },
    )


def _source_quality(
    validation: list[CorpusExample],
) -> dict[str, object]:
    python_total = 0
    python_valid = 0
    json_total = 0
    json_valid = 0
    for example in validation:
        if example.language == "python":
            python_total += 1
            try:
                ast.parse(example.text)
            except (SyntaxError, ValueError, RecursionError):
                pass
            else:
                python_valid += 1
        stripped = example.text.strip()
        if stripped.startswith(("{", "[")):
            json_total += 1
            try:
                json.loads(stripped)
            except (json.JSONDecodeError, RecursionError):
                pass
            else:
                json_valid += 1
    return {
        "python_source_ast_parse_rate": (
            python_valid / python_total if python_total else None
        ),
        "python_source_rows": python_total,
        "strict_json_source_rate": (
            json_valid / json_total if json_total else None
        ),
        "json_candidate_rows": json_total,
        "execution_validity": "not_measured_without_generated_task_outputs",
    }


def _run_arm(
    arm: dict[str, object],
    *,
    corpus: tuple[CorpusExample, ...],
    validation: list[CorpusExample],
    probe: dict[str, object],
) -> dict[str, object]:
    memory = HardExampleMemory(
        capacity=int(probe["hard_memory_capacity"])
    )
    sampler = EfficientCorpusSampler(
        corpus,
        arm=str(arm["sampler"]),
        seed=int(probe["seed"]),
        hard_memory=memory,
    )
    warmup = sampler.sample(
        count=int(probe["warmup_examples"]),
        step=0,
        progress=float(probe["curriculum_progress"][0]),
    )
    if arm["sampler"] == "curriculum_efficient":
        calibration_nll, _ = _byte_bigram_nll(list(warmup), list(warmup[:64]))
        for example in warmup[:256]:
            payload = example.text.encode("utf-8")[:1024]
            memory.observe(
                example,
                calibration_nll + len(payload) / max(example.token_count, 1),
            )
    late = sampler.sample(
        count=int(probe["late_examples"]),
        step=1,
        progress=float(probe["curriculum_progress"][1]),
    )
    selected = warmup + late
    heldout_nll, family_nll = _byte_bigram_nll(
        list(selected),
        validation,
    )
    metrics = selection_metrics(selected, reference=corpus)
    return {
        "arm": arm["name"],
        "sampler": arm["sampler"],
        "examples": len(selected),
        "seen_approximate_tokens": metrics.tokens,
        "heldout_byte_bigram_nll": heldout_nll,
        "heldout_nll_by_family": family_nll,
        "family_distribution": metrics.family_distribution,
        "gap_coverage": metrics.gap_coverage,
        "unique_examples": metrics.unique_examples,
        "duplicate_exposures": metrics.duplicate_exposures,
        "mean_difficulty": metrics.mean_difficulty,
        "hard_memory_examples": len(memory),
        "forgetting": "not_measured_in_cpu_probe",
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus_name = str(config["probe"]["corpus"])
    corpus_config = config["corpora"][corpus_name]
    for split in ("train", "validation", "test"):
        missing = [
            path
            for path in corpus_config[split]
            if not _resolve(path).exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing {split} shards: {missing}")

    raw_train = _load_balanced(
        corpus_config["train"],
        rows_per_family=int(config["probe"]["max_train_rows_per_family"]),
        max_scan_rows=int(config["probe"]["max_scan_rows"]),
        code_language_quotas={
            str(language): int(quota)
            for language, quota in config["probe"][
                "max_train_code_rows_per_language"
            ].items()
        },
    )
    deduplicator = ExposureDeduplicator(
        near_threshold=int(
            config["probe"]["near_duplicate_hamming_threshold"]
        ),
        max_near_copies=int(config["probe"]["max_near_copies"]),
    )
    corpus = deduplicator.filter(raw_train)
    validation = list(
        _load_balanced(
            corpus_config["validation"],
            rows_per_family=int(
                config["probe"]["max_validation_rows_per_family"]
            ),
            max_scan_rows=int(config["probe"]["max_scan_rows"]),
            code_language_quotas=None,
        )
    )
    if not corpus or not validation:
        raise RuntimeError("probe needs non-empty train and validation samples")

    arm_reports = [
        _run_arm(
            arm,
            corpus=corpus,
            validation=validation,
            probe=config["probe"],
        )
        for arm in config["arms"]
    ]
    uniform = arm_reports[0]
    for report in arm_reports:
        report["gain_per_seen_token_vs_uniform"] = token_efficiency_gain(
            uniform["heldout_byte_bigram_nll"],
            report["heldout_byte_bigram_nll"],
            report["seen_approximate_tokens"],
        )

    gap_rows = {
        gap: len(gap_training_examples(corpus, gap))
        for gap in GAP_NAMES
    }
    quality = _source_quality(validation)
    generated_only_gaps = set(config["generated_only_gaps"])
    checks = {
        "test_split_not_read": True,
        "all_arms_same_example_budget": len(
            {report["examples"] for report in arm_reports}
        )
        == 1,
        "all_families_selected": all(
            all(value > 0.0 for value in report["family_distribution"].values())
            for report in arm_reports
        ),
        "all_gaps_have_supervision_or_generated_contract": all(
            gap_rows[gap] > 0 or gap in generated_only_gaps
            for gap in GAP_NAMES
        ),
        "deterministic_seed_declared": isinstance(config["probe"]["seed"], int),
    }
    report = {
        "version": "nlp-data-efficient-learning-probe-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "config": str(args.config.relative_to(ROOT)),
        "corpus": corpus_name,
        "corpus_declared_tokens": corpus_config["tokens"],
        "train_sample_rows": len(raw_train),
        "train_rows_after_exact_near_dedup": len(corpus),
        "validation_sample_rows": len(validation),
        "test_rows_read": 0,
        "arms": arm_reports,
        "gap_training_rows": gap_rows,
        "source_quality": quality,
        "forgetting": {
            "status": "deferred",
            "reason": "requires the same neural arm before and after equal-step training",
        },
        "executable_generation": {
            "status": "deferred",
            "reason": "requires generated code plus frozen compile/runtime tests",
        },
        "evidence_boundary": config["evidence_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
