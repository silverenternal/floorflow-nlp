#!/usr/bin/env python3
"""Run matched validation search over compact NLP architecture candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from newera.nlp_flow.components.research.architecture.search.architecture_search import (
    ArchitectureCandidateFactory,
    EfficientArchitectureSearch,
    SearchSpace,
    freeze_selected_architecture,
)
from newera.nlp_flow.components.research.architecture.search.architecture_search_trainer import (
    TorchValidationArchitectureEvaluator,
)

ROOT = Path(__file__).resolve().parents[4]


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/models_training/architecture/search/nlp_architecture_search.json",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if args.candidate_limit is not None:
        if args.candidate_limit <= 0:
            raise ValueError("candidate-limit must be positive")
        payload["measurement"]["candidate_limit"] = args.candidate_limit
    space = SearchSpace.from_payload(payload, root=ROOT)
    candidates = ArchitectureCandidateFactory(space).generate()
    if not candidates:
        raise RuntimeError("search space produced no budget-feasible candidate")
    output_path = _resolve(args.output or payload["report"])
    candidate_payload = [candidate.as_dict() for candidate in candidates]
    if args.contract_only:
        report = {
            "version": "nlp-architecture-search-contract-v1",
            "status": "pass",
            "test_read": False,
            "candidate_count": len(candidates),
            "candidates": candidate_payload,
            "evidence_boundary": (
                "Candidate construction and cost contract only; no training, "
                "capability, or architecture-selection result."
            ),
        }
    else:
        training = payload["training"]
        halving = payload["halving"]
        evaluator = TorchValidationArchitectureEvaluator(
            train_paths=[_resolve(path) for path in training["train"]],
            validation_paths=[
                _resolve(path) for path in training["validation"]
            ],
            tokenizer_path=_resolve(training["tokenizer"]),
            device=args.device or training["device"],
            batch_size=int(training["batch_size"]),
            sequence_length=int(training["sequence_length"]),
            train_batches=int(training["train_batches"]),
            validation_batches=int(training["validation_batches"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            optimizer_name=str(
                training.get("optimizer", "schedule_free_adamw")
            ),
            seed=int(training["seed"]),
            precision=str(training["precision"]),
            text_mode=str(training["text_mode"]),
            emergence_weight=float(training["emergence_loss_weight"]),
            dynamic_depth_weight=float(training["dynamic_depth_weight"]),
            dynamic_target_depth=float(training["dynamic_target_depth"]),
        )
        started = time.perf_counter()
        report = EfficientArchitectureSearch(
            candidates=candidates,
            token_budgets=tuple(halving["token_budgets"]),
            keep_fraction=float(halving["keep_fraction"]),
            quality_plateau=float(halving["quality_plateau"]),
            required_metrics=tuple(payload["protocol"]["promotion_requires"]),
        ).run(evaluator)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["quality_metric"] = payload["protocol"]["quality_metric"]
        report["promotion_requires"] = payload["protocol"][
            "promotion_requires"
        ]
        report["candidate_catalog"] = candidate_payload
        report["config_path"] = str(args.config.relative_to(ROOT))
        freeze = payload["freeze"]
        selected_path = _resolve(freeze["selected_config"])
        report["selected_config_path"] = str(selected_path.relative_to(ROOT))
        freeze_selected_architecture(
            report,
            config_path=selected_path,
            report_path=output_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
