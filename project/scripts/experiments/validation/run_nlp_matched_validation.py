"""Create an auditable dry-run ledger for matched NLP control comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("execution", {}).get("run_enabled") is not False:
        raise ValueError("matched planning tool accepts dry-run-only configs")
    controls = config.get("controls")
    seeds = config.get("seed_set")
    if not isinstance(controls, list) or not controls or not isinstance(seeds, list) or not seeds:
        raise ValueError("matched config needs controls and seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("matched seed set must be unique")
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ledger = []
    for control in controls:
        if control.get("enabled") is not True:
            continue
        for seed in seeds:
            ledger.append(
                {
                    "pair_id": f"{control['id']}:{seed}",
                    "control_id": control["id"],
                    "seed": seed,
                    "config_sha256": digest,
                    "tokens_seen": None,
                    "resident_parameters": None,
                    "total_compute": None,
                    "actual_overhead": None,
                    "feasibility": "not_assessed_dry_run",
                    "measured": False,
                }
            )
    return {
        "version": "nlp-matched-validation-dry-run-v1",
        "status": "planned_not_executed",
        "split": config.get("evaluation_split"),
        "execution_performed": False,
        "gpu_used": False,
        "ledger": ledger,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = build_dry_run(config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "ledger"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
