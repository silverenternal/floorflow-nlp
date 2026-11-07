"""Validate engineering runner manifests without executing commands or kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.research.evaluation.evidence.verifier_result_contract import preflight_runner_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = preflight_runner_manifest(manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"preflight_passed_no_execution", "blocked_gpu_gate"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
