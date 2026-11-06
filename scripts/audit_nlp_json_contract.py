#!/usr/bin/env python3
"""Audit the native 100%-JSON-validity contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.json_contract import validate_json_contract

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_json_contract_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    errors = validate_json_contract(payload)
    report = {
        "version": "nlp-json-contract-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "guarantee": payload["validity_contract"]["guarantee"],
        "strict_json_validity": payload["validity_contract"]["strict_json_parse_validity"],
        "schema_compliance": payload["validity_contract"]["schema_compliance"],
        "heads": payload["native_generation"]["typed_heads"],
        "posthoc_repair": payload["native_generation"]["posthoc_repair"],
        "retry": payload["native_generation"]["retry"],
        "evidence_boundary": payload["evidence_boundary"],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
