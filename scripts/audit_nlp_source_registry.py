#!/usr/bin/env python3
"""Audit source revisions, licenses, and family coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.source_registry import validate_source_registry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/nlp_source_registry_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate_source_registry(payload)
    report = {
        "version": "nlp-source-registry-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "registry": str(args.registry.relative_to(ROOT)),
        "source_count": len(payload.get("sources", [])),
        "sources": [
            {
                "id": source["id"],
                "revision": source["revision"],
                "license": source["license"],
                "usage": source["usage"],
            }
            for source in payload.get("sources", [])
        ],
        "evidence_boundary": "License/source registry only; no content fetched or cleaned.",
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
