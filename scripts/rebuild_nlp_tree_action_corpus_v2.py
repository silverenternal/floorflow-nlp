"""Rebuild corrected train-only tree actions without parsing holdout rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.real_corpus_action_data import (
    build_real_corpus_action_dataset,
    write_real_corpus_action_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
_TRAIN_MARKER = re.compile(r'"split"\s*:\s*"([^"]+)"')


def _read_train_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = _TRAIN_MARKER.search(line)
            if match is None or match.group(1) != "train":
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("split metadata changed while reading train-only AST rows")
            rows.append(row)
    return rows


def build_train_corpus(config_path: Path, *, root: Path = ROOT) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version") != "nlp-tree-action-corpus-v2-train-only-rebuild":
        raise ValueError("unsupported corrected tree-action corpus configuration")
    source_path = root / str(config["ast_corpus"])
    train_rows = _read_train_rows(source_path)
    if not train_rows or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("corrected action corpus requires non-empty train-only input")

    prompt_config = json.loads(
        (root / str(config["prompt_contract"])).read_text(encoding="utf-8")
    )
    grammars = {
        language: load_ast_grammar(root / str(config["grammar"]), language)
        for language in ("python", "rust", "cpp")
    }
    action_rows, audit = build_real_corpus_action_dataset(
        train_rows,
        grammars,
        prompt_config["prompt"],
        prompt_config["fragments"],
    )
    if any(row.split != "train" for row in action_rows):
        raise ValueError("builder emitted a non-train action row")
    output_path = root / str(config["output"])
    corpus_sha256 = write_real_corpus_action_dataset(action_rows, output=output_path)
    return {
        "version": "nlp-tree-action-corpus-audit-v2",
        "status": audit["status"],
        "source_path": str(source_path.relative_to(root)),
        "source_train_rows": len(train_rows),
        "source_train_identity_sha256": hashlib.sha256(
            "\n".join(str(row["case_id"]) + ":" + str(row["source_sha256"]) for row in train_rows).encode()
        ).hexdigest(),
        "input_split_scope": "train only; validation and test rows were not JSON-decoded",
        "output": str(output_path.relative_to(root)),
        "corpus_sha256": corpus_sha256,
        "audit": audit,
        "evidence_boundary": config["evidence_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nlp_tree_action_corpus_v2.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_train_corpus(args.config.resolve())
    report_path = ROOT / str(config["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
