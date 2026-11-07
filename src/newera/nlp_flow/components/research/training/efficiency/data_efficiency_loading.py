"""Bounded deterministic JSONL loaders for NLP data-efficiency experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newera.nlp_flow.components.research.training.efficiency.data_efficiency import CorpusExample


def load_jsonl_examples(
    paths: Sequence[str | Path],
    *,
    max_rows: int,
    example_prefix: str = "example",
) -> tuple[CorpusExample, ...]:
    """Load a bounded deterministic prefix for CPU experiments."""

    from newera.nlp_flow.components.research.training.efficiency.data_efficiency import FAMILY_NAMES, make_example

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    examples: list[CorpusExample] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(examples) >= max_rows:
                    return tuple(examples)
                row = json.loads(line)
                rows_family = str(row.get("family", ""))
                if rows_family not in FAMILY_NAMES:
                    continue
                examples.append(
                    make_example(
                        example_id=str(
                            row.get("id", f"{path.name}:{line_number}")
                        ),
                        text=str(row["text"]),
                        family=rows_family,
                        language=str(row.get("language", "text")),
                    )
                )
    return tuple(examples)


def load_balanced_jsonl_examples(
    paths: Sequence[str | Path],
    *,
    max_per_family: int,
    max_scan_rows: int,
    max_per_code_language: dict[str, int] | None = None,
) -> tuple[CorpusExample, ...]:
    """Scan shards until each family reaches a bounded deterministic quota."""

    from newera.nlp_flow.components.research.training.efficiency.data_efficiency import FAMILY_NAMES, make_example

    if max_per_family <= 0 or max_scan_rows <= 0:
        raise ValueError("family quota and scan limit must be positive")
    quotas = {family: int(max_per_family) for family in FAMILY_NAMES}
    language_quotas = {
        str(language): int(quota)
        for language, quota in (max_per_code_language or {}).items()
    }
    if language_quotas:
        quotas["code"] = sum(language_quotas.values())
    examples: list[CorpusExample] = []
    scanned = 0
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if scanned >= max_scan_rows or not any(quotas.values()):
                    return tuple(examples)
                scanned += 1
                row = json.loads(line)
                family = str(row.get("family", ""))
                if family not in quotas or quotas[family] <= 0:
                    continue
                language = str(row.get("language", "text"))
                if (
                    family == "code"
                    and language_quotas
                    and language_quotas.get(language, 0) <= 0
                ):
                    continue
                examples.append(
                    make_example(
                        example_id=str(
                            row.get("id", f"{path.name}:{line_number}")
                        ),
                        text=str(row["text"]),
                        family=family,
                        language=language,
                    )
                )
                quotas[family] -= 1
                if family == "code" and language_quotas:
                    language_quotas[language] -= 1
    return tuple(examples)


__all__ = ["load_balanced_jsonl_examples", "load_jsonl_examples"]
