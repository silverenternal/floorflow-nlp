"""Standardized benchmark-proxy label rows for the joint trainer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from newera.nlp_flow.model.components.tasks.output.benchmark_proxy import (
    REGRESSION_PROXY_TARGETS,
    SCALAR_PROXY_TARGETS,
    TOKEN_PROXY_TARGETS,
)


@dataclass(frozen=True)
class ProxyLabelBatch:
    scalar_targets: dict[str, float]
    regression_targets: dict[str, float]
    token_targets: dict[str, torch.Tensor]
    token_masks: dict[str, torch.Tensor]
    regression_mask: torch.Tensor
    test_output_token: torch.Tensor | None = None
    test_output_mask: torch.Tensor | None = None
    multiple_choice_target: int | None = None
    choice_embeddings: torch.Tensor | None = None
    case_id: str = ""
    source_split: str = "train"


class ProxyLabelAdapter:
    """Converts training/evaluation rows into the common proxy target schema."""

    TRAIN_SPLITS = frozenset({"train", "validation", "heldout_validation"})

    def __init__(self, *, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = int(sequence_length)

    def adapt(self, row: Mapping[str, object]) -> ProxyLabelBatch:
        split = str(row.get("split", "train"))
        if split not in self.TRAIN_SPLITS:
            raise ValueError(
                f"proxy labels from split {split!r} cannot enter training"
            )
        scalar_input = row.get("scalar_targets", {})
        regression_input = row.get("regression_targets", {})
        token_input = row.get("token_targets", {})
        if not isinstance(scalar_input, Mapping):
            raise TypeError("scalar_targets must be an object")
        if not isinstance(regression_input, Mapping):
            raise TypeError("regression_targets must be an object")
        if not isinstance(token_input, Mapping):
            raise TypeError("token_targets must be an object")
        scalar = {
            name: float(scalar_input.get(name, -1.0))
            for name in SCALAR_PROXY_TARGETS
        }
        regression = {
            name: float(regression_input.get(name, -1.0))
            for name in REGRESSION_PROXY_TARGETS
        }
        regression_mask = torch.tensor(
            [regression[name] >= 0.0 for name in REGRESSION_PROXY_TARGETS],
            dtype=torch.bool,
        )
        token_targets = {}
        token_masks = {}
        for name in TOKEN_PROXY_TARGETS:
            raw = token_input.get(name)
            values = self._token_values(raw)
            token_targets[name] = values
            token_masks[name] = values.ge(0)
        test_output = row.get("test_output_token")
        test_tensor = (
            None
            if test_output is None
            else self._token_values(test_output)
        )
        test_mask = (
            None
            if test_tensor is None
            else test_tensor.ge(0)
        )
        choice = row.get("multiple_choice_target")
        choice_embeddings = self._choice_embeddings(
            row.get("choice_embeddings")
        )
        if choice is not None and choice_embeddings is None:
            raise ValueError(
                "multiple_choice_target requires choice_embeddings"
            )
        return ProxyLabelBatch(
            scalar_targets=scalar,
            regression_targets=regression,
            token_targets=token_targets,
            token_masks=token_masks,
            regression_mask=regression_mask,
            test_output_token=test_tensor,
            test_output_mask=test_mask,
            multiple_choice_target=None if choice is None else int(choice),
            choice_embeddings=choice_embeddings,
            case_id=str(row.get("case_id", "")),
            source_split=split,
        )

    def _token_values(self, raw: object) -> torch.Tensor:
        values = [] if raw is None else list(raw)
        if len(values) > self.sequence_length:
            raise ValueError("proxy token targets exceed sequence length")
        values = values + [-1] * (self.sequence_length - len(values))
        return torch.tensor(values, dtype=torch.long)

    def _choice_embeddings(self, raw: object) -> torch.Tensor | None:
        if raw is None:
            return None
        values = torch.tensor(raw, dtype=torch.float32)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] <= 0:
            raise ValueError("choice_embeddings must be [choices, hidden]")
        return values


def collate_proxy_labels(
    rows: Sequence[ProxyLabelBatch],
) -> dict[str, torch.Tensor]:
    """Collate proxy labels without converting unknown labels into targets."""

    if not rows:
        raise ValueError("at least one proxy label row is required")
    if any(row.source_split == "test" for row in rows):
        raise ValueError("test proxy labels cannot be collated for training")
    token_names = tuple(rows[0].token_targets)
    if any(tuple(row.token_targets) != token_names for row in rows):
        raise ValueError("proxy token target schemas do not align")

    batch = {
        f"proxy_{name}": torch.tensor(
            [row.scalar_targets[name] for row in rows],
            dtype=torch.float32,
        )
        for name in rows[0].scalar_targets
    }
    for name in rows[0].regression_targets:
        batch[f"proxy_regression_{name}"] = torch.tensor(
            [row.regression_targets[name] for row in rows],
            dtype=torch.float32,
        )
    batch["proxy_regression_mask"] = torch.stack(
        [row.regression_mask for row in rows]
    )
    for name in token_names:
        batch[f"proxy_token_{name}"] = torch.stack(
            [row.token_targets[name] for row in rows]
        )
        batch[f"proxy_token_mask_{name}"] = torch.stack(
            [row.token_masks[name] for row in rows]
        )

    with_test_output = [row for row in rows if row.test_output_token is not None]
    if with_test_output:
        if len(with_test_output) != len(rows):
            raise ValueError(
                "test-output labels must be present or absent for every row"
            )
        batch["proxy_test_output_token"] = torch.stack(
            [
                row.test_output_token for row in rows
                if row.test_output_token is not None
            ]
        )
        batch["proxy_test_output_mask"] = torch.stack(
            [
                row.test_output_mask for row in rows
                if row.test_output_mask is not None
            ]
        )

    with_choices = [row for row in rows if row.multiple_choice_target is not None]
    if with_choices:
        if len(with_choices) != len(rows):
            raise ValueError(
                "multiple-choice labels must be present for every row"
            )
        embeddings = [
            row.choice_embeddings for row in rows
            if row.choice_embeddings is not None
        ]
        if len(embeddings) != len(rows):
            raise ValueError("every multiple-choice row needs choice embeddings")
        shape = embeddings[0].shape
        if any(item.shape != shape for item in embeddings):
            raise ValueError("multiple-choice candidate shapes do not align")
        batch["proxy_choice_embeddings"] = torch.stack(embeddings)
        batch["proxy_multiple_choice"] = torch.tensor(
            [row.multiple_choice_target for row in rows],
            dtype=torch.long,
        )
    return batch


def load_proxy_label_rows(
    paths: Sequence[str | Path],
    adapter: ProxyLabelAdapter,
) -> tuple[ProxyLabelBatch, ...]:
    rows = []
    for path_value in paths:
        with Path(path_value).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise TypeError(
                        f"{path_value}:{line_number} must be an object"
                    )
                rows.append(adapter.adapt(payload))
    return tuple(rows)


__all__ = [
    "ProxyLabelAdapter",
    "ProxyLabelBatch",
    "collate_proxy_labels",
    "load_proxy_label_rows",
]
