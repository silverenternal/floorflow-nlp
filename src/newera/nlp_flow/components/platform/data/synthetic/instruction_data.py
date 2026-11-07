"""Row-aligned instruction samples with answer-only supervision."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, get_worker_info

from newera.nlp_flow.model.components.tasks.output.benchmark_proxy import SCALAR_PROXY_TARGETS
from newera.nlp_flow.model.components.tasks.metadata.task_metadata import family_id, language_id

_ROLE_PATTERN = re.compile(r"(?m)^(User|Assistant|Tool|Code|System):")
_ROLE_IDS = {
    "User": 0,
    "Assistant": 1,
    "Tool": 2,
    "Code": 3,
    "System": 4,
}


def role_ids_for_offsets(
    text: str,
    offsets: list[tuple[int, int]],
) -> list[int]:
    matches = list(_ROLE_PATTERN.finditer(text))
    spans = []
    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        spans.append((start, end, _ROLE_IDS[match.group(1)]))
    role_ids = []
    cursor = 0
    for start, _end in offsets:
        while cursor + 1 < len(spans) and start >= spans[cursor + 1][0]:
            cursor += 1
        role_ids.append(spans[cursor][2] if spans else 0)
    return role_ids


def encode_chat_prompt(
    text: str,
    tokenizer: Tokenizer,
    *,
    max_tokens: int,
) -> tuple[list[int], list[int]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = list(encoding.ids)
    turn_ids = role_ids_for_offsets(text, encoding.offsets)
    if len(ids) > max_tokens:
        preserve_head = min(16, max_tokens // 3)
        preserve_tail = max_tokens - preserve_head
        ids = ids[:preserve_head] + ids[-preserve_tail:]
        turn_ids = (
            turn_ids[:preserve_head]
            + turn_ids[-preserve_tail:]
        )
    return ids, turn_ids


class RowAlignedInstructionDataset(IterableDataset):
    """Keep every prompt and response in one bounded, independently aligned row."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        tokenizer_path: str | Path,
        *,
        context_length: int,
        max_sequences: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if context_length <= 1:
            raise ValueError("context_length must exceed one token")
        if max_sequences is not None and max_sequences <= 0:
            raise ValueError("max_sequences must be positive")
        self.paths = (
            (Path(paths),)
            if isinstance(paths, (str, Path))
            else tuple(Path(path) for path in paths)
        )
        self.tokenizer_path = Path(tokenizer_path)
        self.context_length = int(context_length)
        self.max_sequences = max_sequences
        self.seed = int(seed)

    def _load_rows(self) -> list[dict]:
        rows = []
        for path in self.paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(
                            f"{path}:{line_number} must contain a JSON object"
                        )
                    if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                        raise ValueError(
                            f"{path}:{line_number} requires a non-empty prompt"
                        )
                    if (
                        not isinstance(row.get("response"), str)
                        or not row["response"].strip()
                    ):
                        raise ValueError(
                            f"{path}:{line_number} requires a non-empty response"
                        )
                    rows.append(row)
        return rows

    @staticmethod
    def _role_ids_for_prefix(
        text: str,
        offsets: list[tuple[int, int]],
    ) -> list[int]:
        return role_ids_for_offsets(text, offsets)

    def _encode(self, row: dict, tokenizer: Tokenizer) -> dict[str, torch.Tensor] | None:
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")
        if bos is None or eos is None:
            raise ValueError("tokenizer needs <bos> and <eos>")
        prefix = (
            f"{str(row['prompt']).strip()}\nAssistant:"
            if row.get("role_formatted") is True
            else f"User: {str(row['prompt']).strip()}\nAssistant:"
        )
        prefix_encoding = tokenizer.encode(prefix, add_special_tokens=False)
        prefix_token_ids = prefix_encoding.ids
        response_ids = [
            *tokenizer.encode(
                str(row["response"]).strip(),
                add_special_tokens=False,
            ).ids,
            eos,
        ]
        if len(response_ids) + 1 > self.context_length:
            return None
        prefix_budget = self.context_length - len(response_ids) - 1
        prefix_role_ids = self._role_ids_for_prefix(
            prefix,
            prefix_encoding.offsets,
        )
        if len(prefix_token_ids) > prefix_budget:
            preserve_head = min(16, max(0, prefix_budget // 3))
            preserve_tail = prefix_budget - preserve_head
            prefix_token_ids = (
                prefix_token_ids[:preserve_head]
                + prefix_token_ids[-preserve_tail:]
            )
            prefix_role_ids = (
                prefix_role_ids[:preserve_head]
                + prefix_role_ids[-preserve_tail:]
            )
        prefix_ids = [bos, *prefix_token_ids]
        input_ids = prefix_ids + response_ids
        labels = [-100] * len(prefix_ids) + response_ids
        attention_mask = [True] * len(input_ids)
        turn_ids = [0, *prefix_role_ids] + [1] * len(response_ids)
        response_start = len(prefix_ids)
        continue_targets = [0.0] * len(input_ids)
        for index in range(response_start, len(input_ids) - 1):
            continue_targets[index] = 1.0
        reply_mode, language, validity = self._control_targets(row)
        row_family = family_id(row.get("family", "chat_instruction"))
        row_language = (
            language_id("python")
            if language == 1
            else language_id("rust")
            if language == 2
            else language_id("cpp")
            if language == 3
            else language_id("text")
        )
        execution = float(int(row.get("execution_valid", -1)))
        reply_mode_ids = [-100] * len(prefix_ids) + [reply_mode] * len(response_ids)
        language_ids = [-100] * len(prefix_ids) + [language] * len(response_ids)
        turn_boundary_targets = [-100] * len(input_ids)
        turn_boundary_targets[-2] = 1
        validity_targets = (
            [0.0] * (len(prefix_ids) * 3)
            + validity * len(response_ids)
        )
        validity_mask = (
            [False] * (len(prefix_ids) * 3)
            + [True] * (len(response_ids) * 3)
        )
        padding = self.context_length - len(input_ids)
        if padding:
            input_ids.extend([eos] * padding)
            labels.extend([-100] * padding)
            attention_mask.extend([False] * padding)
            turn_ids.extend([0] * padding)
            continue_targets.extend([0.0] * padding)
            reply_mode_ids.extend([-100] * padding)
            language_ids.extend([-100] * padding)
            turn_boundary_targets.extend([-100] * padding)
            validity_targets.extend([0.0] * (padding * 3))
            validity_mask.extend([False] * (padding * 3))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "turn_ids": torch.tensor(turn_ids, dtype=torch.long),
            "family_ids": torch.full(
                (self.context_length,),
                row_family,
                dtype=torch.long,
            ),
            "language_ids": torch.full(
                (self.context_length,),
                row_language,
                dtype=torch.long,
            ),
            "continue_targets": torch.tensor(
                continue_targets,
                dtype=torch.float32,
            ),
            "reply_mode_targets": torch.tensor(
                reply_mode_ids,
                dtype=torch.long,
            ),
            "language_targets": torch.tensor(
                language_ids,
                dtype=torch.long,
            ),
            "turn_boundary_targets": torch.tensor(
                turn_boundary_targets,
                dtype=torch.long,
            ),
            "validity_targets": torch.tensor(
                validity_targets,
                dtype=torch.float32,
            ).reshape(self.context_length, 3),
            "validity_mask": torch.tensor(
                validity_mask,
                dtype=torch.bool,
            ).reshape(self.context_length, 3),
            **self._proxy_target_tensors(
                language=language,
                execution=execution,
            ),
            "surface_valid": torch.tensor(
                [bool(row.get("surface_valid", True))] * self.context_length,
                dtype=torch.bool,
            ),
            "execution_valid": torch.tensor(
                [int(row.get("execution_valid", -1))] * self.context_length,
                dtype=torch.long,
            ),
        }

    @staticmethod
    def _proxy_target_tensors(
        *,
        language: int,
        execution: float,
    ) -> dict[str, torch.Tensor]:
        values = {
            name: -1.0
            for name in SCALAR_PROXY_TARGETS
        }
        if execution >= 0:
            if language == 1:
                values.update(
                    {
                        "compile_python": execution,
                        "execution_python": execution,
                        "runtime_success": execution,
                        "calibration": execution,
                    }
                )
            elif language == 2:
                values.update(
                    {
                        "compile_rust": execution,
                        "execution_rust": execution,
                        "runtime_success": execution,
                        "calibration": execution,
                    }
                )
            elif language == 3:
                values.update(
                    {
                        "compile_cpp": execution,
                        "execution_cpp": execution,
                        "runtime_success": execution,
                        "calibration": execution,
                    }
                )
        return {
            f"proxy_{name}": torch.tensor(value, dtype=torch.float32)
            for name, value in values.items()
        }

    @staticmethod
    def _control_targets(row: dict) -> tuple[int, int, list[float]]:
        response = str(row["response"]).strip()
        valid_code = bool(row.get("execution_valid", -1) == 1)
        valid_json = False
        if response.startswith(("{", "[")):
            try:
                json.loads(response)
            except json.JSONDecodeError:
                pass
            else:
                valid_json = True
        if valid_json:
            return 2, 0, [0.0, 1.0, 0.0]
        if response.startswith("def "):
            return 1, 1, [float(valid_code), 0.0, 0.0]
        if response.startswith("fn "):
            return 1, 2, [float(valid_code), 0.0, 0.0]
        if response.startswith("#include"):
            return 1, 3, [float(valid_code), 0.0, 0.0]
        if "AST" in response or "ast" in response:
            return 3, 0, [0.0, 0.0, 1.0]
        return 0, 0, [0.0, 0.0, 0.0]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        rows = self._load_rows()
        indices = [
            index
            for index, row in enumerate(rows)
            for _repeat in range(int(row.get("replay_count", 1)))
        ]
        random.Random(self.seed).shuffle(indices)
        produced = 0
        for position, index in enumerate(indices):
            if position % worker_count != worker_id:
                continue
            encoded = self._encode(rows[index], tokenizer)
            if encoded is None:
                continue
            yield encoded
            produced += 1
            if self.max_sequences is not None and produced >= self.max_sequences:
                return


__all__ = [
    "RowAlignedInstructionDataset",
    "encode_chat_prompt",
    "role_ids_for_offsets",
]
