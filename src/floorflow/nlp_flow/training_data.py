"""Deterministic streaming and packing for frozen NLP tokenizers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, get_worker_info

from floorflow.nlp_flow.model.task_metadata import family_id, language_id


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"JSONL row {line_number} must be an object")
            if not isinstance(payload.get("text"), str):
                raise TypeError(f"JSONL row {line_number} needs text")
            yield payload


def decode_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if add_bos:
        if bos is None:
            raise ValueError("tokenizer has no <bos> token")
        ids.insert(0, bos)
    if add_eos:
        if eos is None:
            raise ValueError("tokenizer has no <eos> token")
        ids.append(eos)
    return ids


def render_training_text(
    row: dict,
    *,
    text_mode: str,
    families: frozenset[str] | None = None,
) -> str | None:
    if families is not None and str(row.get("family")) not in families:
        return None
    text = str(row["text"]).strip()
    if not text:
        return None
    if text_mode == "raw":
        return text
    if text_mode != "chat":
        raise ValueError(f"unknown text mode: {text_mode}")
    family = str(row.get("family", "general_language"))
    if family == "chat_instruction":
        return text
    if family == "general_language":
        topic = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0][:180]
        return f"User: Tell me a short story about: {topic}\nAssistant: {text}\n"
    if family == "code":
        lines = text.splitlines()
        prompt = lines[0][:300] if lines else ""
        target = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
        language = str(row.get("language", "code"))
        return (
            f"User: Continue this {language} code and return only code.\n"
            f"Code:\n{prompt}\nAssistant:\n{target}\n"
        )
    if family == "model_architecture":
        return (
            "User: Explain the following model-architecture note clearly.\n"
            f"Note:\n{text}\nAssistant:\n{text}\n"
        )
    return (
        "User: Give an engineering assistant response for this system material.\n"
        f"Material:\n{text}\nAssistant:\n{text}\n"
    )


def training_validity_labels(row: dict) -> tuple[bool, int]:
    """Return surface validity and execution validity (-1 means unknown)."""

    surface = bool(row.get("surface_valid", True))
    execution = int(row.get("execution_valid", -1))
    if execution not in {-1, 0, 1}:
        raise ValueError("execution_valid must be -1, 0, or 1")
    return surface, execution


def packed_token_batches(
    path: str | Path | Sequence[str | Path],
    tokenizer: Tokenizer,
    *,
    context_length: int,
    max_sequences: int | None = None,
    drop_last: bool = True,
    worker_id: int = 0,
    worker_count: int = 1,
    text_mode: str = "raw",
    families: frozenset[str] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Pack round-robin shard rows into fixed-length causal-LM sequences."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")
    if worker_count <= 0 or not 0 <= worker_id < worker_count:
        raise ValueError("worker_id must be in [0, worker_count)")
    paths = (
        (Path(path),)
        if isinstance(path, (str, Path))
        else tuple(Path(item) for item in path)
    )
    if not paths:
        raise ValueError("at least one token source is required")
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> token")
    buffer: list[int] = []
    surface_buffer: list[bool] = []
    execution_buffer: list[int] = []
    family_buffer: list[int] = []
    language_buffer: list[int] = []
    produced = 0
    row_index = 0
    iterators = [iter_jsonl(source_path) for source_path in paths]
    active = set(range(len(paths)))
    while active:
        for source_index in tuple(active):
            try:
                row = next(iterators[source_index])
            except StopIteration:
                active.remove(source_index)
                continue
            current_index = row_index
            row_index += 1
            if current_index % worker_count != worker_id:
                continue
            text = render_training_text(
                row,
                text_mode=text_mode,
                families=families,
            )
            if text is None:
                continue
            token_ids = decode_text(text, tokenizer)
            surface_valid, execution_valid = training_validity_labels(row)
            row_family = family_id(row.get("family"))
            row_language = language_id(row.get("language"))
            buffer.extend(token_ids)
            surface_buffer.extend([surface_valid] * len(token_ids))
            execution_buffer.extend([execution_valid] * len(token_ids))
            family_buffer.extend([row_family] * len(token_ids))
            language_buffer.extend([row_language] * len(token_ids))
            while len(buffer) >= context_length:
                ids = buffer[:context_length]
                del buffer[:context_length]
                surface = surface_buffer[:context_length]
                del surface_buffer[:context_length]
                execution = execution_buffer[:context_length]
                del execution_buffer[:context_length]
                family = family_buffer[:context_length]
                del family_buffer[:context_length]
                language = language_buffer[:context_length]
                del language_buffer[:context_length]
                yield {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(ids, dtype=torch.long),
                    "surface_valid": torch.tensor(surface, dtype=torch.bool),
                    "execution_valid": torch.tensor(execution, dtype=torch.long),
                    "family_ids": torch.tensor(family, dtype=torch.long),
                    "language_ids": torch.tensor(language, dtype=torch.long),
                }
                produced += 1
                if max_sequences is not None and produced >= max_sequences:
                    return
    if buffer and not drop_last and (max_sequences is None or produced < max_sequences):
        active = len(buffer)
        padding = context_length - active
        input_ids = buffer + [eos] * padding
        labels = buffer + [-100] * padding
        yield {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "surface_valid": torch.tensor(
                surface_buffer + [False] * padding,
                dtype=torch.bool,
            ),
            "execution_valid": torch.tensor(
                execution_buffer + [-1] * padding,
                dtype=torch.long,
            ),
            "family_ids": torch.tensor(
                family_buffer + [0] * padding,
                dtype=torch.long,
            ),
            "language_ids": torch.tensor(
                language_buffer + [0] * padding,
                dtype=torch.long,
            ),
        }


class PackedTokenDataset(IterableDataset):
    """Worker-aware streaming dataset over one frozen split."""

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        tokenizer_path: str | Path,
        *,
        context_length: int,
        max_sequences: int | None = None,
        drop_last: bool = True,
        text_mode: str = "raw",
        families: frozenset[str] | None = None,
    ) -> None:
        super().__init__()
        self.paths = (
            (Path(path),)
            if isinstance(path, (str, Path))
            else tuple(Path(item) for item in path)
        )
        self.tokenizer_path = Path(tokenizer_path)
        self.context_length = int(context_length)
        self.max_sequences = max_sequences
        self.drop_last = bool(drop_last)
        self.text_mode = text_mode
        self.families = families

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        yield from packed_token_batches(
            self.paths,
            tokenizer,
            context_length=self.context_length,
            max_sequences=self.max_sequences,
            drop_last=self.drop_last,
            worker_id=worker_id,
            worker_count=worker_count,
            text_mode=self.text_mode,
            families=self.families,
        )


__all__ = [
    "PackedTokenDataset",
    "decode_text",
    "iter_jsonl",
    "packed_token_batches",
    "render_training_text",
    "training_validity_labels",
]
