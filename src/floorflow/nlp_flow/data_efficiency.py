"""Deterministic gap-aware curriculum and replay utilities for NLP training."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from floorflow.nlp_flow.data_efficiency_loading import (
    load_balanced_jsonl_examples,
    load_jsonl_examples,
)

FAMILY_NAMES = ("general_language", "code", "engineering_acceleration", "model_architecture")
GAP_NAMES = (
    "json_validity", "ast_structure", "python_execution", "rust_execution",
    "cpp_execution", "engineering_acceleration", "model_architecture",
    "language_retention", "compositional_emergence",
)
TRAINING_ARMS = ("uniform", "family_balanced", "curriculum_efficient")
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", re.UNICODE)


@dataclass(frozen=True)
class CorpusExample:
    id: str
    text: str
    family: str
    language: str
    token_count: int
    difficulty: int
    gaps: tuple[str, ...]
    content_hash: str
    near_hash: int


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start: float
    end: float
    difficulty_center: float
    family_weights: dict[str, float]
    replay_fraction: float
    hard_fraction: float


@dataclass(frozen=True)
class SelectionMetrics:
    examples: int
    tokens: int
    unique_examples: int
    family_distribution: dict[str, float]
    gap_coverage: dict[str, float]
    mean_difficulty: float
    duplicate_exposures: int


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(_normalise_text(text).encode("utf-8")).hexdigest()


def simhash64(text: str) -> int:
    """Return a deterministic 64-bit SimHash over lower-cased token trigrams."""

    tokens = [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text[:32768])
    ][:4096]
    if not tokens:
        return 0
    features = (
        tokens
        if len(tokens) < 3
        else ["\x1f".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    )
    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            for feature in features
        ),
        dtype=np.uint64,
        count=len(features),
    )
    shifts = np.arange(64, dtype=np.uint64)
    bits = ((hashes[:, None] >> shifts[None, :]) & 1).astype(np.int16)
    weights = (bits * 2 - 1).sum(axis=0)
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def infer_gap_tags(
    *,
    text: str,
    family: str,
    language: str,
) -> tuple[str, ...]:
    tags: set[str] = set()
    if family == "general_language":
        tags.add("language_retention")
    if family == "engineering_acceleration":
        tags.add("engineering_acceleration")
    if family == "model_architecture":
        tags.add("model_architecture")
    if family == "code":
        if language == "python":
            tags.add("python_execution")
        elif language == "rust":
            tags.add("rust_execution")
        elif language in {"cpp", "c"}:
            tags.add("cpp_execution")
        if any(marker in text for marker in ("def ", "class ", "fn ", "struct ", "{")):
            tags.add("ast_structure")
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
        except (json.JSONDecodeError, RecursionError):
            pass
        else:
            tags.add("json_validity")
    if (
        ("User:" in text and "Assistant:" in text and "Code:" in text)
        or len(tags) >= 3
    ):
        tags.add("compositional_emergence")
    return tuple(sorted(tags))


def difficulty_rank(text: str, token_count: int) -> int:
    structural_markers = sum(
        marker in text
        for marker in ("\nclass ", "\ndef ", "\nfn ", "\nstruct ", "test_", "await ")
    )
    if token_count < 80 and structural_markers == 0:
        return 0
    if token_count < 300 and structural_markers <= 1:
        return 1
    if token_count < 1000 and structural_markers <= 3:
        return 2
    return 3


def make_example(
    *,
    example_id: str,
    text: str,
    family: str,
    language: str = "text",
    token_count: int | None = None,
) -> CorpusExample:
    if family not in FAMILY_NAMES:
        raise ValueError(f"unsupported family: {family}")
    if not text.strip():
        raise ValueError("example text cannot be empty")
    count = (
        int(token_count)
        if token_count is not None
        else len(_TOKEN_PATTERN.findall(text))
    )
    if count <= 0:
        raise ValueError("token_count must be positive")
    return CorpusExample(
        id=str(example_id),
        text=text,
        family=family,
        language=language,
        token_count=count,
        difficulty=difficulty_rank(text, count),
        gaps=infer_gap_tags(text=text, family=family, language=language),
        content_hash=content_hash(text),
        near_hash=simhash64(text),
    )


class ExposureDeduplicator:
    """Reject exact duplicates and bound near-duplicate exposure."""

    def __init__(self, *, near_threshold: int = 3, max_near_copies: int = 1) -> None:
        if not 0 <= near_threshold <= 64:
            raise ValueError("near_threshold must be in [0, 64]")
        if max_near_copies <= 0:
            raise ValueError("max_near_copies must be positive")
        self.near_threshold = int(near_threshold)
        self.max_near_copies = int(max_near_copies)
        self._exact: set[str] = set()
        self._near: list[int] = []
        self._near_bands: list[dict[int, list[int]]] = [
            {} for _ in range(6)
        ]

    @staticmethod
    def _band_keys(value: int) -> tuple[int, ...]:
        return tuple(
            (value >> (band * 11)) & ((1 << 11) - 1)
            for band in range(6)
        )

    def _near_candidates(self, value: int) -> set[int]:
        if self.near_threshold >= 6:
            return set(range(len(self._near)))
        candidates: set[int] = set()
        for bucket, key in zip(self._near_bands, self._band_keys(value)):
            candidates.update(bucket.get(key, ()))
        return candidates

    def admit(self, example: CorpusExample) -> bool:
        if example.content_hash in self._exact:
            return False
        near_copies = sum(
            hamming_distance(
                example.near_hash,
                self._near[index],
            )
            <= self.near_threshold
            for index in self._near_candidates(example.near_hash)
        )
        if near_copies >= self.max_near_copies:
            return False
        self._exact.add(example.content_hash)
        index = len(self._near)
        self._near.append(example.near_hash)
        for bucket, key in zip(
            self._near_bands,
            self._band_keys(example.near_hash),
        ):
            bucket.setdefault(key, []).append(index)
        return True

    def filter(self, examples: Iterable[CorpusExample]) -> tuple[CorpusExample, ...]:
        return tuple(example for example in examples if self.admit(example))


class CurriculumScheduler:
    def __init__(self, stages: Sequence[CurriculumStage] | None = None) -> None:
        self.stages = tuple(stages or _default_stages())
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        if self.stages[0].start != 0.0 or self.stages[-1].end != 1.0:
            raise ValueError("curriculum must cover progress [0, 1]")
        for left, right in zip(self.stages, self.stages[1:]):
            if left.end != right.start:
                raise ValueError("curriculum stages must be contiguous")

    def at(self, progress: float) -> CurriculumStage:
        bounded = min(1.0, max(0.0, float(progress)))
        for stage in self.stages:
            if stage.start <= bounded < stage.end or (
                bounded == 1.0 and stage.end == 1.0
            ):
                return stage
        raise RuntimeError("curriculum lookup did not find a stage")


def _default_stages() -> tuple[CurriculumStage, ...]:
    return (
        CurriculumStage(
            "grammar_foundation",
            0.0,
            0.2,
            0.5,
            {
                "general_language": 0.45,
                "code": 0.35,
                "engineering_acceleration": 0.10,
                "model_architecture": 0.10,
            },
            0.05,
            0.10,
        ),
        CurriculumStage(
            "code_fluency",
            0.2,
            0.45,
            1.2,
            {
                "general_language": 0.20,
                "code": 0.55,
                "engineering_acceleration": 0.15,
                "model_architecture": 0.10,
            },
            0.10,
            0.20,
        ),
        CurriculumStage(
            "verifier_grounded",
            0.45,
            0.75,
            2.0,
            {
                "general_language": 0.15,
                "code": 0.45,
                "engineering_acceleration": 0.30,
                "model_architecture": 0.10,
            },
            0.15,
            0.30,
        ),
        CurriculumStage(
            "composition_replay",
            0.75,
            1.0,
            2.8,
            {
                "general_language": 0.15,
                "code": 0.35,
                "engineering_acceleration": 0.30,
                "model_architecture": 0.20,
            },
            0.20,
            0.35,
        ),
    )


class HardExampleMemory:
    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._examples: dict[str, CorpusExample] = {}
        self._losses: dict[str, float] = {}

    def observe(self, example: CorpusExample, loss: float) -> None:
        value = float(loss)
        if not math.isfinite(value):
            raise ValueError("loss must be finite")
        self._examples[example.id] = example
        self._losses[example.id] = max(value, self._losses.get(example.id, -math.inf))
        if len(self._losses) > self.capacity:
            lowest = min(self._losses, key=lambda key: (self._losses[key], key))
            self._losses.pop(lowest)
            self._examples.pop(lowest, None)

    def examples(self, limit: int | None = None) -> tuple[CorpusExample, ...]:
        size = self.capacity if limit is None else max(0, int(limit))
        keys = heapq.nlargest(
            size,
            self._losses,
            key=lambda key: (self._losses[key], key),
        )
        return tuple(self._examples[key] for key in keys)

    def __len__(self) -> int:
        return len(self._losses)


class EfficientCorpusSampler:
    """Select uniform, family-balanced, or curriculum-efficient examples."""

    def __init__(
        self,
        examples: Sequence[CorpusExample],
        *,
        arm: str,
        seed: int,
        curriculum: CurriculumScheduler | None = None,
        hard_memory: HardExampleMemory | None = None,
        gap_targets: dict[str, float] | None = None,
    ) -> None:
        if arm not in TRAINING_ARMS:
            raise ValueError(f"arm must be one of {TRAINING_ARMS}")
        if not examples:
            raise ValueError("examples cannot be empty")
        self.examples = tuple(examples)
        self.arm = arm
        self.seed = int(seed)
        self.curriculum = curriculum or CurriculumScheduler()
        self.hard_memory = hard_memory or HardExampleMemory()
        targets = gap_targets or {gap: 1.0 for gap in GAP_NAMES}
        self.gap_targets = {
            gap: float(targets.get(gap, 0.0)) for gap in GAP_NAMES
        }
        self._family_index: dict[str, tuple[CorpusExample, ...]] = {
            family: tuple(item for item in self.examples if item.family == family)
            for family in FAMILY_NAMES
        }
        self._selected = Counter()
        self._selected_gaps = Counter()
        self._selected_families = Counter()
        self._exposure = Counter()

    def _weighted_choice(
        self,
        candidates: Sequence[CorpusExample],
        weights: Sequence[float],
        rng: random.Random,
    ) -> CorpusExample:
        total = sum(weights)
        if total <= 0.0:
            return candidates[rng.randrange(len(candidates))]
        point = rng.random() * total
        cumulative = 0.0
        for example, weight in zip(candidates, weights):
            cumulative += weight
            if point <= cumulative:
                return example
        return candidates[-1]

    def _family(self, stage: CurriculumStage, rng: random.Random) -> str:
        families = [
            family
            for family in FAMILY_NAMES
            if self._family_index[family]
            and stage.family_weights.get(family, 0.0) > 0.0
        ]
        weights = [stage.family_weights[family] for family in families]
        return self._weighted_choice(
            [self._family_index[family][0] for family in families],
            weights,
            rng,
        ).family

    def _score(
        self,
        example: CorpusExample,
        stage: CurriculumStage,
    ) -> float:
        family_target = stage.family_weights.get(example.family, 0.0)
        family_balance = family_target / (
            1.0 + self._selected_families[example.family]
        )
        gap_balance = sum(
            self.gap_targets[gap]
            / (1.0 + self._selected_gaps[gap])
            for gap in example.gaps
        )
        difficulty_fit = math.exp(
            -0.5 * (example.difficulty - stage.difficulty_center) ** 2
        )
        exposure_penalty = 1.0 / (1.0 + self._exposure[example.id])
        hard_ids = {item.id for item in self.hard_memory.examples()}
        hard_bonus = 1.0 + stage.hard_fraction if example.id in hard_ids else 1.0
        return (
            (0.05 + family_balance)
            * (0.05 + gap_balance)
            * difficulty_fit
            * exposure_penalty
            * hard_bonus
        )

    def sample(
        self,
        *,
        count: int,
        step: int,
        progress: float,
    ) -> tuple[CorpusExample, ...]:
        if count <= 0 or step < 0:
            raise ValueError("count must be positive and step non-negative")
        rng = random.Random(self.seed + step * 1_000_003)
        stage = self.curriculum.at(progress)
        selected: list[CorpusExample] = []
        if self.arm == "curriculum_efficient":
            replay_count = min(
                count,
                round(count * stage.replay_fraction),
            )
            replay = self.hard_memory.examples(replay_count)
            selected.extend(replay)
        while len(selected) < count:
            if self.arm == "uniform":
                example = self.examples[rng.randrange(len(self.examples))]
            elif self.arm == "family_balanced":
                family = self._family(stage, rng)
                candidates = self._family_index[family]
                example = candidates[rng.randrange(len(candidates))]
            else:
                weights = [self._score(item, stage) for item in self.examples]
                example = self._weighted_choice(self.examples, weights, rng)
            selected.append(example)
            self._selected[example.id] += 1
            self._exposure[example.id] += 1
            self._selected_families[example.family] += 1
            self._selected_gaps.update(example.gaps)
        return tuple(selected)


def selection_metrics(
    selected: Sequence[CorpusExample],
    *,
    reference: Sequence[CorpusExample],
) -> SelectionMetrics:
    from floorflow.nlp_flow.data_efficiency_metrics import (
        selection_metrics as implementation,
    )

    return implementation(selected, reference=reference)


def gap_training_examples(
    examples: Sequence[CorpusExample],
    gap: str,
    *,
    replay: Sequence[CorpusExample] = (),
    replay_fraction: float = 0.1,
) -> tuple[CorpusExample, ...]:
    from floorflow.nlp_flow.data_efficiency_metrics import (
        gap_training_examples as implementation,
    )

    return implementation(
        examples,
        gap,
        replay=replay,
        replay_fraction=replay_fraction,
    )


def token_efficiency_gain(
    loss_before: float,
    loss_after: float,
    seen_tokens: int,
) -> float:
    from floorflow.nlp_flow.data_efficiency_metrics import (
        token_efficiency_gain as implementation,
    )

    return implementation(loss_before, loss_after, seen_tokens)


def forgetting(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    from floorflow.nlp_flow.data_efficiency_metrics import (
        forgetting as implementation,
    )

    return implementation(before, after)


__all__ = [
    "FAMILY_NAMES",
    "GAP_NAMES",
    "TRAINING_ARMS",
    "CorpusExample",
    "CurriculumScheduler",
    "CurriculumStage",
    "EfficientCorpusSampler",
    "ExposureDeduplicator",
    "HardExampleMemory",
    "SelectionMetrics",
    "content_hash",
    "difficulty_rank",
    "forgetting",
    "gap_training_examples",
    "hamming_distance",
    "infer_gap_tags",
    "load_balanced_jsonl_examples",
    "load_jsonl_examples",
    "make_example",
    "selection_metrics",
    "simhash64",
    "token_efficiency_gain",
]
