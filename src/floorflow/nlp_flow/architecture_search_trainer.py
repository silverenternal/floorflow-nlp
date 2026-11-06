"""Matched short-budget training evaluator for architecture search probes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.nlp_flow.architecture_search_types import (
    ArchitectureCandidate,
    CandidateEvaluation,
)
from floorflow.nlp_flow.engineering import ScheduleFreeAdamW
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.training_data import PackedTokenDataset
from floorflow.nlp_flow.training_losses import batch_loss


@dataclass
class _TrainingState:
    model: CompactNLPDecoder
    optimizer: torch.optim.Optimizer
    steps: int = 0
    training_tokens: int = 0


class TorchValidationArchitectureEvaluator:
    """Train candidates with the same seed, batches, and token budgets."""

    def __init__(
        self,
        *,
        train_paths: Sequence[str | Path],
        validation_paths: Sequence[str | Path],
        tokenizer_path: str | Path,
        device: str | torch.device,
        batch_size: int,
        sequence_length: int,
        train_batches: int,
        validation_batches: int,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 0.01,
        optimizer_name: str = "schedule_free_adamw",
        seed: int = 19,
        precision: str = "bf16",
        text_mode: str = "chat",
        families: frozenset[str] | None = None,
        emergence_weight: float = 0.0,
        dynamic_depth_weight: float = 0.0,
        dynamic_target_depth: float = 2.0,
        quality_fn: Callable[[dict[str, float]], float] | None = None,
    ) -> None:
        for value, name in (
            (batch_size, "batch_size"),
            (sequence_length, "sequence_length"),
            (train_batches, "train_batches"),
            (validation_batches, "validation_batches"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not train_paths or not validation_paths:
            raise ValueError("train and validation paths are required")
        if optimizer_name not in {"schedule_free_adamw", "adamw"}:
            raise ValueError(f"unsupported search optimizer: {optimizer_name}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.seed = int(seed)
        self.precision = precision
        self.emergence_weight = float(emergence_weight)
        self.dynamic_depth_weight = float(dynamic_depth_weight)
        self.dynamic_target_depth = float(dynamic_target_depth)
        self.quality_fn = quality_fn
        self.train_batches = self._materialize(
            train_paths,
            tokenizer_path,
            count=train_batches,
            text_mode=text_mode,
            families=families,
        )
        self.validation_batches = self._materialize(
            validation_paths,
            tokenizer_path,
            count=validation_batches,
            text_mode=text_mode,
            families=families,
        )
        self._states: dict[str, _TrainingState] = {}

    def _materialize(
        self,
        paths: Sequence[str | Path],
        tokenizer_path: str | Path,
        *,
        count: int,
        text_mode: str,
        families: frozenset[str] | None,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        dataset = PackedTokenDataset(
            tuple(Path(path) for path in paths),
            Path(tokenizer_path),
            context_length=self.sequence_length,
            max_sequences=count * self.batch_size,
            text_mode=text_mode,
            families=families,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0,
        )
        batches = tuple(loader)
        if len(batches) < count:
            raise ValueError(
                f"requested {count} batches but the data yielded {len(batches)}"
            )
        return batches[:count]

    def _autocast(self):
        if self.device.type == "cuda" and self.precision == "bf16":
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        return nullcontext()

    def _state(self, candidate: ArchitectureCandidate) -> _TrainingState:
        existing = self._states.get(candidate.candidate_id)
        if existing is not None:
            return existing
        torch.manual_seed(self.seed)
        config = NLPModelConfig(**candidate.config)
        model = CompactNLPDecoder(config).to(self.device)
        optimizer = (
            ScheduleFreeAdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            if self.optimizer_name == "schedule_free_adamw"
            else torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        )
        state = _TrainingState(model=model, optimizer=optimizer)
        self._states[candidate.candidate_id] = state
        return state

    def evaluate(
        self,
        candidate: ArchitectureCandidate,
        budget_tokens: int,
    ) -> CandidateEvaluation:
        tokens_per_step = self.batch_size * self.sequence_length
        target_steps = budget_tokens // tokens_per_step
        if target_steps <= 0:
            raise ValueError("token budget is smaller than one training batch")
        state = self._state(candidate)
        if target_steps < state.steps:
            raise ValueError("successive-halving budgets must be increasing")

        state.model.train()
        if isinstance(state.optimizer, ScheduleFreeAdamW):
            state.optimizer.train()
        started = time.perf_counter()
        starting_tokens = state.training_tokens
        for step in range(state.steps, target_steps):
            batch = self.train_batches[step % len(self.train_batches)]
            input_ids = batch["input_ids"].to(self.device)
            state.optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = batch_loss(
                state.model,
                batch,
                device=self.device,
                autocast=self._autocast(),
                emergence_weight=self.emergence_weight,
                dynamic_depth_weight=self.dynamic_depth_weight,
                dynamic_target_depth=self.dynamic_target_depth,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite architecture-search loss")
            loss.backward()
            if not all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in state.model.parameters()
            ):
                raise RuntimeError("non-finite architecture-search gradient")
            state.optimizer.step()
            state.steps += 1
            state.training_tokens += int(input_ids.numel())
        elapsed = time.perf_counter() - started

        if isinstance(state.optimizer, ScheduleFreeAdamW):
            state.optimizer.eval()
        validation_loss = self._validation_loss(state.model)
        latency_ms = self._measure_latency(state.model)
        metrics = {
            "validation_loss": validation_loss,
            "training_tokens_per_second": (
                (state.training_tokens - starting_tokens)
                / max(elapsed, 1.0e-6)
            ),
            "parameters": float(candidate.parameters),
        }
        quality = (
            self.quality_fn(metrics)
            if self.quality_fn is not None
            else -validation_loss
        )
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            budget_tokens=budget_tokens,
            training_tokens=state.training_tokens,
            quality_score=quality,
            metrics=metrics,
            evaluation_split="validation",
            measured_latency_ms=latency_ms,
        )

    def _validation_loss(self, model: CompactNLPDecoder) -> float:
        model.eval()
        losses = []
        with torch.no_grad(), self._autocast():
            for batch in self.validation_batches:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = model(input_ids).token_logits
                losses.append(
                    float(
                        F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            labels[:, 1:].reshape(-1),
                            ignore_index=-100,
                        )
                    )
                )
        if not losses or not all(math.isfinite(value) for value in losses):
            raise RuntimeError("architecture validation produced no finite loss")
        return sum(losses) / len(losses)

    def _measure_latency(self, model: CompactNLPDecoder) -> float:
        batch = self.validation_batches[0]["input_ids"].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad(), self._autocast():
            model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000.0

    def discard(self, candidate_id: str) -> None:
        state = self._states.pop(candidate_id, None)
        if state is None:
            return
        state.model.to("cpu")
        del state
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = ["TorchValidationArchitectureEvaluator"]
