"""Learned turn, stopping, output-mode, language, and validity controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

TURN_TYPES = ("user", "assistant", "tool", "code", "system")
REPLY_MODES = ("text", "code", "json", "ast")
CODE_LANGUAGES = ("text", "python", "rust", "cpp")
VALIDITY_TYPES = ("code", "json", "ast")


@dataclass(frozen=True)
class ConversationControlConfig:
    hidden_size: int
    turn_types: int = len(TURN_TYPES)

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.turn_types <= 0:
            raise ValueError("conversation dimensions must be positive")


class TurnSegmentEmbedding(nn.Module):
    """Zero-initialized segment signal for user/assistant/tool/code turns."""

    def __init__(self, hidden_size: int, turn_types: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(turn_types, hidden_size))

    def forward(self, turn_ids: torch.Tensor) -> torch.Tensor:
        if turn_ids.ndim != 2:
            raise ValueError("turn_ids must be [batch, sequence]")
        if turn_ids.min() < 0 or turn_ids.max() >= self.weight.shape[0]:
            raise ValueError("turn id is outside the configured turn types")
        return self.weight[turn_ids]


class ConversationControlHead(nn.Module):
    """Predict continuation, turn boundary, output mode, language, and validity."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.continue_head = nn.Linear(hidden_size, 1)
        self.turn_boundary_head = nn.Linear(hidden_size, 3)
        self.reply_mode_head = nn.Linear(hidden_size, len(REPLY_MODES))
        self.language_head = nn.Linear(hidden_size, len(CODE_LANGUAGES))
        self.validity_head = nn.Linear(hidden_size, len(VALIDITY_TYPES))

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch, sequence, hidden]")
        state = self.norm(hidden)
        return {
            "continue_logits": self.continue_head(state).squeeze(-1),
            "turn_boundary_logits": self.turn_boundary_head(state),
            "reply_mode_logits": self.reply_mode_head(state),
            "language_logits": self.language_head(state),
            "validity_logits": self.validity_head(state),
        }

    def losses(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        required = {
            "continue_logits",
            "turn_boundary_logits",
            "reply_mode_logits",
            "language_logits",
            "validity_logits",
        }
        if set(predictions) != required or not set(targets) <= required:
            raise ValueError("conversation predictions or targets are inconsistent")
        masks = {} if masks is None else masks
        unknown_masks = set(masks) - required
        if unknown_masks:
            raise ValueError(f"unknown conversation masks: {sorted(unknown_masks)}")
        losses = {}
        for name in required:
            if name not in targets:
                continue
            target = targets[name]
            prediction = predictions[name]
            active = masks.get(name)
            if active is None:
                active = (
                    target.ge(0)
                    if target.dtype in {torch.long, torch.int64}
                    else torch.ones_like(target, dtype=torch.bool)
                )
            else:
                active = active.bool()
            classification = name in {
                "turn_boundary_logits",
                "reply_mode_logits",
                "language_logits",
            }
            expected_shape = (
                prediction.shape[:-1] if classification else prediction.shape
            )
            if target.shape != expected_shape or active.shape != expected_shape:
                raise ValueError(f"{name} target or mask shape mismatch")
            if not active.any():
                raise ValueError(f"{name} has no active conversation targets")
            key = name.removesuffix("_logits")
            if name in {
                "continue_logits",
                "turn_boundary_logits",
                "reply_mode_logits",
                "language_logits",
            }:
                losses[key] = (
                    F.binary_cross_entropy_with_logits(
                        prediction[active],
                        target[active].float(),
                    )
                    if key == "continue"
                    else F.cross_entropy(
                        prediction[active],
                        target[active].long(),
                    )
                )
            else:
                losses[key] = F.binary_cross_entropy_with_logits(
                    prediction[active],
                    target[active].float(),
                )
        if not losses:
            raise ValueError("no conversation losses were selected")
        return losses

    @staticmethod
    def should_stop(continue_logits: torch.Tensor) -> torch.Tensor:
        if continue_logits.ndim == 2:
            continue_logits = continue_logits[:, -1]
        if continue_logits.ndim != 1:
            raise ValueError("continue logits must be [batch] or [batch, sequence]")
        return continue_logits.le(0.0)


__all__ = [
    "CODE_LANGUAGES",
    "REPLY_MODES",
    "TURN_TYPES",
    "VALIDITY_TYPES",
    "ConversationControlConfig",
    "ConversationControlHead",
    "TurnSegmentEmbedding",
]
