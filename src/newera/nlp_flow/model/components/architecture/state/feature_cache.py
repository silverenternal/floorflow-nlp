"""Local frozen feature cache for channel-only training probes."""

from __future__ import annotations

from pathlib import Path

import torch

from newera.nlp_flow.engineering.components.free_schedule_spec.storage import save_torch_atomic


def save_channel_feature_cache(
    path: str | Path,
    *,
    hidden_states: torch.Tensor,
    channel_states: dict[str, torch.Tensor],
    config_hash: str,
) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [batch, sequence, channels]")
    for name, value in channel_states.items():
        if value.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(f"channel state {name} does not align with hidden states")
    save_torch_atomic(
        path,
        {
            "version": "nlp-channel-feature-cache-v1",
            "config_hash": config_hash,
            "hidden_states": hidden_states.detach().cpu(),
            "channel_states": {
                name: value.detach().cpu()
                for name, value in channel_states.items()
            },
        },
    )


def load_channel_feature_cache(
    path: str | Path,
    *,
    expected_config_hash: str | None = None,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != "nlp-channel-feature-cache-v1":
        raise ValueError("unsupported channel feature cache")
    if expected_config_hash is not None and payload["config_hash"] != expected_config_hash:
        raise ValueError("channel feature cache config mismatch")
    return payload


__all__ = ["load_channel_feature_cache", "save_channel_feature_cache"]
