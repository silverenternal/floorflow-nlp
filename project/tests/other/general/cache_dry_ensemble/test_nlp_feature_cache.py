from __future__ import annotations

import torch

from newera.nlp_flow.model.components.architecture.state.feature_cache import (
    load_channel_feature_cache,
    save_channel_feature_cache,
)


def test_feature_cache_round_trip(tmp_path) -> None:
    hidden = torch.randn(2, 4, 8)
    channels = {
        "syntax": torch.randn(2, 4, 4),
        "symbol": torch.randn(2, 4, 4),
        "discourse": torch.randn(2, 4, 4),
        "execution": torch.randn(2, 4, 4),
        "engineering": torch.randn(2, 4, 4),
    }
    path = tmp_path / "features.pt"
    save_channel_feature_cache(
        path,
        hidden_states=hidden,
        channel_states=channels,
        config_hash="test",
    )

    payload = load_channel_feature_cache(path, expected_config_hash="test")

    assert torch.equal(payload["hidden_states"], hidden)
    assert set(payload["channel_states"]) == set(channels)
