from __future__ import annotations

from pathlib import Path

from floorflow.nlp_flow.engineering import ExperimentSpec
from floorflow.nlp_flow.experiment_config import candidate_config

ROOT = Path(__file__).resolve().parents[1]


def test_joint_emergence_spec_really_enables_the_training_objective() -> None:
    spec = ExperimentSpec.from_json(
        ROOT / "configs/nlp_open_emergence_joint_v1.json"
    )
    config = candidate_config(spec)
    model = spec.payload["model"]

    assert config.enable_emergence_objective is True
    assert config.enable_conversation_control is True
    assert config.enable_benchmark_proxy is True
    assert config.enable_unified_competence is True
    assert config.enable_expert_factory is False
    assert model["emergence_loss_weight"] > 0.0
    assert model["emergence_reward_weight"] > 0.0
    assert model["conversation_control_loss_weight"] > 0.0
    assert model["benchmark_proxy_loss_weight"] > 0.0
