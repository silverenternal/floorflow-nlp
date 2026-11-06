from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from floorflow.nlp_flow.engineering import (
    ExperimentSpec,
    audit_parameter_budget,
    build_jsonl_manifest,
    build_runtime,
    count_unique_trainable_parameters,
    load_nlp_checkpoint,
    parameter_breakdown,
    restore_nlp_checkpoint,
    save_nlp_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self, tied: bool = False) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.head = self.embed if tied else nn.Linear(4, 4, bias=False)
        self.gate = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.embed(value)))


def load_experiment() -> ExperimentSpec:
    return ExperimentSpec.from_json(ROOT / "configs/nlp_engineering_p2m_control_v1.json")


def write_jsonl(path: Path, prefix: str, count: int = 2) -> None:
    path.write_text(
        "".join(
            json.dumps({"id": f"{prefix}-{index}", "text": f"{prefix} text {index}"}) + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_engineering_spec_is_canonical_and_target_free() -> None:
    spec = load_experiment()

    assert spec.validate() == ()
    assert spec.payload["data"]["target_used_at_inference"] is False
    assert spec.sha256 == ExperimentSpec.from_json(ROOT / "configs/nlp_engineering_p2m_control_v1.json").sha256


def test_engineering_spec_rejects_wrong_parameter_ratio() -> None:
    payload = load_experiment().payload
    payload["model"]["target_parameters"] += 1000

    with pytest.raises(ValueError, match="planned corpus tokens / 20"):
        ExperimentSpec.from_mapping(payload)


def test_engineering_spec_rejects_parameter_stacking() -> None:
    payload = load_experiment().payload
    payload["model"]["net_parameter_growth_allowed"] = True

    with pytest.raises(ValueError, match="net_parameter_growth_allowed must be false"):
        ExperimentSpec.from_mapping(payload)


def test_experiment_spec_accepts_ordered_data_shards() -> None:
    payload = json.loads(
        (ROOT / "configs/nlp_training_p5m_candidate_cpu_smoke_v1.json").read_text()
    )

    spec = ExperimentSpec.from_mapping(payload)

    assert spec.tier == "P5M"
    assert len(payload["data"]["train"]) == 2


def test_unique_parameter_count_counts_tied_weights_once() -> None:
    model = TinyModel(tied=True)

    assert count_unique_trainable_parameters(model) == 16
    assert parameter_breakdown(model) == {"embed": 16}


def test_parameter_budget_passes_and_fails_closed() -> None:
    assert audit_parameter_budget(2_300_000, 46_000_000).passed
    with pytest.raises(ValueError, match="parameter budget failed"):
        audit_parameter_budget(2_300_000, 92_000_000)


def test_jsonl_manifest_hashes_rows_and_rejects_duplicate_splits(tmp_path: Path) -> None:
    paths = {}
    for split, prefix in (
        ("train", "train"),
        ("validation", "validation"),
        ("test", "test"),
    ):
        paths[split] = tmp_path / f"{split}.jsonl"
        write_jsonl(paths[split], prefix)

    manifest = build_jsonl_manifest(paths)

    assert {split: row["rows"] for split, row in manifest["splits"].items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    with pytest.raises(ValueError, match="split files must be distinct"):
        build_jsonl_manifest(
            {
                "train": paths["train"],
                "validation": paths["train"],
                "test": paths["test"],
            }
        )


def test_checkpoint_round_trip_restores_model_and_rng(tmp_path: Path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(21)
    random.seed(21)
    np.random.seed(21)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    checkpoint = tmp_path / "step.pt"
    save_nlp_checkpoint(
        checkpoint,
        step=7,
        samples_seen=128,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        spec_sha256="spec",
        data_manifest_sha256="data",
        tokenizer_sha256="tokenizer",
        config={"tier": "P2M"},
        device="cpu",
    )

    payload = load_nlp_checkpoint(checkpoint, expected_spec_sha256="spec")
    random.random()
    np.random.rand()
    torch.rand(1)
    restored_model = TinyModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    state = restore_nlp_checkpoint(
        payload,
        model=restored_model,
        optimizer=restored_optimizer,
        device="cpu",
    )
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert state == {"step": 7, "samples_seen": 128}
    assert actual == pytest.approx(expected)


def test_checkpoint_rejects_wrong_experiment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step.pt"
    save_nlp_checkpoint(
        checkpoint,
        step=0,
        samples_seen=0,
        model=TinyModel(),
        optimizer=None,
        scheduler=None,
        spec_sha256="expected",
        data_manifest_sha256="data",
        tokenizer_sha256="tokenizer",
        config={},
        device="cpu",
    )

    with pytest.raises(ValueError, match="does not match"):
        load_nlp_checkpoint(checkpoint, expected_spec_sha256="other")


def test_cpu_runtime_has_shared_manifest_and_fallback() -> None:
    runtime = build_runtime(device="cpu", precision="bf16", seed=7, deterministic=True)

    assert runtime.device.type == "cpu"
    assert runtime.manifest()["deterministic"] is True
    assert runtime.accelerator.precision == "bf16"
