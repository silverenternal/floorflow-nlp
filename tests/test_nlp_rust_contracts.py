from __future__ import annotations

import subprocess
from pathlib import Path

from floorflow.nlp_flow._generated_protocol import (
    ARCHITECTURE_AXIS_NAMES,
    BENCHMARK_REGRESSION_TARGETS,
    BENCHMARK_SCALAR_TARGETS,
    BENCHMARK_TOKEN_TARGETS,
    JSON_ACTION_NAMES,
    TRAINING_SCOPE_GROUP_NAMES,
)
from floorflow.nlp_flow.model.training_scopes import SCOPE_PREFIXES
from floorflow.nlp_flow.output_contract import OUTPUT_CONTRACTS

ROOT = Path(__file__).resolve().parents[1]


def test_generated_python_contract_is_current() -> None:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            "rust/Cargo.toml",
            "-p",
            "nlp_flow_codegen",
            "--",
            "--format",
            "python",
            "--check",
            "src/floorflow/nlp_flow/_generated_protocol.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_generated_contracts_cover_model_iteration_surfaces() -> None:
    assert len(BENCHMARK_SCALAR_TARGETS) == 27
    assert len(BENCHMARK_TOKEN_TARGETS) == 8
    assert len(BENCHMARK_REGRESSION_TARGETS) == 3
    assert set(SCOPE_PREFIXES) == set(TRAINING_SCOPE_GROUP_NAMES)
    assert "unified_generated_expert_mode" in ARCHITECTURE_AXIS_NAMES
    assert JSON_ACTION_NAMES[0] == "start_object"
    assert len(set(JSON_ACTION_NAMES)) == len(JSON_ACTION_NAMES)


def test_generated_output_contracts_distinguish_json_and_patch() -> None:
    assert OUTPUT_CONTRACTS["json"].required_dimensions == (
        "structural",
        "schema",
    )
    assert OUTPUT_CONTRACTS["patch"].required_dimensions == (
        "structural",
        "schema",
        "executable",
    )
