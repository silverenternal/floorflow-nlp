from __future__ import annotations

from floorflow.nlp_flow.verifier_result_contract import (
    VerificationIdentity,
    preflight_runner_manifest,
)


def _identity() -> VerificationIdentity:
    digest = "a" * 64
    return VerificationIdentity("case-1", digest, "b" * 64, "c" * 64, "d" * 64)


def test_runner_preflight_orders_correctness_before_performance_and_never_launches() -> None:
    identity = _identity()
    base = {
        "schema_version": "nlp-engineering-runner-v1",
        "cases": [{
            **identity.__dict__,
            "environment_kind": "cpu",
            "stages": ["patch", "compile", "tests", "numerical", "performance"],
            "resume_identity": identity.__dict__,
        }],
    }
    passed = preflight_runner_manifest(base)
    assert passed["status"] == "preflight_passed_no_execution"
    assert not passed["launch_performed"]

    bad_order = {**base, "cases": [{**base["cases"][0], "stages": ["performance", "numerical"]}]}
    rejected = preflight_runner_manifest(bad_order)
    assert rejected["status"] == "invalid"
    assert any("unique and ordered" in item for item in rejected["errors"])

    gpu = {**base, "cases": [{**base["cases"][0], "environment_kind": "gpu"}]}
    blocked = preflight_runner_manifest(gpu)
    assert blocked["status"] == "blocked_gpu_gate"
    assert not blocked["launch_performed"]


def test_runner_preflight_rejects_stale_resume_identity() -> None:
    identity = _identity()
    row = {
        **identity.__dict__,
        "environment_kind": "cpu",
        "stages": ["tests", "numerical"],
        "resume_identity": {**identity.__dict__, "candidate_sha256": "e" * 64},
    }
    report = preflight_runner_manifest({"schema_version": "nlp-engineering-runner-v1", "cases": [row]})
    assert report["status"] == "invalid"
    assert "resume identity is stale" in report["errors"][0]
