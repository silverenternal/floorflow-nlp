import pytest

from newera.nlp_flow.components.platform.contracts.protocol.prompt_provenance import (
    Evidence,
    PromptField,
    compile_prompt,
    content_sha256,
    verify_prompt_bundle,
)


def _source(name: str, origin: str, text: str) -> Evidence:
    return Evidence(name, origin, text, content_sha256(text))


def test_user_prompt_accepts_only_registered_verbatim_user_evidence() -> None:
    sources = {"request": _source("request", "user_available", "Implement sum(a, b)")}
    field = PromptField("task", "Implement sum(a, b)", "request")
    result = compile_prompt("user_task_generation", [field], sources)
    assert result["fields"][0]["origin"] == "user_available"
    assert len(result["sha256"]) == 64
    verify_prompt_bundle(result, sources)
    tampered = {**result, "fields": [{**result["fields"][0], "origin": "target_derived"}]}
    with pytest.raises(ValueError, match="metadata or fingerprint"):
        verify_prompt_bundle(tampered, sources)
    with pytest.raises(ValueError, match="not present"):
        compile_prompt("user_task_generation", [PromptField("task", "Implement subtract", "request")], sources)


@pytest.mark.parametrize("name", ["symbol_names", "construct_summary", "target_span", "tests"])
def test_target_hints_and_hidden_tests_cannot_be_relabelled_by_field_name(name: str) -> None:
    origin = "evaluation_only" if name == "tests" else "target_derived"
    sources = {"hidden": _source("hidden", origin, "secret implementation")}
    with pytest.raises(ValueError):
        compile_prompt("user_task_generation", [PromptField(name, "secret implementation", "hidden")], sources)


def test_reconstruction_origin_is_visible_but_evaluation_evidence_is_forbidden() -> None:
    sources = {"target": _source("target", "target_derived", "function f")}
    result = compile_prompt("reconstruction_recovery", [PromptField("symbol", "f", "target")], sources)
    assert result["fields"][0]["origin"] == "target_derived"
    sources["hidden"] = _source("hidden", "evaluation_only", "assert f() == 1")
    with pytest.raises(ValueError, match="evaluation-only"):
        compile_prompt("reconstruction_recovery", [PromptField("tests", "assert f() == 1", "hidden")], sources)


def test_stale_source_hash_duplicate_field_and_forged_evidence_id_fail() -> None:
    source = _source("task", "user_available", "original")
    with pytest.raises(ValueError, match="fingerprint"):
        compile_prompt(
            "user_task_generation",
            [PromptField("task", "tampered", "task")],
            {
                "task": Evidence("task", "user_available", "tampered", source.sha256),
            },
        )
    with pytest.raises(ValueError, match="unique"):
        compile_prompt("user_task_generation", [PromptField("task", "original", "task")] * 2, {"task": source})
    with pytest.raises(ValueError, match="unregistered"):
        compile_prompt("user_task_generation", [PromptField("task", "original", "alias")], {"alias": source})
