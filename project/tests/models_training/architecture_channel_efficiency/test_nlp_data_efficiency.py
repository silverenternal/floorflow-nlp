from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from newera.nlp_flow.components.research.training.efficiency.data_efficiency import (
    CurriculumScheduler,
    EfficientCorpusSampler,
    ExposureDeduplicator,
    HardExampleMemory,
    forgetting,
    gap_training_examples,
    infer_gap_tags,
    load_balanced_jsonl_examples,
    make_example,
    selection_metrics,
    token_efficiency_gain,
)

ROOT = Path(__file__).resolve().parents[4]


def _corpus():
    rows = []
    for index in range(80):
        family = (
            "general_language"
            if index % 4 == 0
            else "code"
            if index % 4 == 1
            else "engineering_acceleration"
            if index % 4 == 2
            else "model_architecture"
        )
        language = (
            "python"
            if index % 3 == 0
            else "rust"
            if index % 3 == 1
            else "text"
        )
        depth = index % 4
        text = (
            f"User: task {index}\nAssistant: "
            + ("x = 1\n" * (20 * depth + 1))
        )
        rows.append(
            make_example(
                example_id=f"example-{index}",
                text=text,
                family=family,
                language=language,
            )
        )
    return tuple(rows)


def test_gap_tags_cover_language_json_structure_and_engineering() -> None:
    python_json = make_example(
        example_id="json",
        text='{"ok": true, "items": [1, 2]}',
        family="code",
        language="python",
    )
    python_code = make_example(
        example_id="python",
        text="def run():\n    return 1",
        family="code",
        language="python",
    )
    rust = make_example(
        example_id="rust",
        text="fn run() -> i32 { 1 }\nstruct Item { value: i32 }",
        family="code",
        language="rust",
    )
    engineering = make_example(
        example_id="engineering",
        text="Triton kernel launch and Linux perf trace",
        family="engineering_acceleration",
        language="text",
    )

    assert "json_validity" in python_json.gaps
    assert {"ast_structure", "python_execution"} <= set(python_code.gaps)
    assert {"ast_structure", "rust_execution"} <= set(rust.gaps)
    assert "engineering_acceleration" in engineering.gaps
    assert infer_gap_tags(
        text="plain prose",
        family="general_language",
        language="text",
    ) == ("language_retention",)


def test_exact_and_near_duplicates_are_bounded_deterministically() -> None:
    base = make_example(
        example_id="base",
        text="def add(a, b):\n    return a + b\n" * 8,
        family="code",
        language="python",
    )
    exact = make_example(
        example_id="exact",
        text=base.text,
        family="code",
        language="python",
    )
    near = make_example(
        example_id="near",
        text=base.text.replace("add", "plus", 1),
        family="code",
        language="python",
    )
    distinct = make_example(
        example_id="distinct",
        text="fn main() { println!(\"hello\"); }\n" * 8,
        family="code",
        language="rust",
    )
    dedup = ExposureDeduplicator(near_threshold=8, max_near_copies=1)

    admitted = dedup.filter((base, exact, near, distinct))

    assert [item.id for item in admitted] == ["base", "distinct"]


def test_curriculum_stages_are_contiguous_and_late_stage_is_harder() -> None:
    model = CurriculumScheduler()
    assert model.at(0.0).name == "grammar_foundation"
    assert model.at(0.5).name == "verifier_grounded"
    assert model.at(1.0).name == "composition_replay"
    assert model.at(1.0).difficulty_center > model.at(0.0).difficulty_center


def test_efficient_sampler_is_deterministic_and_covers_all_families() -> None:
    corpus = _corpus()
    first = EfficientCorpusSampler(
        corpus,
        arm="curriculum_efficient",
        seed=17,
    )
    second = EfficientCorpusSampler(
        corpus,
        arm="curriculum_efficient",
        seed=17,
    )

    left = first.sample(count=160, step=7, progress=0.8)
    right = second.sample(count=160, step=7, progress=0.8)
    early = EfficientCorpusSampler(
        corpus,
        arm="curriculum_efficient",
        seed=19,
    ).sample(count=160, step=7, progress=0.05)

    assert [item.id for item in left] == [item.id for item in right]
    metrics = selection_metrics(left, reference=corpus)
    assert all(value > 0.0 for value in metrics.family_distribution.values())
    assert metrics.mean_difficulty > selection_metrics(
        early,
        reference=corpus,
    ).mean_difficulty


def test_hard_example_memory_prioritizes_losses_and_replays() -> None:
    corpus = _corpus()[:8]
    memory = HardExampleMemory(capacity=3)
    for index, example in enumerate(corpus):
        memory.observe(example, float(index))

    assert [example.id for example in memory.examples()] == [
        "example-7",
        "example-6",
        "example-5",
    ]
    sampler = EfficientCorpusSampler(
        corpus,
        arm="curriculum_efficient",
        seed=3,
        hard_memory=memory,
    )
    selected = sampler.sample(count=20, step=0, progress=0.9)

    replay = {"example-5", "example-6", "example-7"}
    assert sum(item.id in replay for item in selected) >= 2


def test_gap_training_reuses_primary_rows_and_bounded_replay() -> None:
    corpus = _corpus()
    python = gap_training_examples(
        corpus,
        "python_execution",
        replay=corpus,
        replay_fraction=0.1,
    )
    rust = gap_training_examples(
        corpus,
        "rust_execution",
        replay=corpus,
        replay_fraction=0.1,
    )

    assert python
    assert rust
    python_primary = {
        item.id for item in corpus if "python_execution" in item.gaps
    }
    rust_primary = {
        item.id for item in corpus if "rust_execution" in item.gaps
    }
    assert python_primary <= {item.id for item in python}
    assert rust_primary <= {item.id for item in rust}
    assert {item.id for item in python} != {item.id for item in rust}


def test_efficiency_and_forgetting_metrics_are_explicit() -> None:
    assert token_efficiency_gain(4.0, 3.0, 500) == pytest.approx(0.002)
    assert forgetting(
        {"code": 2.0, "language": 3.0},
        {"code": 2.5, "language": 2.8},
    ) == pytest.approx({"code": 0.5, "language": 0.0})
    with pytest.raises(ValueError, match="seen_tokens"):
        token_efficiency_gain(2.0, 1.0, 0)


def test_duplicate_exposure_metric_uses_content_not_row_ids() -> None:
    base = _corpus()[0]
    copy = make_example(
        example_id="copy",
        text=base.text,
        family=base.family,
        language=base.language,
    )
    metrics = selection_metrics((base, copy), reference=(base, copy))

    assert Counter(item.content_hash for item in (base, copy))[base.content_hash] == 2
    assert metrics.duplicate_exposures == 1


def test_config_separates_measured_and_deferred_capability_metrics() -> None:
    config = json.loads(
        (
            ROOT / "project/configs/models_training/learning/heads/nlp_data_efficient_learning.json"
        ).read_text(encoding="utf-8")
    )

    assert {arm["sampler"] for arm in config["arms"]} == {
        "uniform",
        "family_balanced",
        "curriculum_efficient",
    }
    assert config["gap_training"]["test_split_forbidden"] is True
    assert "heldout_byte_bigram_nll" in config["metrics"]["primary"]
    assert "executable_generation_rate" in (
        config["metrics"]["deferred_to_gpu_extension"]
    )


def test_balanced_loader_scans_past_a_long_family_prefix(tmp_path) -> None:
    path = tmp_path / "ordered.jsonl"
    rows = [
        {"text": f"language {index}", "family": "general_language"}
        for index in range(20)
    ]
    rows.extend(
        [
            {"text": f"def f{index}(): return {index}", "family": "code"}
            for index in range(3)
        ]
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    examples = load_balanced_jsonl_examples(
        (path,),
        max_per_family=2,
        max_scan_rows=100,
    )

    assert Counter(item.family for item in examples) == {
        "general_language": 2,
        "code": 2,
    }
