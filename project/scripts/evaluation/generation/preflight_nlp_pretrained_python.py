#!/usr/bin/env python3
"""Freeze and audit the CPU-only inputs for the CodeT5 Python diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from newera.nlp_flow.components.generation.policies.synthetic.synthetic_tree_policy import build_synthetic_tasks

ROOT = Path(__file__).resolve().parents[4]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _rows_sha(rows: list[Any], field: str) -> str:
    data = "\n".join(
        json.dumps(
            {"task_id": row.task_id, field: getattr(row, field)},
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    ).encode("utf-8")
    return _sha_bytes(data)


def _remote_get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "newera-preflight/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _probe_tokenizer(
    model: dict[str, Any], prompts: list[str], targets: list[str], version: str
) -> dict[str, Any]:
    probe = (
        "import json,sys\n"
        "import transformers\n"
        "import tokenizers\n"
        "from transformers import AutoTokenizer\n"
        "payload=json.load(sys.stdin)\n"
        "t=AutoTokenizer.from_pretrained(payload['repo'], revision=payload['revision'], use_fast=True)\n"
        "src=t(payload['prompts'], add_special_tokens=True)['input_ids']\n"
        "tgt=t(payload['targets'], add_special_tokens=True)['input_ids']\n"
        "print(json.dumps({'transformers_version':transformers.__version__,"
        "'tokenizers_version':tokenizers.__version__,"
        "'class':type(t).__name__,'vocab_size':t.vocab_size,'length':len(t),"
        "'model_max_length':t.model_max_length,'max_prompt_tokens':max(map(len,src)),"
        "'max_target_tokens':max(map(len,tgt)),'prompt_roundtrips':sum("
        "t.decode(ids,skip_special_tokens=True)==text for ids,text in zip(src,payload['prompts'])),"
        "'target_roundtrips':sum(t.decode(ids,skip_special_tokens=True)==text "
        "for ids,text in zip(tgt,payload['targets']))},sort_keys=True))\n"
    )
    payload = json.dumps(
        {
            "repo": model["repository"],
            "revision": model["revision"],
            "prompts": prompts,
            "targets": targets,
        }
    )
    result = subprocess.run(
        [
            "uv", "run", "--isolated", "--with", f"transformers=={version}",
            "--python", "3.11", "python", "-c", probe,
        ],
        input=payload,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pinned tokenizer probe failed: {result.stderr[-1000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _model_parameter_count(model: dict[str, Any]) -> int:
    d_model = int(model["d_model"])
    d_ff = int(model["d_ff"])
    layers = int(model["num_layers"])
    decoder_layers = int(model["num_decoder_layers"])
    heads = int(model["num_heads"])
    d_kv = int(model["d_kv"])
    vocab = int(model["vocab_size"])
    # T5 shares token embeddings across encoder, decoder, and output projection.
    embedding = vocab * d_model
    encoder_layer = 4 * d_model * (heads * d_kv) + 2 * d_model * d_ff + 2 * d_model
    decoder_layer = (
        8 * d_model * (heads * d_kv)
        + 2 * d_model * d_ff
        + 3 * d_model
    )
    relative_bias = 2 * int(model["relative_attention_num_buckets"]) * heads
    final_norms = 2 * d_model
    return embedding + layers * encoder_layer + decoder_layers * decoder_layer + relative_bias + final_norms


def _audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    dataset = config["dataset"]
    train = build_synthetic_tasks(
        start=int(dataset["train"]["start"]),
        count=int(dataset["train"]["count"]),
    )
    validation = build_synthetic_tasks(
        start=int(dataset["validation"]["start"]),
        count=int(dataset["validation"]["count"]),
    )
    dataset_checks: dict[str, Any] = {}
    for split, rows in (("train", train), ("validation", validation)):
        expected = dataset[split]
        actual = {
            "count": len(rows),
            "task_ids_sha256": _rows_sha(rows, "task_id"),
            "prompt_sha256": _rows_sha(rows, "prompt"),
            "reference_source_sha256": _rows_sha(rows, "source"),
            "unique_names": len({row.function_name for row in rows}),
            "constant_values": sorted({row.constant for row in rows}),
        }
        dataset_checks[split] = actual
        for key in ("task_ids_sha256", "prompt_sha256", "reference_source_sha256"):
            if actual[key] != expected[key]:
                errors.append(f"{split} {key} differs from frozen config")
        if actual["count"] != int(expected["count"]):
            errors.append(f"{split} count differs from frozen config")
    train_names = {row.function_name for row in train}
    validation_names = {row.function_name for row in validation}
    train_constants = {row.constant for row in train}
    validation_constants = {row.constant for row in validation}
    dataset_checks["name_overlap"] = sorted(train_names & validation_names)
    dataset_checks["constant_overlap"] = sorted(train_constants & validation_constants)
    if dataset_checks["name_overlap"]:
        errors.append("train/validation function names overlap")
    if dataset_checks["constant_overlap"] != dataset["split_facts"]["constant_values_seen_in_both_splits"]:
        errors.append("train/validation constant overlap differs from frozen config")

    baseline = config["baseline"]
    baseline_path = ROOT / baseline["report"]
    if _sha_file(baseline_path) != baseline["report_sha256"]:
        errors.append("historical baseline report hash changed")
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_ids = [row["task_id"] for row in baseline_report["case_rows"]]
    if baseline_ids != [row.task_id for row in validation]:
        errors.append("historical baseline cases do not match frozen validation IDs")
    if len(baseline_ids) != 24:
        errors.append("historical baseline case count is not 24")

    model = config["pretrained_model"]
    model_config = json.loads(
        _remote_get(
            f"https://huggingface.co/{model['repository']}/resolve/"
            f"{model['revision']}/config.json"
        )
    )
    model_config_bytes = _remote_get(
        f"https://huggingface.co/{model['repository']}/resolve/"
        f"{model['revision']}/config.json"
    )
    model_checks: dict[str, Any] = {
        "config_sha256": _sha_bytes(model_config_bytes),
        "parameter_count_derived": _model_parameter_count(model_config),
        "license": None,
        "revision": model["revision"],
    }
    if model_checks["config_sha256"] != model["model_config"]["sha256"]:
        errors.append("pinned model config hash changed")
    if model_checks["parameter_count_derived"] != model["parameter_count"]["exact"]:
        errors.append("derived parameter count differs from frozen count")
    try:
        from transformers import T5Config, T5ForConditionalGeneration

        t5_config = T5Config.from_dict(model_config)
        instantiated = T5ForConditionalGeneration(t5_config)
        model_checks["parameter_count_instantiated"] = sum(
            parameter.numel() for parameter in instantiated.parameters()
        )
        model_checks["output_embedding_tied"] = (
            instantiated.shared.weight.data_ptr()
            == instantiated.lm_head.weight.data_ptr()
        )
        if model_checks["parameter_count_instantiated"] != model["parameter_count"]["exact"]:
            errors.append("instantiated model parameter count differs from frozen count")
        del instantiated
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        model_checks["parameter_count_instantiation_error"] = str(error)
        errors.append("could not instantiate model architecture to verify parameter count")
    api = json.loads(
        _remote_get(
            f"https://huggingface.co/api/models/{model['repository']}/revision/"
            f"{model['revision']}"
        )
    )
    model_checks["license"] = api.get("cardData", {}).get("license")
    if api.get("sha") != model["revision"]:
        errors.append("model hub returned a different revision")
    if model_checks["license"] != model["license"]:
        errors.append("model license differs from frozen metadata")
    tokenizer_files = {
        "tokenizer_config.json": model["tokenizer"]["tokenizer_config_sha256"],
        "special_tokens_map.json": model["tokenizer"]["special_tokens_map_sha256"],
        "added_tokens.json": model["tokenizer"]["added_tokens_sha256"],
        "vocab.json": model["tokenizer"]["vocab_sha256"],
        "merges.txt": model["tokenizer"]["merges_sha256"],
    }
    tokenizer_hashes = {}
    for filename, expected_hash in tokenizer_files.items():
        digest = _sha_bytes(
            _remote_get(
                f"https://huggingface.co/{model['repository']}/resolve/"
                f"{model['revision']}/{filename}"
            )
        )
        tokenizer_hashes[filename] = digest
        if digest != expected_hash:
            errors.append(f"pinned tokenizer file changed: {filename}")
    model_checks["tokenizer_file_hashes"] = tokenizer_hashes
    tokenizer_probe = _probe_tokenizer(
        model,
        [row.prompt for row in train + validation],
        [row.source for row in train + validation],
        str(config["training"]["runtime"]["transformers"]),
    )
    model_checks["tokenizer_runtime_probe"] = tokenizer_probe
    if tokenizer_probe["class"] != "RobertaTokenizerFast":
        errors.append("pinned runtime loaded an unexpected tokenizer class")
    if tokenizer_probe["transformers_version"] != str(config["training"]["runtime"]["transformers"]):
        errors.append("tokenizer probe ran with an unexpected Transformers version")
    if tokenizer_probe["tokenizers_version"] != str(config["training"]["runtime"]["tokenizers"]):
        errors.append("tokenizer probe ran with an unexpected tokenizers version")
    if tokenizer_probe["vocab_size"] != int(model["model_config"]["vocab_size"]):
        errors.append("pinned tokenizer vocabulary size differs from model config")
    if tokenizer_probe["max_prompt_tokens"] > int(config["training"]["max_source_tokens"]):
        errors.append("configured source token budget truncates a frozen prompt")
    if tokenizer_probe["max_target_tokens"] > int(config["training"]["max_target_tokens"]):
        errors.append("configured target token budget truncates a reference target")
    if tokenizer_probe["prompt_roundtrips"] != len(train) + len(validation):
        errors.append("prompt tokenization does not round-trip every task")
    if tokenizer_probe["target_roundtrips"] != len(train) + len(validation):
        errors.append("target tokenization does not round-trip every reference")
    weight_request = urllib.request.Request(
        f"https://huggingface.co/{model['repository']}/resolve/"
        f"{model['revision']}/{model['checkpoint_file']}",
        method="HEAD",
    )
    with urllib.request.urlopen(weight_request, timeout=30) as response:
        model_checks["checkpoint_etag"] = response.headers.get("ETag", "").strip('"')
        model_checks["checkpoint_size_bytes"] = int(response.headers.get("Content-Length", "0"))
    if model_checks["checkpoint_etag"] != model["checkpoint_etag"]:
        errors.append("pinned checkpoint ETag changed")
    if model_checks["checkpoint_size_bytes"] != model["checkpoint_size_bytes"]:
        errors.append("pinned checkpoint size changed")

    source_paths = {
        "task_builder": dataset["builder_path"],
        "baseline_runner": baseline["runner"],
        "baseline_runtime": baseline["runtime"],
        "parser": config["executor"]["parser_path"],
        "pilot_runner": "project/scripts/experiments/validation/run_nlp_pretrained_python_baseline.py",
    }
    source_hashes = {name: _sha_file(ROOT / path) for name, path in source_paths.items()}
    expected_hashes = {
        "task_builder": dataset["builder_sha256"],
        "baseline_runner": baseline["runner_sha256"],
        "baseline_runtime": baseline["runtime_sha256"],
        "parser": config["executor"]["parser_sha256"],
    }
    for name, digest in source_hashes.items():
        if name in expected_hashes and digest != expected_hashes[name]:
            errors.append(f"implementation source changed: {name}")
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("transformers", "tokenizers", "sentencepiece")
    }
    dependency_versions = {}
    for name, distribution in (
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("torch", "torch"),
    ):
        try:
            dependency_versions[name] = version(distribution)
        except PackageNotFoundError:
            dependency_versions[name] = None
    report = {
        "version": "nlp-pretrained-python-preflight-v1",
        "status": "ready" if not errors else "fail",
        "configuration": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha_file(config_path),
        },
        "preflight_runner_sha256": _sha_file(Path(__file__).resolve()),
        "dataset": {
            **dataset_checks,
            "allowed_model_input": dataset["allowed_model_input"],
            "forbidden_inputs": dataset["forbidden_model_inputs"],
            "test_content_read": False,
        },
        "baseline": {
            "report_sha256": _sha_file(baseline_path),
            "metrics": baseline_report["metrics"],
            "paired_validation_ids": len(baseline_ids),
            "historical_split_label_warning": "validation names are unseen; constants are not unseen",
        },
        "pretrained_model": model_checks,
        "implementation_hashes": source_hashes,
        "runtime_dependencies_installed": dependencies,
        "current_environment_versions": dependency_versions,
        "gpu_used": False,
        "errors": errors,
        "evidence_boundary": config["evidence_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/other/general/capability_open_matched/nlp_pretrained_python.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_pretrained_python_preflight_v1.json",
    )
    args = parser.parse_args()
    report = _audit(args.config.resolve(), args.output.resolve())
    print(json.dumps({key: report[key] for key in (
        "status", "configuration", "dataset", "baseline", "pretrained_model",
        "runtime_dependencies_installed", "gpu_used", "errors",
    )}, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
