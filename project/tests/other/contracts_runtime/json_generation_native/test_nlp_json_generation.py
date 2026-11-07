from __future__ import annotations

import torch
from tokenizers import ByteLevelBPETokenizer

from newera.nlp_flow.components.generation.outputs.json.json_generation import (
    NativeJSONGenerator,
    gold_action_sequence,
    validate_action_sequence,
)
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def _tokenizer():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["generate structured diagnostic patch ast output"],
        vocab_size=260,
        min_frequency=1,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
    )
    return tokenizer


def test_gold_sequences_cover_every_frozen_output_kind() -> None:
    for kind in ("ast", "patch", "tool_action", "diagnostic", "engineering_result"):
        validate_action_sequence(kind, gold_action_sequence(kind))


def test_native_generator_emits_strict_schema_shaped_json() -> None:
    torch.manual_seed(4)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=260,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
            enable_typed_channels=True,
            enable_json_action_head=True,
            enable_json_schema_key_head=True,
        )
    ).eval()
    generator = NativeJSONGenerator(model, _tokenizer())
    result = generator.generate("generate diagnostic", "diagnostic")

    assert result.value["schema_version"] == "nlp-json-output-v1"
    assert result.value["kind"] == "diagnostic"
    assert set(result.value["payload"]) == {
        "diagnosis",
        "recommended_action",
        "evidence",
    }
    assert result.invalid_transition_probability == 0.0
