from __future__ import annotations

import torch
from tokenizers import ByteLevelBPETokenizer

from floorflow.nlp_flow.chat import generate_reply, render_conversation
from floorflow.nlp_flow.instruction_data import encode_chat_prompt
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def _tokenizer() -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["User: hello\nAssistant: hello there\n"],
        vocab_size=260,
        min_frequency=1,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
    )
    return tokenizer


def test_conversation_template_keeps_history() -> None:
    prompt = render_conversation(
        [("hello", "hi")],
        "how are you?",
    )

    assert prompt.startswith("User: hello\nAssistant: hi\n")
    assert prompt.endswith("User: how are you?\nAssistant:")


def test_local_generation_is_deterministic_for_greedy_decode() -> None:
    torch.manual_seed(7)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=260,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
        )
    ).eval()
    tokenizer = _tokenizer()
    first = generate_reply(
        model,
        tokenizer,
        "User: hello\nAssistant:",
        max_new_tokens=3,
        temperature=0.0,
        device="cpu",
    )
    second = generate_reply(
        model,
        tokenizer,
        "User: hello\nAssistant:",
        max_new_tokens=3,
        temperature=0.0,
        device="cpu",
    )

    assert first.text == second.text
    assert first.generated_tokens <= 3


def test_learned_conversation_stop_head_can_terminate_generation() -> None:
    torch.manual_seed(17)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=260,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
            enable_conversation_control=True,
            use_learned_stop_at_inference=True,
        )
    ).eval()
    with torch.no_grad():
        model.conversation_head.continue_head.bias.fill_(-20.0)
    result = generate_reply(
        model,
        _tokenizer(),
        "User: hello\nAssistant:",
        max_new_tokens=3,
        temperature=0.0,
        device="cpu",
    )

    assert result.stop_reason in {"learned_stop", "eos"}


def test_chat_prompt_turn_ids_preserve_user_and_assistant_roles() -> None:
    tokenizer = _tokenizer()
    prompt = render_conversation(
        [("write add", "def add(a, b): return a + b")],
        "rename it",
    )
    ids, turn_ids = encode_chat_prompt(
        prompt,
        tokenizer,
        max_tokens=128,
    )

    assert len(ids) == len(turn_ids)
    assert 0 in turn_ids
    assert 1 in turn_ids
    assert turn_ids[-1] == 1
