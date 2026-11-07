from __future__ import annotations

from tokenizers import ByteLevelBPETokenizer


def test_code_aware_byte_level_tokenizer_round_trips() -> None:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        [
            "def f(x):\n    return x + 1\n",
            "fn f<'a>(x: &'a str) -> &'a str { x }\n",
            "template <typename T> T add(T a, T b) { return a + b; }\n",
        ],
        vocab_size=256,
        min_frequency=1,
        special_tokens=["<unk>"],
    )

    for text in (
        "def f(x):\n    return x\n",
        "fn main() { println!(\"hi\"); }\n",
        "{\"k\": \"välue\"}",
    ):
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
