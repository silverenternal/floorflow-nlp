#!/usr/bin/env python3
"""Train a code-aware BPE tokenizer on train only and produce a token ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parents[5]


def read_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def text_iterator(path: Path) -> Iterator[str]:
    for row in read_rows(path):
        yield str(row["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT / ".data/nlp/mixed_v1/corpus_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/tokenizer_v1",
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_iterator(args.corpus_root / "train.jsonl"),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
    )
    tokenizer_path = args.output_root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=512)
    ledger = {}
    total = 0
    for split in ("train", "validation", "test"):
        by_family: Counter[str] = Counter()
        by_language: Counter[str] = Counter()
        rows = 0
        for row in read_rows(args.corpus_root / f"{split}.jsonl"):
            count = len(tokenizer.encode(str(row["text"])).ids)
            by_family[row["family"]] += count
            by_language[row["language"]] += count
            rows += 1
        split_total = sum(by_family.values())
        total += split_total
        ledger[split] = {
            "rows": rows,
            "tokens": split_total,
            "tokens_by_family": dict(sorted(by_family.items())),
            "tokens_by_language": dict(sorted(by_language.items())),
        }
    roundtrip = {
        "python": "def f(x: int) -> int:\n    return x + 1\n",
        "rust": "fn f<'a>(x: &'a str) -> &'a str { x }\n",
        "cpp": "template <typename T>\nT add(T a, T b) { return a + b; }\n",
        "json": "{\"key\": \"välue\\n\", \"n\": 1.5}",
    }
    roundtrip_ok = {}
    for language, text in roundtrip.items():
        encoded = tokenizer.encode(text)
        roundtrip_ok[language] = tokenizer.decode(encoded.ids) == text
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    report = {
        "version": "nlp-tokenizer-v1",
        "tokenizer": str(tokenizer_path),
        "sha256": digest,
        "vocab_size": tokenizer.get_vocab_size(),
        "fit_split": "train",
        "ledger": ledger,
        "total_tokens": total,
        "roundtrip": roundtrip_ok,
        "roundtrip_pass": all(roundtrip_ok.values()),
        "p5m_token_target": 92_000_000,
        "p5m_target_met": total >= 92_000_000,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
