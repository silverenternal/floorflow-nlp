#!/usr/bin/env python3
"""Interactive multi-turn chat for a compact NLP checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from floorflow.nlp_flow.chat import (
    generate_reply,
    model_config_from_experiment,
    render_conversation,
)
from floorflow.nlp_flow.engineering import load_nlp_checkpoint, restore_nlp_checkpoint
from floorflow.nlp_flow.model import CompactNLPDecoder

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / ".data/nlp/tokenizer_v1/tokenizer.json",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    checkpoint = load_nlp_checkpoint(args.checkpoint.resolve(), map_location=device)
    model = CompactNLPDecoder(model_config_from_experiment(checkpoint["config"]))
    restore_nlp_checkpoint(checkpoint, model=model, device=device)
    model = model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer.resolve()))
    turns: list[tuple[str, str]] = []
    print("NLP Flow chat. Commands: /reset, /quit")
    while True:
        try:
            message = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message in {"/quit", "/exit"}:
            break
        if message == "/reset":
            turns.clear()
            print("Conversation reset.")
            continue
        prompt = render_conversation(turns, message)
        result = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + len(turns),
            device=device,
        )
        print(f"Assistant: {result.text}")
        print(
            f"[{result.generated_tokens} tokens, "
            f"{result.generated_tokens / max(result.elapsed_seconds, 1.0e-6):.1f} tok/s]"
        )
        turns.append((message, result.text))


if __name__ == "__main__":
    main()
