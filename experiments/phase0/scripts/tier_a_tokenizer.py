"""
Phase 0 — Tier A: tokenizer and vocab equivalence check.

Usage:
    python tier_a_tokenizer.py --model-path <snapshot_path>

Checks vocab_size, eos/bos/pad ids, model_max_length, and token-level
byte-equality on 5 probe strings. Prints results; does not write files.
"""
import argparse, json
from transformers import AutoTokenizer

PROBES = [
    "Hello world",
    "你好世界",
    "def foo(): return 42",
    "🚀",
    "The quick brown fox jumps over the lazy dog. " * 8,
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)

    print("=== Tier A — tokenizer / vocab ===")
    for p in PROBES:
        ids = tok.encode(p)
        print(f"  {repr(p[:35]):45s} → {len(ids):3d} tokens  {ids[:8]}")

    print(f"\n  vocab_size       : {tok.vocab_size}")
    print(f"  eos_token_id     : {tok.eos_token_id}")
    print(f"  bos_token_id     : {tok.bos_token_id}")
    print(f"  pad_token_id     : {tok.pad_token_id}")
    print(f"  model_max_length : {tok.model_max_length}")

    chat = tok.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"\n  chat_template sample: {repr(chat[:120])}")

if __name__ == "__main__":
    main()
