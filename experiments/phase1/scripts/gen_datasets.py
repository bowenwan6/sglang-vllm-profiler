#!/usr/bin/env python3
"""
Generate byte-identical autobench JSONL datasets for Phase 1 with text-only prompts.

Excludes all special token IDs (>= vocab_size) to avoid triggering the
Qwen3-VL multimodal code path (<|image_pad|>, <|vision_start|>, etc.)
which caused OOM crashes with the auto_benchmark random generator.

Usage (from /data/profiling_lab):
    python3 experiments/phase1/scripts/gen_datasets.py
"""

import hashlib, json, random
from pathlib import Path
from transformers import AutoTokenizer

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB = Path("/data/profiling_lab")
SEED = 1

CASES = {
    "caseA_short":       dict(prompt_len=128,  output_len=128,  n=600),
    "caseB_longprefill": dict(prompt_len=2048, output_len=128,  n=300),
    "caseC_batched":     dict(prompt_len=512,  output_len=128,  n=2500),
    "caseD_decode":      dict(prompt_len=512,  output_len=512,  n=1200),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_prompt(tok, token_ids_pool: list, length: int) -> str:
    """Sample `length` token IDs from the pool and decode to text."""
    ids = random.choices(token_ids_pool, k=length)
    return tok.decode(ids, skip_special_tokens=False)


def main():
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"

    print(f"Loading tokenizer from {SNAPSHOT}")
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    vocab_size = tok.vocab_size  # 151643 — IDs 0..151642 are safe text tokens
    print(f"  vocab_size: {vocab_size}")
    print(f"  Excluding all_special_ids: {sorted(tok.all_special_ids)}")

    # Only use regular text tokens (IDs 0..vocab_size-1), no special tokens
    safe_ids = [i for i in range(vocab_size) if i not in tok.all_special_ids]
    print(f"  Safe token pool size: {len(safe_ids)}")

    random.seed(SEED)

    datasets_dir = LAB / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    sha_lines = []

    for case_name, cfg in CASES.items():
        out_path = datasets_dir / f"{case_name}.jsonl"
        if out_path.exists():
            print(f"  {case_name}: already exists, regenerating (overwrite)")

        print(f"  Generating {case_name}: prompt={cfg['prompt_len']} out={cfg['output_len']} n={cfg['n']}")
        rows = []
        for _ in range(cfg["n"]):
            prompt = gen_prompt(tok, safe_ids, cfg["prompt_len"])
            actual_len = len(tok.encode(prompt, add_special_tokens=False))
            row = {
                "prompt": prompt,
                "output_len": cfg["output_len"],
                "prompt_len": actual_len,
                "metadata": {
                    "source_dataset_name": "text_only_random",
                    "source_dataset_path": "custom",
                },
            }
            rows.append(row)

        actual_lens = [r["prompt_len"] for r in rows]
        print(f"    prompt_len: min={min(actual_lens)} max={max(actual_lens)} avg={sum(actual_lens)/len(actual_lens):.1f}")
        print(f"    output_len: {cfg['output_len']} (fixed)")

        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        digest = sha256(out_path)
        sha_lines.append(f"{digest}  {out_path}")
        print(f"  {case_name}: written {out_path}")

    sha_file = LAB / "experiments/phase1/raw/dataset_sha256.txt"
    sha_file.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    with open(sha_file, "w") as f:
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("# Note: text-only prompts, token IDs 0..(vocab_size-1) excluding all_special_ids\n")
        f.write("\n".join(sha_lines) + "\n")
    print(f"\nSHA-256 logged to {sha_file}")


if __name__ == "__main__":
    main()
