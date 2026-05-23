#!/usr/bin/env python3
"""
Phase 1 dataset generator — text-only autobench JSONL for Qwen3-VL-8B.

Samples token IDs only from the safe text pool (0..vocab_size-1 minus all_special_ids)
to avoid the Qwen3-VL multimodal code path (<|image_pad|>, <|vision_start|>, ...) that
OOMs the vision tower. Writes to datasets/qwen3vl8b/ (does NOT touch old datasets/case*.jsonl).

Usage (from repo root):
    HF_HUB_OFFLINE=1 python3 experiments/qwen3vl8b/phase1/scripts/gen_datasets.py
"""

import hashlib, json, os, random
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB = Path("/data/sglang-vllm-profiler")
DATASETS_DIR = LAB / "datasets/qwen3vl8b"
SHA_FILE = LAB / "experiments/qwen3vl8b/phase1/raw/dataset_sha256.txt"
SEED = 1

CASES = {
    "caseA_short":       dict(prompt_len=128,  output_len=128, n=600),
    "caseB_longprefill": dict(prompt_len=2048, output_len=128, n=300),
    "caseC_batched":     dict(prompt_len=512,  output_len=128, n=2500),
    "caseD_decode":      dict(prompt_len=512,  output_len=512, n=1200),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_prompt(tok, pool, length):
    ids = random.choices(pool, k=length)
    return tok.decode(ids, skip_special_tokens=False)


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"Loading tokenizer from {SNAPSHOT}")
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    vocab_size = tok.vocab_size
    special = set(tok.all_special_ids)
    print(f"  vocab_size={vocab_size}  excluding special ids={sorted(special)}")
    safe_ids = [i for i in range(vocab_size) if i not in special]
    print(f"  safe pool size={len(safe_ids)}")

    random.seed(SEED)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SHA_FILE.parent.mkdir(parents=True, exist_ok=True)
    sha_lines = []

    for case, cfg in CASES.items():
        out_path = DATASETS_DIR / f"{case}.jsonl"
        print(f"  Generating {case}: prompt={cfg['prompt_len']} out={cfg['output_len']} n={cfg['n']}")
        rows = []
        for _ in range(cfg["n"]):
            prompt = gen_prompt(tok, safe_ids, cfg["prompt_len"])
            actual_len = len(tok.encode(prompt, add_special_tokens=False))
            rows.append({
                "prompt": prompt,
                "output_len": cfg["output_len"],
                "prompt_len": actual_len,
                "metadata": {"source_dataset_name": "text_only_random", "source_dataset_path": "custom"},
            })
        lens = [r["prompt_len"] for r in rows]
        print(f"    prompt_len min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.1f}")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = sha256(out_path)
        sha_lines.append(f"{digest}  {out_path}")
        print(f"    written {out_path}")

    with open(SHA_FILE, "w") as f:
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("# text-only prompts, token IDs 0..(vocab_size-1) excluding all_special_ids, SEED=1\n")
        f.write("\n".join(sha_lines) + "\n")
    print(f"\nSHA-256 logged to {SHA_FILE}")


if __name__ == "__main__":
    main()
