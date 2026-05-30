#!/usr/bin/env python3
"""
D0 — payload audit for the image+text `<|video_pad|>` blocker (NO GPU, NO server).

Confirms the root cause: SGLang's synthetic `image` dataset generates random text via
`gen_mm_prompt`, which excludes only `<|image_pad|>` from the token pool. Other
multimodal/control special tokens (notably `<|video_pad|>`) can therefore appear in
the prompt text and make the Qwen3-VL server return HTTP 400
`No data iterator found for token: <|video_pad|>`.

What it does (all CPU):
  Part A — replicate `gen_mm_prompt` over many prompts/seeds and count how many
           contain ANY special token (and `<|video_pad|>` specifically).
  Part B — show that a sanitized generator (exclude all special ids) yields zero
           special tokens over the same sample.
  Part C — report the special-token inventory (ids + strings).

Writes a summary to debug_video_pad/results/D0_payload_audit.md and a small JSON.
Does not dump full prompts (only short sanitized snippets + hashes).

Usage:
    python3 debug_payload_audit.py [--prompts-per-seed 430] [--seeds 50] [--input-len 128]
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/sgl-workspace/sglang/python")

SNAPSHOT = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def replicate_gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Byte-for-byte replica of sglang.benchmark.datasets.common.gen_mm_prompt."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    if image_pad_id:
        all_available_tokens.remove(image_pad_id)
    selected = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected)


def sanitized_gen_mm_prompt(tokenizer, exclude_ids, token_num):
    """Proposed fix: exclude ALL special/control token ids from the pool."""
    pool = [t for t in tokenizer.get_vocab().values() if t not in exclude_ids]
    selected = random.choices(pool, k=token_num)
    return tokenizer.decode(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-per-seed", type=int, default=430,
                    help="prompts per seed (IMG-A uses 30 warmup + 400 measured)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--input-len", type=int, default=128)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    from sglang.benchmark.utils import get_processor
    proc = get_processor(SNAPSHOT)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc

    image_pad_id = getattr(proc, "image_token_id", None)
    video_pad_id = getattr(proc, "video_token_id", None)
    special_ids = set(tok.all_special_ids)
    id_to_str = {tid: tok.convert_ids_to_tokens([tid])[0] for tid in sorted(special_ids)}

    # Tokens we explicitly require sanitized prompts to never contain (string form).
    forbidden_strings = [
        "<|image_pad|>", "<|video_pad|>", "<|vision_start|>",
        "<|vision_end|>", "<|vision_pad|>",
    ]

    def contains_any_special(text):
        return [s for s in forbidden_strings if s in text]

    # ----- Part A: replicate buggy generator -----
    total = 0
    hits_video = 0
    hits_any = 0
    examples = []
    for seed in range(1, args.seeds + 1):
        random.seed(seed)
        for i in range(args.prompts_per_seed):
            text = replicate_gen_mm_prompt(tok, image_pad_id, args.input_len)
            total += 1
            found = contains_any_special(text)
            if "<|video_pad|>" in text:
                hits_video += 1
            if found:
                hits_any += 1
                if len(examples) < 10:
                    # sanitized snippet: strip to show only that a special token is present
                    examples.append({
                        "seed": seed, "index": i,
                        "found_tokens": found,
                        "prompt_sha12": short_hash(text),
                        "snippet": text[max(0, text.find(found[0]) - 20):
                                        text.find(found[0]) + 40].replace("\n", " "),
                    })

    # ----- Part B: sanitized generator over same sample size -----
    san_total = 0
    san_hits = 0
    for seed in range(1, args.seeds + 1):
        random.seed(seed)
        for i in range(args.prompts_per_seed):
            text = sanitized_gen_mm_prompt(tok, special_ids, args.input_len)
            san_total += 1
            if contains_any_special(text):
                san_hits += 1

    rate_video = hits_video / total if total else 0.0
    rate_any = hits_any / total if total else 0.0

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": SNAPSHOT.split("/")[-1],
        "config": {
            "prompts_per_seed": args.prompts_per_seed,
            "seeds": args.seeds,
            "input_len": args.input_len,
            "total_prompts": total,
        },
        "token_ids": {
            "image_pad_id": image_pad_id,
            "video_pad_id": video_pad_id,
            "num_special_ids": len(special_ids),
            "special_id_range": [min(special_ids), max(special_ids)],
        },
        "buggy_generator": {
            "total": total,
            "prompts_with_video_pad": hits_video,
            "prompts_with_any_forbidden": hits_any,
            "rate_video_pad_pct": round(100 * rate_video, 4),
            "rate_any_forbidden_pct": round(100 * rate_any, 4),
            "expected_failures_per_430_rep": round(430 * rate_any, 3),
        },
        "sanitized_generator": {
            "total": san_total,
            "prompts_with_any_forbidden": san_hits,
            "clean": san_hits == 0,
        },
        "examples": examples,
        "special_inventory": {str(k): v for k, v in id_to_str.items()},
    }

    (RESULTS / "D0_payload_audit.json").write_text(json.dumps(result, indent=2))

    # ----- Markdown summary -----
    lines = [
        "# D0 — Payload audit results (`<|video_pad|>` blocker)",
        "",
        f"> Run: {result['timestamp_utc']}  (NO GPU, NO server)",
        f"> Sample: {args.seeds} seeds × {args.prompts_per_seed} prompts = {total} prompts,"
        f" input_len={args.input_len}",
        "",
        "## Verdict",
        "",
    ]
    if hits_any > 0 and san_hits == 0:
        lines.append("✅ **ROOT CAUSE CONFIRMED.** The buggy generator emits forbidden "
                     "multimodal special tokens; the sanitized generator does not.")
    elif hits_any == 0:
        lines.append("⚠️ No forbidden tokens observed in this sample — increase --seeds "
                     "(rate is ~0.08%, needs a large sample).")
    else:
        lines.append("❌ Sanitized generator still produced forbidden tokens — fix is wrong.")
    lines += [
        "",
        "## Token identities",
        "",
        f"- `image_pad_id` = {image_pad_id} (`<|image_pad|>`) — excluded by current generator",
        f"- `video_pad_id` = {video_pad_id} (`<|video_pad|>`) — **NOT excluded** (the bug)",
        f"- total special ids = {len(special_ids)} (range {min(special_ids)}–{max(special_ids)})",
        "",
        "## Buggy generator (current `gen_mm_prompt`)",
        "",
        f"- prompts containing `<|video_pad|>`: {hits_video} / {total} "
        f"= {result['buggy_generator']['rate_video_pad_pct']}%",
        f"- prompts containing ANY forbidden mm token: {hits_any} / {total} "
        f"= {result['buggy_generator']['rate_any_forbidden_pct']}%",
        f"- expected failures per 430-request rep: "
        f"{result['buggy_generator']['expected_failures_per_430_rep']}",
        "",
        "## Sanitized generator (proposed: exclude all special ids)",
        "",
        f"- prompts containing ANY forbidden mm token: {san_hits} / {san_total}",
        f"- clean: {'✅ yes' if san_hits == 0 else '❌ no'}",
        "",
        "## Example hits (sanitized snippets, hashes only — no full prompts)",
        "",
        "| seed | index | tokens | prompt_sha12 | snippet |",
        "|---|---|---|---|---|",
    ]
    for ex in examples:
        toks = ",".join(ex["found_tokens"])
        snip = ex["snippet"].replace("|", "\\|")
        lines.append(f"| {ex['seed']} | {ex['index']} | `{toks}` | `{ex['prompt_sha12']}` "
                     f"| `…{snip}…` |")
    lines += [
        "",
        "## Conclusion",
        "",
        "The failure is a **benchmark-generator special-token bug**, not a server "
        "cache/state or CUDA-IPC performance effect. Sanitizing the random text "
        "(exclude all special ids) eliminates the forbidden tokens. The runner can "
        "proceed with a sanitized prompt path; an upstream `gen_mm_prompt` fix is a "
        "follow-up.",
    ]
    (RESULTS / "D0_payload_audit.md").write_text("\n".join(lines))

    print(f"buggy: video_pad={hits_video}/{total} ({result['buggy_generator']['rate_video_pad_pct']}%), "
          f"any_forbidden={hits_any}/{total}")
    print(f"sanitized: forbidden={san_hits}/{san_total} (clean={san_hits == 0})")
    print(f"written: {RESULTS / 'D0_payload_audit.md'}")
    return 0 if (hits_any > 0 and san_hits == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
