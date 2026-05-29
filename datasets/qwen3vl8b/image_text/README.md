# datasets/qwen3vl8b/image_text/ — Qwen3-VL image+text benchmark data (issue #4)

Data identity for the #4 image+text benchmarks. See the protocol at
`experiments/qwen3vl8b/v2/image_text_benchmarks/protocol.md`.

## Dataset model: synthetic, harness-native

The #4 benchmarks use SGLang's built-in `image` dataset (`sglang.bench_serving --dataset-name image`),
which **synthesizes images inline** (random or blank pixels → base64 data URIs embedded in each request).
There is **no external URL download** and **no large image binary** to check in.

Images are reproducible: `bench_serving` seeds both `random` and `numpy.random` from `--seed`. A run is
therefore fully determined by its **recipe**, not by a checked-in file.

## Canonical dataset identity (record per run, in place of a file sha256)

- SGLang `bench_serving` **git commit** / version (the generator's provenance)
- `--seed`
- `--image-count`, `--image-resolution`, `--image-format`, `--image-content`, `--random-image-count`
- `--random-input-len`, `--random-output-len`, `--random-range-ratio`
- `--num-prompts`

Headline defaults (see protocol §3): `--image-resolution 720p`, `--image-format png`,
`--image-content random`, `--image-count 1`, `--seed 1`.

## Optional byte-level provenance

If a checked-in artifact is later required, dump the generated request payloads to
`<workload>.generated.jsonl` here and record its sha256 in this file. Not required for the headline — the
seed+params recipe is the canonical identity.

## If real images are ever used (not the current plan)

Real images must be **checked in** under `assets/` with a per-file sha256 in
[`assets_manifest.md`](assets_manifest.md) — never fetched at runtime. Prompt JSONL files would live here
with a recorded sha256. This path is documented for completeness; the current plan is synthetic.

## Status

No generated/real data committed yet — created when an approved Phase 4.1+ run produces it.
