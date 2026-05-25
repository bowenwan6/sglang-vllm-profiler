# Case A — Phase 5.2 H1 Controlled Intervention Protocol

Main experiment `qwen3vl8b`. Workload **Case A** (128→128, c=1). GPU **3** only.

## Objective

Test **causally** whether expanding SGLang's **prefill graph / compile coverage** reduces end-to-end
benchmark **TTFT** on Case A. This is the validation Phase 5.1 could not do: Phase 5.1 confirmed the
*mechanism* (for this Qwen3-VL config, decode CUDA graph is nominally on but the prefill-side piecewise
CUDA graph is auto-disabled for VLM models, and `enable_torch_compile=False`), and also found that the
existing SGLang profiler traces **cannot reliably quantify serving-path graph coverage** (a
profiling/captured-window confound: even with `disable_cuda_graph=False`, graph replay was not reliably
observed in the formal DECODE trace). **H1 therefore remains a hypothesis, not a confirmed root cause.**

## Methodology guards

- This is a **controlled intervention**; the primary comparison is **TTFT differences between SGLang
  variants (S0–S3)**, measured on the real serving path under the Case-A locked benchmark.
- **Profiler traces are NOT the primary evidence** this round; benchmark TTFT is.
- **vLLM (V0) is only a contemporaneous reference anchor** on the same GPU / time window — not a
  causal comparison variable for the SGLang flag effect.
- `--enforce-piecewise-cuda-graph` is a **testing lever** ("skip auto-disable … for testing",
  `server_args.py:1276`). A positive result validates the *direction* only; it does **not** by itself
  equal a production fix.
- `--enable-torch-compile` and `--enforce-piecewise-cuda-graph` are **independent alternatives** —
  never combined in one run.
- No SGLang source changes. No re-collection of Phase 3 traces.

## Locked benchmark protocol (identical across all variants)

- Model snapshot `0c351dd…`; dtype bf16; TP=1; attention backend flashinfer (SGLang).
- Dataset `datasets/qwen3vl8b/caseA_short.jsonl` (sha `fab4917772e08744…`).
- `sglang.bench_serving --dataset-name autobench`, `--max-concurrency 1`, `--num-prompts 400`,
  `--warmup-requests 30`, `--seed 1`, `--extra-request-body '{"temperature":0,"top_p":1}'`,
  `--output-details`. **reps = 3.**
- One server at a time on GPU 3; fresh server per variant; confirm GPU 3 < 2000 MiB after each shutdown.

## Variants

| ID | Framework | Flags | Role |
|---|---|---|---|
| **V0** | vLLM | default (`--dtype bfloat16 --tensor-parallel-size 1`) | contemporaneous reference anchor (not a causal variable) |
| **S0** | SGLang | `--disable-overlap-schedule` | baseline (decode graph nominally ON, VLM piecewise prefill graph OFF, compile OFF) |
| **S1** | SGLang | `--disable-overlap-schedule --disable-cuda-graph --disable-piecewise-cuda-graph` | negative control (graph reduced) |
| **S2** | SGLang | `--disable-overlap-schedule --enforce-piecewise-cuda-graph` | coverage-expansion candidate 1 (force prefill piecewise graph) |
| **S3** | SGLang | `--disable-overlap-schedule --enable-torch-compile` | coverage-expansion candidate 2 (independent alternative) |

S2/S3: run a minimal greedy **correctness smoke check** before benchmarking. If server start fails,
OOM, CUDA error, incorrect/empty output, or `failed requests > 0`, **record and skip that variant's
perf result as invalid** — do not fix via source changes.

## Recorded per variant

server flags · GPU id · model snapshot · dataset SHA · framework version · timestamp · warmup / reps /
num_prompts / concurrency · error rate / failed requests · per-rep TTFT p50/p95/p99 · TTFT p50 median +
CV · TPOT p50/p99 · throughput.

## Decision rules (baseline = S0)

- **H1 strengthened** if S2 or S3 (0 errors, correct output, acceptable CV) lowers **TTFT p50 median
  by >5%** vs S0 **and** visibly narrows the gap to the V0 anchor.
- S1 markedly worse than S0 → supporting evidence that graph coverage affects TTFT.
- S2/S3 no improvement, unstable, or unable to run → **H1 weakened / mechanism-level only**; no causal claim.
- Judge on **median + CV**, never a single outlier rep.

## Stop conditions

GPU 3 unavailable or not freed < 2000 MiB · server crash / OOM / CUDA error / traceback · correctness
smoke fails · failed requests > 0 · candidate flag unsupported · need for source change · conflicting
uncommitted working-tree change.

## GPU / safety

`CUDA_VISIBLE_DEVICES=3` for every server/client; never GPU 0/1/7; serial lifecycle; GPU 3 < 2000 MiB
between launches; new artifacts only under the Phase-5 paths; do not touch Phase 1/2 raw JSON, Phase 3
traces, or SGLang source.
