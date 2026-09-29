# Q3 — ViT CUDA graph on Qwen3-VL-8B: results

Generated from `results/*.json`. Design and pre-registered predictions: [`PLAN.md`](../PLAN.md) (2026-09-29 amendment applies).

## Stack

- sglang `0.0.0.dev19038+g89e1316ea` @ `89e1316eae9a` (measurement patch applied: {'runner': True, 'model': True, 'vision': True}), torch `2.13.0+cu130 13.0 True 1`
- GPU: NVIDIA H200, 595.71.05 (143771 MiB); snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; profiler repo `exp/q3-vit-graph` @ `cca07bf`; output length 16
- server: `/data/bowenwan6/home/miniforge3/envs/sgl-profiler/bin/python3 -m sglang.launch_server --model-path /data/bowenwan6/home/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --port 30000 --dtype bfloat16 --tp 1 --host 127.0.0.1 --attention-backend flashinfer --mm-attention-backend fa3 --mm-feature-transport cuda_ipc --cuda-graph-backend-prefill disabled --disable-radix-cache --chunked-prefill-size 8192 --mm-preprocess-cache-size-mb 0 --enable-request-time-stats-logging --skip-server-warmup --mem-fraction-static 0.75 --enable-precise-embedding-interpolation`

## Parity (both arms encode the same fixtures)

Pre-registered verdict **FAIL**; post-hoc downstream rule (approved by Bowen (chat, 2026-09-29 13:55 UTC)): on = **PASS**, off_rot = PASS, on_default_interp = FAIL; rule: greedy tokens identical or first divergence at an eager margin < 0.5 nat; mean |dlogprob| over shared tokens <= 0.05 nat

**PASS_WITH_DEVIATION** — encoder-output relative error per image fixture: [0.0582, 0.0582, 0.069] (tolerance 0.02); reasons: ["encoder outputs differ beyond 0.02: [{'i': 0, 'shape': [100, 16384], 'rel_fro': 0.05821667239069939, 'rel_max': 0.23099415004253387, 'max_abs': 1.234375}, {'i': 1, 'shape': [100, 16384], 'rel_fro': 0.05821667239069939, 'rel_max': 0.23099415004253387, 'max_abs': 1.234375}, {'i': 2, 'shape': [256, 16384], 'rel_fro': 0.06895607709884644, 'rel_max': 0.2975171208381653, 'max_abs': 1.357421875}]"]
- greedy text: 4/5 fixtures identical; divergences: `image512_colors` at token 6 (off-arm margin 0.00 nat, benign)
- eager with the unfused rotary vs eager (implementation noise floor): rel_fro [0.0212, 0.0212, 0.0723]
- graph arm vs eager with the unfused rotary: rel_fro [0.0557, 0.0557, 0.0702]
- graph arm with the **default** interpolation flag vs eager: rel_fro [0.4694, 0.4694, 0.3105] (upstream-issue evidence)

## Pilot: the eager encoder's time budget and the amended H1 prediction

Servers: ['off', 'on', 'off_rot', 'off_notiming']. Per workload: 5 warmup, 5 profiled, 30 measured requests; c=1. W_v is the unprofiled `VIT_TIMING` CPU wall, G_v the trace's GPU busy time, U* = W_v − G_v, dG_rot the extra GPU time of the unfused rotary, overlap = min(G_v, max(0, W_l − G_l)) from the sync-free eager profile; r = 1.0 ms. Prediction: lo = U* − r − dG_rot, point = lo + overlap, hi = W_v − r − dG_rot.

| workload | patches | W_v | G_v | **U\*** | infl. | dG_rot | overlap | launches off→on | graph launches on | **pred lo / point / hi** | pilot ΔTTFT | TTFT off | TTFT on | verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `R0_text` | 0 | — | — | **—** | — | — | — | —→— | — | **— / — / —** | -0.10 | 30.80 | 30.91 | yes |
| `R1_256` | 256 | 16.79 | 2.04 | **14.75** | 5.35 | 0.75 | 2.04 | 426→10 | 1 | **13.00 / 15.04 / 15.04** | 14.91 | 54.57 | 39.66 | yes |
| `R2_360p` | 880 | 16.46 | 3.44 | **13.02** | 5.83 | 1.17 | 3.44 | 426→10 | 1 | **10.85 / 14.29 / 14.29** | 10.45 | 58.02 | 47.56 | yes |
| `R3_512` | 1024 | 16.56 | 3.75 | **12.81** | 5.94 | 1.29 | 3.75 | 426→10 | 1 | **10.52 / 14.27 / 14.27** | 10.57 | 59.44 | 48.87 | yes |
| `R4_640` | 1600 | 16.59 | 5.86 | **10.73** | 6.46 | 1.79 | 5.86 | 426→10 | 1 | **7.94 / 13.80 / 13.80** | 7.26 | 65.34 | 58.08 | yes |
| `R5_720p` | 3520 | 17.27 | 12.06 | **5.21** | 6.61 | 3.73 | 4.04 | 426→10 | 1 | **0.48 / 4.52 / 12.54** | -1.21 | 89.29 | 90.50 | yes |
| `R6_1080p` | 8160 | 17.57 | 35.06 | **-17.50** | 6.22 | 8.29 | 0.00 | 426→10 | 1 | **-26.79 / -26.79 / 8.27** | -6.21 | 186.43 | 192.64 | yes |

Gate G1: **GO**  warnings: ['D6: TTFT differs between 16 and 128 output tokens by 7.5%; run the sweep with --output-len 128 to stay comparable']; output-length check: {'ttft_out16': 54.573, 'ttft_out128': 50.471, 'rel_diff': 0.0752}

### TTFT decomposition, eager arm (critical path, medians)

prefill step = `Q3_STEP_EXTEND` critical path from the sync-free profile; ViT = `VIT_TIMING` GPU span (encoder critical path); LM = step − ViT; outside forward = client TTFT − queue − step (HTTP, base64/PNG decode, image processor, tokenisation, feature transport, first-token streaming; random-content PNGs do not compress, so this term carries a benchmark artifact that grows with resolution).

| workload | client TTFT | queue | prefill step (crit) | ViT (crit) | of which un-overlapped U\* | LM | outside forward |
|---|---|---|---|---|---|---|---|
| `R0_text` | 30.80 | 0.41 | 25.09 | — | — | — | 5.31 |
| `R1_256` | 54.57 | 0.12 | 48.83 | 16.83 | 14.75 | 32.00 | 5.62 |
| `R2_360p` | 58.02 | 0.10 | 49.10 | 16.50 | 13.02 | 32.60 | 8.82 |
| `R3_512` | 59.44 | 0.10 | 49.36 | 16.60 | 12.81 | 32.76 | 9.99 |
| `R4_640` | 65.34 | 0.10 | 50.41 | 16.62 | 10.73 | 33.79 | 14.83 |
| `R5_720p` | 89.29 | 0.11 | 50.80 | 17.37 | 5.21 | 33.44 | 38.38 |
| `R6_1080p` | 186.43 | 0.13 | 90.92 | 36.77 | -17.50 | 54.16 | 95.38 |

### H2 — what the fixed cost is made of

- ❌ 256²: encoder ≥ 40 % of TTFT
- ✅ 256²: un-overlapped ≥ 60 % of the encoder call
- ✅ 720p and 1080p: G_v > U* (compute-dominated)

shares: {"R1_256": {"vit_share_of_ttft": 0.308, "unoverlapped_share_of_vit": 0.879, "G_v_gt_U": false}, "R2_360p": {"vit_share_of_ttft": 0.284, "unoverlapped_share_of_vit": 0.791, "G_v_gt_U": false}, "R3_512": {"vit_share_of_ttft": 0.279, "unoverlapped_share_of_vit": 0.774, "G_v_gt_U": false}, "R4_640": {"vit_share_of_ttft": 0.254, "unoverlapped_share_of_vit": 0.647, "G_v_gt_U": false}, "R5_720p": {"vit_share_of_ttft": 0.194, "unoverlapped_share_of_vit": 0.301, "G_v_gt_U": true}, "R6_1080p": {"vit_share_of_ttft": 0.197, "unoverlapped_share_of_vit": -0.996, "G_v_gt_U": true}}

**H2 NOT SUPPORTED.**

## Sweep: measured TTFT effect (A/B/B/A blocks, one server per cell)

| workload | patches | blocks | TTFT off p50 | TTFT on p50 | saving (median) | SE | effect | paired spread | gate | verified |
|---|---|---|---|---|---|---|---|---|---|---|
| `R0_text` | 0 | 3 | 28.26 ms | 27.92 ms | **0.34 ms** | 0.10 | **-1.19%** | 1.16 pp | PASS | yes |
| `R1_256` | 256 | 3 | 45.96 ms | 32.44 ms | **13.52 ms** | 0.70 | **-29.41%** | 4.15 pp | PASS | yes |
| `R2_360p` | 880 | 3 | 52.99 ms | 39.89 ms | **12.89 ms** | 0.30 | **-24.42%** | 1.98 pp | PASS | yes |
| `R3_512` | 1024 | 3 | 53.88 ms | 40.95 ms | **12.96 ms** | 0.10 | **-24.04%** | 0.64 pp | PASS | yes |
| `R4_640` | 1600 | 3 | 59.61 ms | 47.66 ms | **11.89 ms** | 0.50 | **-19.97%** | 2.54 pp | PASS | yes |
| `R5_720p` | 3520 | 4 | 86.99 ms | 87.63 ms | **-1.20 ms** | 0.72 | **+1.39%** | 4.02 pp | WEAK | yes |
| `R6_1080p` | 8160 | 4 | 183.71 ms | 191.15 ms | **-7.44 ms** | 0.48 | **+4.05%** | 1.04 pp | PASS | yes |

## H1: does the eager trace predict the graph's gain?

tol = max(1.0 ms, 25 % of |pred|, 2·SE); non-informative if 2·SE > |pred| + 1 ms. Supported if ≥ 4 informative sizes with ≤ 1 miss, text control inside the 3.6 % floor, saving non-increasing with patches (±1.0 ms) and 1080p inside the floor.

| workload | patches | pred lo / point / hi | measured saving | SE | tol | error | informative | in interval | pass |
|---|---|---|---|---|---|---|---|---|---|
| `R1_256` | 256 | 13.00 / 15.04 / 15.04 | 13.52 ms | 0.70 | 3.76 | -1.52 | yes | yes | ✅ |
| `R2_360p` | 880 | 10.85 / 14.29 / 14.29 | 12.89 ms | 0.30 | 3.57 | -1.40 | yes | yes | ✅ |
| `R3_512` | 1024 | 10.52 / 14.27 / 14.27 | 12.96 ms | 0.10 | 3.57 | -1.31 | yes | yes | ✅ |
| `R4_640` | 1600 | 7.94 / 13.80 / 13.80 | 11.89 ms | 0.50 | 3.45 | -1.91 | yes | yes | ✅ |
| `R5_720p` | 3520 | 0.48 / 4.52 / 12.54 | -1.20 ms | 0.72 | 1.43 | -5.71 | yes | no | ❌ |
| `R6_1080p` | 8160 | -26.79 / -26.79 / 8.27 | -7.44 ms | 0.48 | 6.70 | 19.35 | yes | yes | ❌ |

Text control `R0_text`: -1.19% (inside the 3.6 % floor).

**H1 NOT SUPPORTED** — conditions: {'informative_sizes>=4': True, 'misses<=1': False, 'text_control_inside_floor': True, 'saving_non_increasing': True, '1080p_inside_floor': False}; informative: ['R1_256', 'R2_360p', 'R3_512', 'R4_640', 'R5_720p', 'R6_1080p']; misses: ['R5_720p', 'R6_1080p']; non-informative: []

## Mixed resolutions: exact-shape keys under realistic variety (H3)

300 requests, heights 256–720 and widths 256–1280 px drawn uniformly with a fixed seed, so both arms see the same sequence. The graph key is the patch count, so distinct 32-px grids with the same product share a graph: 165 captures and a hit rate of 0.45 were expected.

| | off | on |
|---|---|---|
| **mean TTFT** (H3 metric) | 64.83 ms | 94.70 ms (penalty 29.88 ms) |
| TTFT p50 / p99 | 62.66 / 93.80 ms | 75.63 / 189.01 ms |
| graph captures / requests | — | 167 / 300 (hit rate by key 0.443) |
| capture cost p50 / max / total | — | 27.00 / 4580.30 / 9769 ms |
| TTFT, first-seen shapes (p50) | — | 87.98 ms, penalty vs off on the same requests 22.02 ms |
| TTFT, repeated shapes (p50) | — | 51.59 ms, gain vs off on the same requests 8.22 ms |
| GPU memory end / peak | 111186 / 111186 MiB | 127646 / 126116 MiB (guard tripped: False) |
- ✅ mean TTFT worse on the graph arm
- ✅ hit rate within ±5 pp of 45 %
- ✅ repeated shapes still gain

**H3 SUPPORTED.**

## Verdicts

- **H2: NOT SUPPORTED**
- **H1: NOT SUPPORTED**
- **H3: SUPPORTED**

