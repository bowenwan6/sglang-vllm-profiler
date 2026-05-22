# SGLang vs vLLM Profiling 当前状态报告（run2_qwen3vl8b）

## Abstract / Executive Summary

本轮实验比较 SGLang 与 vLLM 在 `Qwen/Qwen3-VL-8B-Instruct` text-only 路径上的 serving 表现。当前最重要的发现是：**SGLang 的主要差距集中在 TTFT，而不是 TPOT 或 decode throughput**。在 Phase 1 baseline 中，四个 workload 的 TPOT 基本 parity，但 SGLang 的 TTFT 均慢于 vLLM：Case A 为 4.89x，Case B 为 3.20x，Case C/D 约 1.3x。

Phase 2 进一步收敛了问题范围：Case A 中 `--disable-overlap-schedule` 可以把 residual gap 从 Phase 1 的 4.89x 收窄到 1.56x，说明 overlap scheduler 在短请求 c=1 场景有明确开销；Case C 经 W500 warmup 后稳定，确认 SGLang 在 batched c=16 场景仍慢约 1.32x；Case B 两边都 bimodal，结论带 confidence ceiling M；Case D residual gap 仅约 1.09x，优先级较低。

Phase 3 已完成 trace collection，并补采了 SGLang EXTEND/PREFILL traces。当前 trace 证据足够进入 Phase 4 triage。Phase 4 尚未开始，因此本文不下最终 root cause 或优化建议，只总结当前 benchmark、shaping 与 trace readiness 状态。

## 1. Experiment Setup

| Item | Value |
|---|---|
| Active run | `run2_qwen3vl8b` |
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Snapshot | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| GPU | Single H200, serialized runs |
| SGLang | `0.0.0.dev1+g0c8049d9b` |
| vLLM | `0.21.0` |
| Torch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| Dtype / TP | bf16 / TP=1 |
| Sampling | greedy: `temperature=0`, `top_p=1` |

run2 是重装机器后的重新测量。模型 snapshot 与 run1 一致，但 SGLang、vLLM、torch、CUDA、FlashInfer 等版本均不同，因此 run1 只能作为 historical reference，不能与 run2 数字直接混用。

## 2. Methodology / Phase Status

实验采用 phase-gated pipeline：先确认可比性，再找 gap，再排除配置/方差因素，最后采集 trace 供后续分析。Phase 3 只负责采证据，不负责解释；真正的 kernel/source/root-cause triage 会在 Phase 4 完成。

| Phase | Purpose | Status |
|---|---|---|
| Phase 0 | 验证模型、tokenizer、config 和 greedy 输出一致 | Complete |
| Phase 1 | 建立 baseline，判断 gap 来自 TTFT、TPOT 还是 throughput | Complete |
| Phase 2 | 做 shaping / variance gate，锁定可 profile 的 case | Complete |
| Phase 3 | 收集 SGLang/vLLM traces，不做解释 | Complete |
| Phase 4 | Trace triage，定位 kernel / scheduler / memory 等原因 | Pending |
| Phase 5 | 验证 top hypotheses | Pending |

## 3. Phase 1 Baseline Results

| Case | Workload | SGLang TTFT p50 | vLLM TTFT p50 | Ratio | TPOT | Note |
|---|---|---:|---:|---:|---|---|
| A | 128 -> 128, c=1 | 61.8 ms | 12.6 ms | 4.89x | parity | cleanest short-latency case |
| B | 2048 -> 128, c=1 | 66.7 ms | 20.8 ms | 3.20x | parity | vLLM bimodal |
| C | 512 -> 128, c=16 | 247.5 ms | 187.9 ms | 1.32x | parity | variance gate needed |
| D | 512 -> 512, c=16 | 253.0 ms | 189.7 ms | 1.33x | parity | p99 tail / bimodal |

Phase 1 的核心结论是：**TTFT 是唯一主要 gap**。TPOT 和 throughput 基本 parity，说明 decode token-by-token 路径没有明显落后。A -> B 中 prompt 长度增加 16x，但 SGLang TTFT 只增加 4.9 ms，说明 prefill compute 不是主因，更像固定 scheduler / dispatch overhead。

## 4. Phase 2 Shaping / Variance Gate

| Case | Winner Config | SGLang TTFT p50 | vLLM Ref | Residual Gap | Phase 3 Protocol |
|---|---|---:|---:|---:|---|
| A | `--disable-overlap-schedule` | 19.6 ms, CV 3.2% | 12.6 ms | 1.56x | warmup 30, 3 reps |
| B | default | 30.3 ms, CV 68.4% | 21.5 ms, CV 85.9% | 1.41x | warmup 300, 5 reps |
| C | default, W500 | 249.1 ms, CV 2.9% | 189.0 ms, CV 1.9% | 1.32x | warmup 500, 5 reps |
| D | default, W30 | 206.2 ms, CV 3.3% | 189.7 ms | 1.09x | warmup 30, 3 reps |

Case A 的 shaping 表明 overlap scheduler 在 c=1 short-latency 场景有真实成本。关闭 overlap schedule 后，TTFT 从 default 约 21.8 ms 降到 19.6 ms，Phase 1 的 4.89x gap 收窄到 1.56x。这使 Case A 成为 Phase 4 的最高优先级对象。

Case C 的结论经过一次重要修正。W30/W100/W300 下 SGLang CV 都未通过 5% gate，并曾出现 “SGLang faster / 0.79x” 的假象。W500 5 reps 后，SGLang 稳定在 249.1 ms，CV 2.9%，确认 SGLang 仍慢约 1.32x，与 Phase 1 一致。之前的 reversal 是 under-warmup artifact。

Case B 两边都 bimodal，所有 cross-framework 结论带 confidence ceiling M。Case D residual gap 很小，预计 Phase 4 payoff 较低。

## 5. Phase 3 Trace Readiness

| Trace Group | Status |
|---|---|
| SGLang DECODE mapping/formal | A/B/C/D complete |
| vLLM prefill_like/decode_like | A/B/C/D complete |
| SGLang EXTEND mapping | A/B/C/D complete |
| SGLang EXTEND formal | A/C/D complete |
| Case B SGLang EXTEND formal | Missing after 8 attempts |

Phase 3 主采集完成后，最初 SGLang traces 只捕获到 DECODE stage。随后补采 EXTEND/PREFILL：使用 `max_new_tokens=1` 的 prefill-only load，将 profiler window 对准 EXTEND stage。补采结果为 7/8 成功：A/C/D 拿到 EXTEND mapping + formal，B 拿到 EXTEND mapping，但 graph-on EXTEND formal 在 8 次尝试后仍无法捕获。

这个缺口可以接受。Case B 本身 noisy 且 confidence ceiling M；同时 graph-off EXTEND mapping 已经提供 kernel -> source attribution。Phase 4 中需要明确标注：Case B prefill-stage perf timing 缺少 graph-on formal trace。

## 6. Current Interpretation

当前可支持的结论是：

1. **SGLang 的主要问题在 TTFT，不在 TPOT。**
2. **Case A 是最干净、最 actionable 的对象。** 关掉 overlap scheduler 后仍有 1.56x residual gap，Phase 4 应优先看 scheduler/dispatch overhead。
3. **Case C 是稳定 batched gap。** W500 后 SGLang 仍慢约 1.32x，适合分析 batch formation、prefill/decode transition、CUDA graph shape 或调度状态。
4. **Case B 是 noisy long-prefill 辅助证据。** 可分析，但所有 cross-framework claim 都要带 confidence ceiling M。
5. **Case D 是 decode-heavy sanity check。** residual gap 只有 1.09x，优先级最低。

## 7. Caveats

| Caveat | Impact |
|---|---|
| SGLang FlashInfer vs vLLM FlashAttention v3 | attention-kernel 相关结论 confidence ceiling M |
| Case B 双框架 bimodal | Case B cross-framework 结论 ceiling M |
| Case B 缺 graph-on EXTEND formal | prefill timing 需谨慎，依赖 graph-off mapping |
| Phase 4 尚未完成 | 目前不能给最终 root cause 或优化建议 |

## 8. Next Steps

Phase 4 将读取 Phase 3 traces，优先顺序建议为：

1. Case A EXTEND/DECODE triage
2. Case C EXTEND/DECODE triage
3. Case B with caveat
4. Case D sanity check

Phase 4 需要输出 kernel table、overlap opportunity、fuse opportunity、vLLM cross-check、hypotheses 和 ranked recommendations。

> Placeholder: Phase 4 triage pending.
> Placeholder: Phase 5 validation pending.
