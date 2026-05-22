# SGLang vs vLLM Profiling 当前状态报告（run2_qwen3vl8b）

<aside>
📌

**Key findings**

1. **关键差距在 TTFT，而不是 TPOT。** 四个 workload 中，SGLang 与 vLLM 的 TPOT / decode throughput 基本 parity；主要差异集中在 first-token 之前的 prefill / dispatch / scheduler 路径。
2. **SGLang 的 TTFT 在 run2 baseline 中显著慢于 vLLM。** Phase 1 的 SGLang/vLLM TTFT p50 ratio 为：Case A **4.89x**、Case B **3.20x**、Case C **1.32x**、Case D **1.33x**。
3. **稳定性本身是关键发现。** Case C 在 W30/W100/W300 下 CV 都较高，并曾出现 misleading 的 “SGLang faster” 假象；W500 后 CV 收敛到 **2.9%**，稳定结论回到 SGLang 慢约 **1.32x**。
4. **Phase 4 优先分析 Case A 和 Case C。** Case A 是最干净的 scheduler/dispatch overhead case；Case C 是稳定的 batched c=16 TTFT gap。Case B noisy，带 confidence ceiling M；Case D gap 小，作为 sanity check。

</aside>

<aside>
🐙

**GitHub**：https://github.com/bowenwan6/sglang-vllm-profiler

</aside>

## 摘要

本报告总结 run2 中 SGLang 与 vLLM 在 `Qwen/Qwen3-VL-8B-Instruct` text-only serving 上的阶段性对比结果。实验目标不是给出泛化 benchmark 排名，而是定位 SGLang 相对 vLLM 的性能差距来自哪个阶段、哪个系统路径，并为后续 trace triage 和优化假设提供证据。

当前已完成 Phase 0–3：功能等价性验证、baseline benchmark、shaping / variance gate，以及 trace collection。主要结论是：**SGLang 的性能差距主要体现在 TTFT，而不是 TPOT**。Phase 1 baseline 中，四个 case 的 TPOT 基本相等，但 TTFT 上 SGLang 均慢于 vLLM。Phase 2 进一步表明，Case A 的一部分差距来自 overlap scheduler 的 c=1 固定开销；Case C 需要足够 warmup 才能稳定，W500 后确认其稳定 gap 为 1.32x；Case B 两边均存在 bimodal，因此只能作为带 caveat 的辅助证据。

Phase 3 已完成 trace 采集，并补齐 SGLang EXTEND/PREFILL 侧证据。当前证据已足够进入 Phase 4，但 Phase 4 尚未开始，因此本文不下最终 root cause 或优化建议。

---

## 1. 实验环境与配置

| Item | Value |
| --- | --- |
| Active run | `run2_qwen3vl8b` |
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Snapshot | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| GPU | Single H200, serialized runs |
| SGLang | `0.0.0.dev1+g0c8049d9b` |
| vLLM | `0.21.0` |
| Torch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| Dtype / TP | bf16 / TP=1 |
| Sampling | greedy: `temperature=0`, `top_p=1` |

run2 是重装机器后的重新测量。虽然模型 snapshot 与 run1 相同，但 SGLang、vLLM、torch、CUDA、FlashInfer 等版本已变化，因此 run1 只能作为 historical reference，不能与 run2 数字直接混用。

---

## 2. 方法论：Phase-gated pipeline

实验采用 phase-gated pipeline：先确认两框架可比，再建立 baseline，随后通过 shaping 和 variance gate 排除配置与噪声因素，最后采集 trace 供 Phase 4 分析。

| Phase | Purpose | Status |
| --- | --- | --- |
| Phase 0 | 验证模型、tokenizer、config 与 greedy 输出一致 | Complete |
| Phase 1 | 建立 baseline；判断 gap 来自 TTFT / TPOT / throughput | Complete |
| Phase 2 | Shaping / variance gate；锁定可 profile 的 case | Complete |
| Phase 3 | 收集 SGLang/vLLM traces，不做解释 | Complete |
| Phase 4 | Trace triage，定位 kernel / scheduler / memory 等原因 | Pending |
| Phase 5 | 验证 top hypotheses | Pending |

---

## 3. Workloads 定义

| Case | Workload | Purpose |
| --- | --- | --- |
| A | 128 -> 128, c=1 | Short latency；最干净的 fixed-overhead case |
| B | 2048 -> 128, c=1 | Long prefill；用于观察长 prompt 与 chunk/prefill 行为 |
| C | 512 -> 128, c=16 | Batched serving；更接近并发场景 |
| D | 512 -> 512, c=16 | Decode-heavy；用于 sanity check decode 路径 |

---

## 4. Phase 1 Baseline：TTFT 是主要 gap

| Case | Workload | SGLang TTFT p50 | vLLM TTFT p50 | SGLang/vLLM | TPOT | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A | 128 -> 128, c=1 | 61.8 ms | 12.6 ms | **4.89x** | parity | cleanest short-latency case |
| B | 2048 -> 128, c=1 | 66.7 ms | 20.8 ms | **3.20x** | parity | vLLM bimodal |
| C | 512 -> 128, c=16 | 247.5 ms | 187.9 ms | **1.32x** | parity | variance gate needed |
| D | 512 -> 512, c=16 | 253.0 ms | 189.7 ms | **1.33x** | parity | p99 tail / bimodal |

Phase 1 的核心结论是：**SGLang 的主要差距在 TTFT**。四个 case 的 TPOT 与 throughput 基本 parity，说明 decode token-by-token 路径没有显著落后。另一个关键信号是 A -> B：prompt 长度增加 16x，但 SGLang TTFT 只增加 4.9 ms。这说明 prefill compute 本身不是主因，更可能是 first-token 前的固定 scheduler / dispatch overhead。

---

## 5. Phase 2 Shaping / Variance Gate

| Case | Winner Config | SGLang TTFT p50 | vLLM Ref | Residual Gap | Phase 3 Protocol |
| --- | --- | ---: | ---: | ---: | --- |
| A | `--disable-overlap-schedule` | 19.6 ms, CV 3.2% | 12.6 ms | **1.56x** | warmup 30, 3 reps |
| B | default | 30.3 ms, CV 68.4% | 21.5 ms, CV 85.9% | 1.41x | warmup 300, 5 reps |
| C | default, W500 | 249.1 ms, CV 2.9% | 189.0 ms, CV 1.9% | **1.32x** | warmup 500, 5 reps |
| D | default, W30 | 206.2 ms, CV 3.3% | 189.7 ms | 1.09x | warmup 30, 3 reps |

### Case A：overlap scheduler 是真实开销来源之一

Case A 中，`--disable-overlap-schedule` 将 SGLang TTFT 从 default 约 21.8 ms 降到 19.6 ms。这说明 overlap scheduler 在 c=1 short-latency 场景有可观固定开销。该 flag 将 Phase 1 的 4.89x gap 收敛到 1.56x，但仍留下稳定 residual gap，因此 Case A 是 Phase 4 的最高优先级对象。

### Case C：W500 校正了 earlier noisy conclusion

Case C 是本轮实验中最重要的 variance correction。W30/W100/W300 下 SGLang CV 分别为 12.5% / 15.2% / 14.9%，都未通过 5% gate，并曾出现 “SGLang faster / 0.79x” 的假象。W500 5 reps 后，SGLang 稳定在 249.1 ms，CV 2.9%，确认稳定结论为 **SGLang 仍慢约 1.32x**。因此，W100/W300 的 reversal 应视为 under-warmup artifact，而非真实性能优势。

### Case B 和 D

Case B 中 SGLang 与 vLLM 都存在 bimodal，所有 cross-framework claim 必须标注 confidence ceiling M。Case D 在 W30 下已稳定，residual gap 仅 1.09x，适合作为 decode-heavy sanity check。

---

## 6. Phase 3 Trace Readiness

| Trace Group | Status |
| --- | --- |
| SGLang DECODE mapping/formal | A/B/C/D complete |
| vLLM prefill_like/decode_like | A/B/C/D complete |
| SGLang EXTEND mapping | A/B/C/D complete |
| SGLang EXTEND formal | A/C/D complete |
| Case B SGLang EXTEND formal | Missing after 8 attempts |

Phase 3 主采集完成后，最初 SGLang traces 只捕获到 DECODE stage。随后使用 `max_new_tokens=1` 的 prefill-only load 将 profiler window 对准 EXTEND stage，补采 EXTEND/PREFILL。补采结果为 7/8 成功：A/C/D 拿到 EXTEND mapping + formal；B 拿到 EXTEND mapping，但 graph-on EXTEND formal 在 8 次尝试后仍无法捕获。

这个缺口可以接受。Case B 本身 noisy 且 confidence ceiling M；同时 graph-off EXTEND mapping 已能提供 kernel -> source attribution。Phase 4 中需要明确标注：**Case B prefill-stage timing 缺少 graph-on formal trace**。

---

## 7. 当前解释与边界

当前报告支持以下中间结论：

1. **SGLang 的主要问题在 TTFT，不在 TPOT。**
2. **Case A 是最干净、最 actionable 的对象。** 关闭 overlap scheduler 后仍有 1.56x residual gap，Phase 4 应优先检查 scheduler / dispatch overhead。
3. **Case C 是稳定 batched gap。** W500 后 SGLang 仍慢约 1.32x，适合分析 batch formation、prefill/decode transition、CUDA graph shape 或调度状态。
4. **Case B 是 noisy long-prefill 辅助证据。** 可分析，但所有 cross-framework claim 都要带 confidence ceiling M。
5. **Case D 是 decode-heavy sanity check。** residual gap 小，优先级最低。

**解释边界**：Phase 4 尚未完成，因此本文不输出最终 root cause，也不提供确定性优化建议。现阶段报告只说明 benchmark 与 trace 证据已经收敛到可分析状态。

---

## 8. Caveats

| Caveat | Impact |
| --- | --- |
| SGLang FlashInfer vs vLLM FlashAttention v3 | attention-kernel 相关结论 confidence ceiling M |
| Case B 双框架 bimodal | Case B cross-framework 结论 ceiling M |
| Case B 缺 graph-on EXTEND formal | prefill timing 需谨慎，依赖 graph-off mapping |
| Phase 4 尚未完成 | 目前不能给最终 root cause 或优化建议 |

---

## 9. 建议图表

| Figure | Content | Purpose |
| --- | --- | --- |
| Figure 1 | Phase 1 四个 case 的 SGLang/vLLM TTFT p50 柱状图 | 直观看出 TTFT gap |
| Figure 2 | Case A default vs `--disable-overlap-schedule` | 展示 overlap scheduler 开销 |
| Figure 3 | Case C W30/W100/W300/W500 的 TTFT p50 与 CV | 展示 W500 如何纠正 noisy conclusion |
| Figure 4 | Phase 3 trace readiness matrix | 展示 Phase 4 证据完备度 |

---

## 10. 下一步

Phase 4 将读取 Phase 3 traces，建议 triage 优先级：

1. **Case A** EXTEND/DECODE triage（最高优先级）
2. **Case C** EXTEND/DECODE triage
3. **Case B**（带 caveat）
4. **Case D** sanity check

Phase 4 预期交付物包括 kernel table、overlap opportunity、fuse opportunity、vLLM cross-check、hypotheses 列表与 ranked recommendations。

> Placeholder: Phase 4 triage pending.

> Placeholder: Phase 5 validation pending.
 
