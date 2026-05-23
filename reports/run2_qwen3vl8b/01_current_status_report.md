# SGLang vs vLLM Profiling 当前状态报告（run2_qwen3vl8b）

<aside>
📌

**Key findings**

1. **关键差距在 TTFT，而不是 TPOT。** 四个 workload 中，SGLang 与 vLLM 的 TPOT / decode throughput 基本 parity；主要差异集中在 first-token 之前的 prefill / dispatch / scheduler 路径。
2. **SGLang 的 TTFT 在 run2 baseline 中显著慢于 vLLM。** Phase 1 的 SGLang/vLLM TTFT p50 ratio 为：Case A **4.89x**、Case B **3.20x**、Case C **1.32x**、Case D **1.33x**。
3. **稳定性本身是关键发现。** Case C 在 W30/W100/W300 下 CV 都较高，并曾出现 misleading 的 “SGLang faster” 假象；W500 后 CV 收敛到 **2.9%**，稳定结论回到 SGLang 慢约 **1.32x**。
4. **Phase 4 的主要发现：最大 GPU 开销是共享的 GEMM，不是跨框架差异源。** SGLang 和 vLLM 都主要花在同一类 `nvjet_sm90_*` FP8 GEMM kernel 上，因此 GEMM 是 absolute-speed 问题，不是解释 SGLang-vLLM gap 的首要原因。
5. **当前最强待验证 hypothesis 是 dispatch / graph coverage 差异。** 在 graph-on formal traces 中，SGLang 仍有不少 GEMM 路径没有像 vLLM 那样稳定落入 CUDA graph / compile region；但这还不是最终结论，Phase 5 必须直接测 CPU launch / dispatch gap。

</aside>

<aside>
🐙

**GitHub**：https://github.com/bowenwan6/sglang-vllm-profiler

</aside>

## 摘要

本报告总结 run2 中 SGLang 与 vLLM 在 `Qwen/Qwen3-VL-8B-Instruct` text-only serving 上的阶段性对比结果。实验目标不是给出泛化 benchmark 排名，而是定位 SGLang 相对 vLLM 的性能差距来自哪个阶段、哪个系统路径，并为后续 trace triage 和优化假设提供证据。

当前已完成 Phase 0–4：功能等价性验证、baseline benchmark、shaping / variance gate、trace collection，以及 trace triage。主要结论是：**SGLang 的性能差距主要体现在 TTFT，而不是 TPOT**。Phase 1 baseline 中，四个 case 的 TPOT 基本相等，但 TTFT 上 SGLang 均慢于 vLLM。Phase 2 进一步表明，Case A 的一部分差距来自 overlap scheduler 的 c=1 固定开销；Case C 需要足够 warmup 才能稳定，W500 后确认其稳定 gap 为 1.32x；Case B 两边均存在 bimodal，因此只能作为带 caveat 的辅助证据。

Phase 4 已完成对 A/C/B/D 的离线 trace triage。当前最重要的分析结果是：两边 GPU time 都由同一类 `nvjet_sm90_*` FP8 GEMM 主导，说明 GEMM 是共享成本，不是跨框架 gap 的主要差异源；更强的待验证 hypothesis 是 **SGLang graph / compile coverage 不如 vLLM 充分**，从而在 first-token 前留下更多 CPU launch / dispatch 固定开销。该判断不是最终 root cause：graph-off mapping trace 只用于 kernel-to-source 映射，不能单独证明真实 serving eager；真正需要验证的是 graph-on formal traces 中观察到的 coverage 差异是否对应可量化 CPU gap。

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
| Phase 4 | Trace triage，定位 kernel / scheduler / memory 等原因 | Complete |
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
| SGLang EXTEND mapping | A/C/D complete；Case B unavailable |
| SGLang EXTEND formal | A/C/D complete |
| Case B SGLang EXTEND | Mapping gz corrupt；formal missing after repeated attempts |

Phase 3 主采集完成后，最初 SGLang traces 只捕获到 DECODE stage。随后使用 `max_new_tokens=1` 的 prefill-only load 将 profiler window 对准 EXTEND stage，补采 EXTEND/PREFILL。补采结果为 7/8 成功：A/C/D 拿到 EXTEND mapping + formal；B 拿到 EXTEND mapping，但 graph-on EXTEND formal 在 8 次尝试后仍无法捕获。

Phase 4 期间进一步审计发现，Case B 的 EXTEND mapping `.gz` 文件也不可用，重采后仍未得到可用 EXTEND trace。因此 Case B 的 prefill-stage SGLang 侧证据被降级为 unavailable。这个缺口可以接受但必须显式标注：Case B 本身 noisy 且 confidence ceiling M，Phase 4 不应基于 Case B prefill-stage 给强结论。

---

## 7. Phase 4 Trace Triage：主要发现

Phase 4 对 A/C/B/D 的 SGLang traces 与 vLLM cross-check traces 做了离线 triage，输出 kernel table、overlap/fuse opportunity、per-case observations，以及全局 hypotheses。需要强调：Phase 4 的产物是 **evidence-backed hypotheses**，不是最终优化结论。

| Case | Triage status | Main observation | Interpretation strength |
| --- | --- | --- | --- |
| A | EXTEND/DECODE + vLLM complete | graph-on formal 中 SGLang coverage 不如 vLLM graph/compile 路径充分；residual gap 1.56x | Strongest evidence for H1, pending validation |
| C | EXTEND/DECODE + vLLM complete | batched c=16 下仍观察到同类 graph/compile coverage 差异；gap 1.32x | Strong evidence for H1, pending validation |
| B | DECODE + vLLM complete；EXTEND unavailable | 双框架 bimodal；长 prefill 结论受限 | Ceiling M；deprioritize |
| D | EXTEND/DECODE + vLLM complete | decode-heavy sanity，gap 仅 1.09x | Corroborating evidence |

### Phase 4 hypotheses

| ID | Hypothesis | Gap relevance | Impact | Confidence | Phase 5 action |
| --- | --- | --- | --- | --- | --- |
| H1 | SGLang graph / compile coverage 相比 vLLM 不充分，导致额外 CPU launch / dispatch gap | Primary gap candidate | High | Medium | 测 SGLang prefill CPU launch gap；测试 CUDA graph / piecewise graph / torch.compile 覆盖是否收窄 TTFT |
| H2 | `nvjet_sm90_*` FP8 GEMM 是最大 GPU cost；PR #22392 CUTLASS FP8 可能加速 | Absolute speed, not gap closer | Medium absolute / Low gap | High for attribution | 可并行 A/B PR #22392，但不要当成 vLLM gap fix |
| H3 | FlashInfer vs FlashAttention v3 attention backend 差异 | Not primary driver | Low | Medium ceiling | 仅作为 confidence ceiling 记录 |
| H4 | Case B gap 来自 bimodality + c=1 fixed overhead | Deprioritize Case B | Low | Medium | 先解决 bimodality / trace availability，再谈 kernel claim |

Phase 4 最关键的区分是：**最大 GPU kernel 不等于最大 gap source**。SGLang 与 vLLM 都被同一类 FP8 GEMM kernel 主导，这解释了绝对 GPU time，但不能解释为什么 SGLang TTFT 更慢。当前更值得验证的是 CPU-side launch / dispatch / graph coverage：在 graph-on formal traces 中，SGLang 的 graph coverage 看起来不如 vLLM 的 CUDA graph / compile region 充分；但 GPU-time kernel table 本身不能直接量化 CPU launch gap，因此 H1 只能保持 Medium confidence。

方法学上需要特别注意：SGLang 的 two-trace workflow 包含 graph-off mapping trace 和 graph-on formal trace。**graph-off mapping trace 是有意关闭 CUDA graph 的源码映射工具，不能用来证明真实 serving 路径没有 graph。** 本报告中的 H1 只把 mapping trace 用于定位源码，把 graph-on formal trace、vLLM cross-check 和 Phase 1/2 的 TTFT scaling 共同作为 hypothesis 证据。

---

## 8. 当前解释与边界

当前报告支持以下中间结论：

1. **SGLang 的主要问题在 TTFT，不在 TPOT。**
2. **H1 是当前最值得验证的方向。** Case A/C/D 的 graph-on traces 支持 “SGLang graph/compile coverage 不如 vLLM 充分” 这一方向，但它还需要 Phase 5 的 CPU launch-gap 直接测量。
3. **GEMM 是最大 GPU 成本，但不是主要 gap 解释。** PR #22392 可能提高 SGLang 绝对性能，但由于 vLLM 也使用同一类 GEMM kernel，它不应被描述为主要 gap-closer。
4. **Case B 是 noisy long-prefill 辅助证据。** EXTEND trace unavailable 且双框架 bimodal，所有 cross-framework claim 都要带 confidence ceiling M。
5. **Case D 是 decode-heavy sanity check。** residual gap 小，说明 decode token-by-token 路径并非主要问题。

**解释边界**：Phase 4 已经给出 ranked hypotheses，但 Phase 5 尚未完成。因此本文仍不输出最终 root cause，也不把任何优化路径写成确定结论。

---

## 9. Caveats

| Caveat | Impact |
| --- | --- |
| SGLang FlashInfer vs vLLM FlashAttention v3 | attention-kernel 相关结论 confidence ceiling M |
| Case B 双框架 bimodal | Case B cross-framework 结论 ceiling M |
| Case B SGLang EXTEND unavailable | prefill-stage SGLang 侧不能给强结论 |
| graph-off mapping trace 不能证明真实 serving eager | H1 必须依赖 graph-on formal + Phase 5 CPU-gap 验证 |
| H1 尚未被 Phase 5 直接验证 | dispatch/graph hypothesis 仍是 M confidence |

---

## 10. 建议图表

| Figure | Content | Purpose |
| --- | --- | --- |
| Figure 1 | Phase 1 四个 case 的 SGLang/vLLM TTFT p50 柱状图 | 直观看出 TTFT gap |
| Figure 2 | Case A default vs `--disable-overlap-schedule` | 展示 overlap scheduler 开销 |
| Figure 3 | Case C W30/W100/W300/W500 的 TTFT p50 与 CV | 展示 W500 如何纠正 noisy conclusion |
| Figure 4 | Phase 3 trace readiness matrix | 展示 Phase 4 证据完备度 |
| Figure 5 | Phase 4 hypotheses ranking | 展示 H1/H2/H3/H4 的 impact × confidence |

---

## 11. 下一步

Phase 5 应优先验证 H1，而不是继续扩大 benchmark 数量。建议顺序：

1. **验证 H1：测 CPU launch / dispatch gap。** 在 Case A 和 Case C 的 prefill window 中测 SGLang inter-kernel CPU gap，确认 GPU kernel 之间是否存在足够解释 TTFT residual gap 的 launch/dispatch 空洞。
2. **验证 graph/compile coverage 是否收窄 gap。** 测试 SGLang CUDA graph、piecewise graph 或 torch.compile 覆盖范围变化，观察 Case A 1.56x 和 Case C 1.32x residual gap 是否下降。
3. **并行评估 H2，但分开叙述。** 如 PR #22392 可用，可以 A/B CUTLASS FP8 替代 nvjet FP8，对绝对 latency 可能有帮助；但它不是当前主要 gap hypothesis。
4. **暂不投入 Case B kernel-level 结论。** 先解决 bimodality 与 EXTEND trace availability，否则只保留 caveat。

> Placeholder: Phase 5 validation pending.
 
