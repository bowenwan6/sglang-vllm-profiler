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

本报告总结 `run2_qwen3vl8b` 中 SGLang 与 vLLM 在 `Qwen/Qwen3-VL-8B-Instruct` text-only serving 路径上的阶段性结果。研究目标不是给出泛化 benchmark 排名，而是回答一个更工程化的问题：**SGLang 相对 vLLM 的 first-token 延迟差距来自哪个阶段、哪类系统路径，以及下一步应验证哪些优化假设**。

当前 run2 已完成 Phase 0–4：功能等价性验证、baseline benchmark、shaping / variance gate、trace collection 和 trace triage。核心实验事实比较稳定：TPOT 与 decode throughput 基本 parity，而 TTFT 上 SGLang 全部慢于 vLLM。Phase 1 显示四个 workload 的 TTFT ratio 分别为 4.89x、3.20x、1.32x、1.33x；Phase 2 进一步将 A/C/D 收敛为可分析 case，并把 Case C 的 noisy reversal 修正为稳定的 1.32x SGLang-slower gap。

Phase 4 的主要贡献是把“哪个 GPU kernel 慢”与“什么导致跨框架 gap”区分开。trace 显示两边 GPU time 都主要消耗在同一类 `nvjet_sm90_*` FP8 GEMM kernel 上，因此 GEMM 是共享的 absolute-speed 成本，不是解释 SGLang-vLLM gap 的首要差异源。当前最强但尚未验证的 hypothesis 是：SGLang 在 graph-on formal traces 中仍有部分关键 GEMM 路径没有像 vLLM 那样稳定落入 CUDA graph / compile region，可能留下额外 CPU launch / dispatch overhead。该判断仍需 Phase 5 直接测量 CPU launch gap 后才能升级为 root cause。

---

## 1. 实验范围与环境

run2 是机器重装后的重新测量。模型 snapshot 与 run1 一致，但框架、torch、CUDA、FlashInfer 等版本均已变化，因此 **run1 只能作为 historical reference，不能与 run2 数字直接混用**。

| Item | Value |
| --- | --- |
| Active run | `run2_qwen3vl8b` |
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Snapshot | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Hardware | Single H200, serialized runs |
| SGLang | `0.0.0.dev1+g0c8049d9b` |
| vLLM | `0.21.0` |
| Torch / CUDA | `2.11.0+cu130` / CUDA 13.0 |
| Dtype / TP | bf16 / TP=1 |
| Sampling | greedy: `temperature=0`, `top_p=1` |

---

## 2. Methodology：Phase-gated Pipeline

实验采用 gate-by-gate 的方法：先验证两框架输出可比，再看 baseline gap，随后通过 shaping 与 variance gate 排除配置噪声，最后用 traces 形成可验证假设。每一阶段只消费上一阶段通过 gate 的数据。

| Phase | Purpose | Status | Output |
| --- | --- | --- | --- |
| Phase 0 | 验证模型、tokenizer、config 与 greedy 输出一致 | Complete | Tier-A/B exact equivalence |
| Phase 1 | 建立 baseline；定位 gap 属于 TTFT / TPOT / throughput | Complete | 4-case benchmark table |
| Phase 2 | Shaping / variance gate；锁定可 profile protocol | Complete | selected cases + warmup/reps |
| Phase 3 | 收集 SGLang/vLLM traces，不解释 | Complete | mapping/formal + vLLM windows |
| Phase 4 | Trace triage；生成 hypotheses | Complete | per-case triage + ranked hypotheses |
| Phase 5 | 验证 top hypotheses | Pending | launch-gap / graph coverage validation |

---

## 3. Workloads

| Case | Workload | Purpose |
| --- | --- | --- |
| A | 128 -> 128, c=1 | Short latency；最干净的 fixed-overhead case |
| B | 2048 -> 128, c=1 | Long prefill；观察长 prompt / chunk / bimodality |
| C | 512 -> 128, c=16 | Batched serving；更接近并发 serving |
| D | 512 -> 512, c=16 | Decode-heavy；用于 sanity check decode path |

---

## 4. Phase 1 Baseline：TTFT 是主要 Gap

| Case | Workload | SGLang TTFT p50 | vLLM TTFT p50 | SGLang/vLLM | TPOT | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A | 128 -> 128, c=1 | 61.8 ms | 12.6 ms | **4.89x** | parity | cleanest short-latency case |
| B | 2048 -> 128, c=1 | 66.7 ms | 20.8 ms | **3.20x** | parity | vLLM bimodal |
| C | 512 -> 128, c=16 | 247.5 ms | 187.9 ms | **1.32x** | parity | variance gate needed |
| D | 512 -> 512, c=16 | 253.0 ms | 189.7 ms | **1.33x** | parity | p99 tail / bimodal |

Phase 1 的直接结论是：**SGLang 的主要差距在 TTFT，而不是 TPOT**。四个 case 的 decode token-by-token 成本接近，说明主问题不在 steady-state decode kernel 本身。另一个重要信号来自 A -> B：prompt length 从 128 增到 2048（16x），但 SGLang TTFT 仅增加 4.9 ms。这说明 prefill compute 不是主要解释，更像 first-token 前存在固定 scheduler / dispatch overhead。

---

## 5. Phase 2 Shaping / Variance Gate

| Case | Winner Config | SGLang TTFT p50 | vLLM Ref | Residual Gap | Phase 3 Protocol |
| --- | --- | ---: | ---: | ---: | --- |
| A | `--disable-overlap-schedule` | 19.6 ms, CV 3.2% | 12.6 ms | **1.56x** | warmup 30, 3 reps |
| B | default | 30.3 ms, CV 68.4% | 21.5 ms, CV 85.9% | 1.41x | warmup 300, 5 reps |
| C | default, W500 | 249.1 ms, CV 2.9% | 189.0 ms, CV 1.9% | **1.32x** | warmup 500, 5 reps |
| D | default, W30 | 206.2 ms, CV 3.3% | 189.7 ms | 1.09x | warmup 30, 3 reps |

**Case A.** `--disable-overlap-schedule` 将 SGLang TTFT 从 default 约 21.8 ms 降到 19.6 ms，说明 overlap scheduler 在 c=1 short-latency 场景确实有固定成本。但即使关闭该路径，仍有 1.56x residual gap，因此 Case A 是最干净的 Phase 4 / Phase 5 对象。

**Case C.** W30/W100/W300 下 SGLang CV 分别为 12.5% / 15.2% / 14.9%，曾出现 “SGLang faster / 0.79x” 的 misleading reversal。W500 5 reps 后，CV 收敛到 2.9%，稳定结论变为 SGLang 249.1 ms vs vLLM 189.0 ms，即 **SGLang 慢约 1.32x**。这说明 earlier reversal 是 under-warmup artifact，而不是性能优势。

**Case B / D.** Case B 双框架 bimodal，所有 cross-framework claim 必须带 confidence ceiling M。Case D residual gap 仅 1.09x，更适合作为 decode-heavy sanity check。

---

## 6. Phase 3 Trace Readiness

| Trace Group | Status | Use in Phase 4 |
| --- | --- | --- |
| SGLang DECODE mapping/formal | A/B/C/D complete | decode-side attribution |
| vLLM prefill_like/decode_like | A/B/C/D complete | cross-framework check |
| SGLang EXTEND mapping | A/C/D complete；Case B unavailable | prefill source attribution |
| SGLang EXTEND formal | A/C/D complete | graph-on prefill timing |
| Case B SGLang EXTEND | mapping gz corrupt；formal missing after repeated attempts | ceiling M only |

Phase 3 最初只捕获到 SGLang DECODE stage，随后使用 `max_new_tokens=1` 的 prefill-only load 补采 EXTEND/PREFILL。A/C/D 的 EXTEND mapping + formal 均可用；Case B 的 graph-on EXTEND formal 多次失败，Phase 4 审计时又发现原 EXTEND mapping `.gz` 不可用，重采仍未恢复。因此 Case B prefill-stage SGLang 证据被降级为 unavailable。

该缺口不阻塞主线：Case B 本身 noisy 且有 ceiling M；A 与 C 已覆盖最关键的 short-latency 与 batched-serving gap。

---

## 7. Phase 4 Trace Triage：从 Kernel 表到 Hypotheses

Phase 4 的目标不是直接给优化结论，而是把 Phase 3 traces 转化为可验证 hypotheses。分析采用 two-trace workflow：`graph-off mapping` 用于 kernel-to-source 映射，`graph-on formal` 用于真实 serving 形态下的 timing / overlap。**graph-off mapping trace 是有意关闭 CUDA graph 的工具，不能单独证明真实 serving 路径没有 graph。**

| Case | Triage status | Main observation | Interpretation strength |
| --- | --- | --- | --- |
| A | EXTEND/DECODE + vLLM complete | graph-on formal 中 SGLang coverage 不如 vLLM graph/compile 路径充分；residual gap 1.56x | strongest H1 evidence, pending validation |
| C | EXTEND/DECODE + vLLM complete | c=16 batched path 仍观察到类似 graph/compile coverage 差异；gap 1.32x | strong H1 evidence, pending validation |
| B | DECODE + vLLM complete；EXTEND unavailable | 双框架 bimodal；长 prefill 结论受限 | ceiling M；deprioritize |
| D | EXTEND/DECODE + vLLM complete | decode-heavy sanity；gap 仅 1.09x | corroborating evidence |

### Phase 4 hypotheses

| ID | Hypothesis | Gap relevance | Impact | Confidence | Phase 5 action |
| --- | --- | --- | --- | --- | --- |
| H1 | SGLang graph / compile coverage 相比 vLLM 不充分，导致额外 CPU launch / dispatch gap | primary gap candidate | High | Medium | 测 SGLang prefill CPU launch gap；测试 CUDA graph / piecewise graph / torch.compile 覆盖是否收窄 TTFT |
| H2 | `nvjet_sm90_*` FP8 GEMM 是最大 GPU cost；PR #22392 CUTLASS FP8 可能加速 | absolute speed, not gap closer | Medium absolute / Low gap | High for attribution | 可并行 A/B PR #22392，但不要当成 vLLM gap fix |
| H3 | FlashInfer vs FlashAttention v3 attention backend 差异 | not primary driver | Low | Medium ceiling | 仅作为 confidence ceiling 记录 |
| H4 | Case B gap 来自 bimodality + c=1 fixed overhead | deprioritize Case B | Low | Medium | 先解决 bimodality / trace availability，再谈 kernel claim |

Phase 4 最重要的结构性判断是：**最大 GPU kernel 不等于最大 gap source**。SGLang 与 vLLM 都被同一类 FP8 GEMM kernel 主导，这解释了绝对 GPU time，但不能解释为什么 SGLang TTFT 更慢。当前更值得验证的是 CPU-side launch / dispatch / graph coverage：在 graph-on formal traces 中，SGLang 的 graph coverage 看起来不如 vLLM 的 CUDA graph / compile region 充分；但 GPU-time kernel table 不能直接量化 CPU launch gap，因此 H1 只能保持 Medium confidence。

---

## 8. 当前解释边界

当前报告可以支持以下中间结论：

1. **SGLang 的主要问题在 TTFT，不在 TPOT。**
2. **H1 是当前最值得验证的方向。** Case A/C/D 的 graph-on traces 支持 “SGLang graph/compile coverage 不如 vLLM 充分” 这一方向，但仍需要 Phase 5 直接测 CPU launch gap。
3. **GEMM 是最大 GPU 成本，但不是主要 gap 解释。** PR #22392 可能提高 SGLang 绝对性能，但由于 vLLM 也使用同一类 GEMM kernel，它不应被描述为主要 gap-closer。
4. **Case B 是 noisy long-prefill 辅助证据。** EXTEND trace unavailable 且双框架 bimodal，所有 cross-framework claim 都要带 confidence ceiling M。
5. **Case D 是 decode-heavy sanity check。** residual gap 小，说明 steady-state decode path 不是主要问题。

这些仍是 Phase 4 hypotheses，不是最终 root cause。Phase 5 前，报告中不应把 H1 写成“已证明 SGLang 因 eager dispatch 慢”，只能写成“最强待验证方向”。

---

## 9. Caveats

| Caveat | Impact |
| --- | --- |
| SGLang FlashInfer vs vLLM FlashAttention v3 | attention-kernel 相关结论 confidence ceiling M |
| Case B 双框架 bimodal | Case B cross-framework 结论 ceiling M |
| Case B SGLang EXTEND unavailable | prefill-stage SGLang 侧不能给强结论 |
| graph-off mapping trace 不能证明真实 serving eager | H1 必须依赖 graph-on formal + Phase 5 CPU-gap 验证 |
| H1 尚未被 Phase 5 直接验证 | dispatch/graph hypothesis 仍是 Medium confidence |

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

## 11. 下一步：Phase 5 Validation

Phase 5 应优先验证 H1，而不是继续扩大 benchmark 数量。

1. **测 CPU launch / dispatch gap。** 在 Case A 和 Case C 的 prefill window 中测 SGLang inter-kernel CPU gap，确认 GPU kernels 之间是否存在足够解释 TTFT residual gap 的 launch / dispatch 空洞。
2. **测试 graph / compile coverage 是否收窄 gap。** 调整 SGLang CUDA graph、piecewise graph 或 torch.compile 覆盖范围，观察 Case A 1.56x 与 Case C 1.32x residual gap 是否下降。
3. **并行评估 H2，但单独叙述。** 若 PR #22392 可用，可以 A/B CUTLASS FP8 替代 nvjet FP8，对绝对 latency 可能有帮助；但它不是当前主要 gap hypothesis。
4. **暂不投入 Case B kernel-level 结论。** 先解决 bimodality 与 EXTEND trace availability，否则只保留 caveat。

> Placeholder: Phase 5 validation pending.
