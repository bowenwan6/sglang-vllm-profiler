# SGLang vs vLLM Profiling 当前状态报告

<aside>
📌

**Key findings**

1. **可操作的 TTFT 问题集中在低并发短请求 Case A。** 在 clean benchmark 中,SGLang Case A(selected baseline `--disable-overlap-schedule`)TTFT 约为 **19.2 ms**,而 vLLM 处于 **13–14 ms** 区间;相比之下,clean Case C 中两者 median TTFT 均约为 **190 ms**,未观察到 material gap。

2. **Case A 的关键瓶颈是 VLM prefill 路径的 graph coverage 不足。** GPU profiling 显示两框架主要消耗在同类 FP8 GEMM kernels 上,说明差异并非来自 SGLang 特有的慢 GEMM。配置与源码审计进一步发现,SGLang 对 Qwen3-VL 默认关闭 prefill piecewise CUDA graph(VLM auto-disable)。

3. **受控干预验证了 Case A 的问题来源。** 强制启用 prefill piecewise CUDA graph 后,SGLang Case A TTFT 从约 **19.2 ms** 降至 **11.7–13.4 ms**,TPOT 基本不变,进入 vLLM TTFT 区间。该结果证明 graph coverage 是 Case A TTFT 的重要可干预因素。

4. **该优化不应直接推广到 batched workload。** 在 clean Case C(`c=16`)中,强制启用 piecewise graph 未产生 Case-A-like median TTFT 改善;当前证据支持面向低并发、text-only、shape 稳定请求的选择性策略,而不是对全部 VLM 全局强制开启。

*(方法学说明:早期 Phase 1/2 SGLang 测量带 SGLang-only KAPI logging,为 instrumentation-confounded exploratory provenance,不作为 clean 结论 —— 见文末 Methodological Note 与 `experiments/qwen3vl8b/methodology_correction.md`。)*

</aside>

<aside>
🐙

**GitHub**：https://github.com/bowenwan6/sglang-vllm-profiler

</aside>

## 摘要

本报告研究 SGLang 与 vLLM 在 `Qwen/Qwen3-VL-8B-Instruct` text-only serving 上的 first-token latency:目标是**定位 first-token latency 的工程瓶颈并验证可干预的优化点**,而非给出泛化 benchmark 排名。结论以 clean(无 instrumentation)benchmark 为准,叙事按"现象 → 分析 → 机制 → 因果验证 → 边界"展开。

**1. Clean 现象。** 在干净 benchmark 中,**Case A(128→128, c=1)有明确 TTFT gap**:SGLang selected baseline(`--disable-overlap-schedule`)≈ 19.2 ms vs vLLM 13–14 ms;而 **Case C(512→128, c=16)无 material gap**:两者 median TTFT 均 ≈ 190 ms。

**2. 分析。** GPU traces 显示两框架的 GPU 时间都主要消耗在**同一类 `nvjet_sm90_*` FP8 GEMM kernel**(72–86%)。因此差异**不是** SGLang 特有的慢 GEMM —— GEMM 是 shared absolute-cost,不是跨框架差异源。

**3. 机制发现。** 配置与 SGLang 源码审计显示:对 Qwen3-VL(multimodal)**SGLang 默认关闭 prefill piecewise CUDA graph**(VLM auto-disable;decode graph 名义开启、torch.compile 关闭),使 prefill 走 eager dispatch。

**4. 因果验证。** 在 clean benchmark 中强制开启 prefill piecewise CUDA graph(`--enforce-piecewise-cuda-graph`)后,**Case A TTFT 从 ≈19.2 ms 降至 11.7–13.4 ms,TPOT 基本不变**,进入 vLLM TTFT 区间。这证明 graph coverage 是 Case A first-token latency 的重要可干预因素。

**5. 边界与方向。** clean **Case C 未出现 Case-A-like 收益**,说明该优化随 batch / shape 变化而不普适;production 方向应是**面向低并发、text-only、shape 稳定请求的 selective enablement**,而非对全部 VLM 全局强开。`--enforce-piecewise-cuda-graph` 是 testing lever,production-safe 策略待设计;Case B/D 暂无 clean cross-framework baseline。

**方法学说明(置于后文):** 早期 Phase 1/2 SGLang 测量带 SGLang-only KAPI logging,会抬高 SGLang TTFT,故旧的"四 workload ratio"与 Case C"1.32× gap"仅作 instrumentation-confounded exploratory provenance,不作为 clean 结论。详见 §Methodological Note 与 `experiments/qwen3vl8b/methodology_correction.md`。

---

## 1. 实验范围与环境

本文只讨论当前主实验结果；历史实验用于内部排查，不纳入本文的数值比较或结论推导。

| Item | Value |
| --- | --- |
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
| Phase 5 | clean H1 validation | Complete (scoped A/C) | Case A clean validated; Case C boundary (no Case-A-like benefit); production impl = future work |

---

## 3. Workloads

| Case | Workload | Purpose |
| --- | --- | --- |
| A | 128 -> 128, c=1 | Short latency；最干净的 fixed-overhead case |
| B | 2048 -> 128, c=1 | Long prefill；观察长 prompt / chunk / bimodality |
| C | 512 -> 128, c=16 | Batched serving；更接近并发 serving |
| D | 512 -> 512, c=16 | Decode-heavy；用于 sanity check decode path |

---

## 4. Clean Results

clean(无 instrumentation)benchmark 是本报告的结论依据。完整 clean 表见 §11;此处给出主线汇总。

**Case A — baseline + intervention(clean, GPU 6/0, 0 failures):**

| Variant | TTFT p50 median | TPOT p50 |
|---|---:|---:|
| SGLang selected baseline (`--disable-overlap-schedule`) | **19.2 ms** | ~5.5 ms |
| SGLang + `--enforce-piecewise-cuda-graph` | **11.7–13.4 ms** | ~5.5 ms |
| vLLM (clean anchor) | 13–14 ms | ~5.3 ms |

→ 强制 prefill piecewise graph 使 Case A TTFT 进入 vLLM 区间,TPOT 不变 → graph coverage 是 Case A first-token latency 的 **validated contributor**。(S2 stability reps=5:median 13.36 ms,CV 12.3% → 只写 "reaches the vLLM TTFT range",不声称稳定优于 vLLM。)

**Case C — boundary test(clean, c=16, 0 failures):** pooled S0 ≈ **192.2 ms**、pooled S2 ≈ **193.6 ms**、vLLM ≈ **189.8 ms** → **无 material TTFT gap、无 Case-A-like median improvement**。

> 早期 Phase 1/2 的 instrumented baseline 表(及其旧 ratio)见 §Side Quests / Methodological Notes,仅作 exploratory provenance。

---

## 5. Workloads & Clean Validation Scope

| Case | Workload | clean validation 状态 |
|---|---|---|
| A | 128→128, c=1 | ✅ clean baseline + intervention + stability(主结论) |
| C | 512→128, c=16 | ✅ clean interleaved boundary test |
| B | 2048→128, c=1 | 仅 Phase-4 结构观察;无 clean cross-framework headline(见 Side Quests) |
| D | 512→512, c=16 | 仅 Phase-4 sanity;无独立 clean baseline |

SGLang Case A baseline 是 **selected baseline `--disable-overlap-schedule`**(Phase-2 screening 选定),不是 `default`。

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
| A | EXTEND/DECODE + vLLM complete | graph-on formal 中 SGLang coverage 不如 vLLM graph/compile 路径充分；residual gap 1.56x | strongest H1 evidence; clean Case A confirmation done (−39% TTFT) |
| C | EXTEND/DECODE + vLLM complete | c=16；早期 1.32x gap **SUPERSEDED**（KAPI-confounded） | clean rerun: 无 material gap、无 Case-A-like S2 收益 |
| B | DECODE + vLLM complete；EXTEND unavailable | 双框架 bimodal；长 prefill 结论受限 | ceiling M；deprioritize |
| D | EXTEND/DECODE + vLLM complete | decode-heavy sanity；gap 仅 1.09x | corroborating evidence |

### Phase 4 hypotheses

| ID | Hypothesis | Gap relevance | Impact | Confidence | Phase 5 action |
| --- | --- | --- | --- | --- | --- |
| H1 | SGLang prefill graph coverage 不足（VLM auto-disable piecewise graph）→ 额外 CPU launch / dispatch overhead（Case A） | Case-A TTFT contributor | High | **Clean-supported for Case A only** | ✅ Case A clean: `--enforce-piecewise-cuda-graph` 降 TTFT ~39%（19.2→~12ms），TPOT 不变（reps=5 stability 已完成）。Case C clean rerun **无 Case-A-like 收益**。Next: production-safe selective-enablement scope（非全局 fix） |
| H2 | `nvjet_sm90_*` FP8 GEMM 是最大 GPU cost；PR #22392 CUTLASS FP8 可能加速 | absolute speed, not gap closer | Medium absolute / Low gap | High for attribution | 可并行 A/B PR #22392，但不要当成 vLLM gap fix |
| H3 | FlashInfer vs FlashAttention v3 attention backend 差异 | not primary driver | Low | Medium ceiling | 仅作为 confidence ceiling 记录 |
| H4 | Case B gap 来自 bimodality + c=1 fixed overhead | deprioritize Case B | Low | Medium | 先解决 bimodality / trace availability，再谈 kernel claim |

Phase 4 最重要的结构性判断是：**最大 GPU kernel 不等于最大 gap source**。SGLang 与 vLLM 都被同一类 FP8 GEMM kernel 主导，这解释了绝对 GPU time，但不能解释为什么 SGLang TTFT 更慢。当前更值得验证的是 CPU-side launch / dispatch / graph coverage：在 graph-on formal traces 中，SGLang 的 graph coverage 看起来不如 vLLM 的 CUDA graph / compile region 充分；但 GPU-time kernel table 不能直接量化 CPU launch gap，因此 H1 只能保持 Medium confidence。

---

## 8. 当前解释边界

当前报告可以支持以下中间结论：

1. **SGLang 的主要问题在 TTFT，不在 TPOT。**
2. **H1 在 clean Case A 被验证(strengthened),但不延伸到 Case C。** Case A:强开 prefill piecewise graph 使 TTFT 降 ~39%、TPOT 不变、0 errors,达 vLLM 区间。**边界:** testing lever(非 production fix);仅 Case A c=1;S2 CV 10–12%,稳定优于 vLLM 未确认。Case C clean rerun **未见 Case-A-like 收益**,旧 1.32× gap 已撤回。
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
| graph-off mapping trace 不能证明 serving eager | 机制定位依赖 graph-on formal + 配置/源码审计(已完成) |
| H1 clean-validated for Case A only | production generalization 未验证;Case C 无同类收益 |

---

## 10. 建议图表

主图应展示 **clean** 结果:

| Figure | Content | Purpose |
| --- | --- | --- |
| Figure 1 (main) | **Clean Case A**: S0 vs S2(`--enforce-piecewise-cuda-graph`)vs vLLM,TTFT p50 | 核心 clean finding:graph coverage 降 Case-A TTFT |
| Figure 2 (main) | **Clean Case C**: S0 vs S2 vs vLLM,TTFT p50(c=16) | boundary result:无 material gap、无 Case-A-like 收益 |
| Figure 3 (main) | **Mechanism diagram**: VLM auto-disable → prefill piecewise graph OFF → Case A eager-launch overhead | 机制定位 |
| Figure 4 (note) | (Methodological note 附图)confounded historical baseline | 仅作方法学说明,**不作主图** |

附:旧表(provenance only)

| Figure | Content | Purpose |
| --- | --- | --- |
| Figure 1 | Phase 1 四个 case 的 SGLang/vLLM TTFT p50 柱状图 | 直观看出 TTFT gap |
| Figure 2 | Case A default vs `--disable-overlap-schedule` | 展示 overlap scheduler 开销 |
| Figure 3 | Case C W30/W100/W300/W500 的 TTFT p50 与 CV | 展示 W500 如何纠正 noisy conclusion |
| Figure 4 | Phase 3 trace readiness matrix | 展示 Phase 4 证据完备度 |
| Figure 5 | Phase 4 hypotheses ranking | 展示 H1/H2/H3/H4 的 impact × confidence |

---

## 11. Phase 5 — Clean Validation Results (complete for scoped A/C)

**Case A clean confirmation 已完成(GPU 6,无 KAPI、无 profiler,0 failures)。** S0→S2→S0 bracket 稳定且复现 Phase 2 baseline:

| Variant | TTFT p50 median | CV | TPOT p50 |
|---|---:|---:|---:|
| S0_before (`--disable-overlap-schedule`) | 19.17 ms | 1.5% | ~5.5 ms |
| S2 (`+ --enforce-piecewise-cuda-graph`) | **11.68 ms** | 10.1% | ~5.5 ms |
| S0_after | 19.23 ms | 3.8% | ~5.5 ms |
| vLLM clean anchor | 13.11 ms | 3.2% | ~5.3 ms |

- S2 相对 S0 降低 TTFT 约 **39%**,TPOT 基本不变,0 errors → **H1 strengthened for clean Case A**。
- S2(11.68ms)**reached the vLLM TTFT range** in Case A,但因 S2 CV=10.1%,**relative advantage requires stability confirmation**,不声称稳定 parity/superiority。
- 第一轮 GPU-3 intervention 因 KAPI logging 污染(S1 产生 14.7GB log),**降级为 instrumented exploratory screen**,不作确认性证据。

**Case A S2 stability(reps=5)** :S2 13.36 ms(CV **12.3%**)vs vLLM 14.49 ms(CV 5.3%),0 failures。S2 稳定大幅领先 ~19 ms 的 S0 baseline,但 CV>5% → **不声称稳定 parity/superiority**(只能说 reaches the vLLM TTFT range)。

**Case C clean rerun(interleaved S0→S2→S0→S2→S0 + vLLM,无 KAPI/profiler)已完成:**

| Variant | median TTFT | CV |
|---|---:|---:|
| pooled S0 (default) | ~192.2 ms | 三 block median 跨度 17.3% |
| pooled S2 (`--enforce-piecewise-cuda-graph`) | ~193.6 ms | 4.0% / 2.9% |
| vLLM clean anchor | ~189.8 ms | 11.0% |

→ **clean Case C 无 material median gap、无 Case-A-like S2 收益**(pooled S2 ≈ pooled S0 ≈ vLLM)。S0/vLLM 有 ~17% session variance,小幅效应未解析。次要:S2 降低批量 run-to-run 方差但不改 median。**这撤回了旧的 Case C 1.32× gap。**

**Phase 5 状态: complete for scoped A/C clean validation; production implementation is future work**(见 §13)。

---

## 12. Side Quests / Methodological Notes

主线之外的三个研究过程注记(不影响上面的 clean 结论):

1. **Measurement hygiene: KAPI logging.** 早期探索阶段曾在 SGLang 侧启用 `SGLANG_KERNEL_API_LOGLEVEL=1`,
   会抬高 latency。因此旧的四-case ratios(`4.89× / 3.20× / 1.32× / 1.33×`)仅作 exploratory provenance,
   不作为 final clean evidence。Instrumentation policy 写在 `plan.md`(latency/validation 禁 KAPI;
   KAPI 仅 crash/debug)。详细 correction:`experiments/qwen3vl8b/methodology_correction.md`。

2. **Case C warmup / variance investigation.** W500 是一个重要 side quest:它识别了 batched workload 的
   warmup 敏感性与方差,并推动了后续 clean interleaved rerun。但 W500 的旧 cross-framework gap 数字不是
   最终结论;最终 clean 结论是 §11 的 Case C boundary result(无 material gap / 无 Case-A-like benefit;
   observed session variance,cause not isolated)。

3. **Case B trace limitation.** Case B 的 SGLang EXTEND trace 不可用,故未进入最终 clean headline。该缺口
   不影响 Case A 的 validated finding,也不影响 Case C 作为 boundary test 的结论。

*(历史 instrumented Phase 1/2 baseline 表保留在 `experiments/qwen3vl8b/phase{1,2}/summary.md`,带
provenance/confounded 标注,记录研究路径;不应读作 clean baseline。)*

## 13. Conclusion / Future Work

**结论:** clean 实验在 Case A 验证了一个真实、可操作的 first-token-latency 贡献因素 —— SGLang 对
Qwen3-VL 默认关闭的 prefill piecewise CUDA graph;强开后 Case A TTFT 进入 vLLM 区间且 TPOT 不变。Case C
(c=16)未见同类收益,界定了适用范围。

**Future work(均为可选、非阻塞):**
- Production-safe **selective graph enablement** 设计(低并发 / text-only / shape-stable),不用 testing lever。
- 可选的更广 clean benchmarking(如需 four-workload cross-framework headline,则补 Case B/D clean baseline)。
- 可选 H2 absolute-speed track(PR #22392 / CUTLASS FP8),与 gap 分开。
