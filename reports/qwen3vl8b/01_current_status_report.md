# SGLang vs vLLM Profiling 当前状态报告

<aside>
📌

**Key findings**

1. **早期 TTFT gap 观察受 instrumentation 污染,不是 clean 结论。** Phase 1 baseline 与 Phase 2 Case C W500 中,SGLang 侧开启了 KAPI logging(`SGLANG_KERNEL_API_LOGLEVEL=1`),vLLM 无对应 instrumentation;Phase 5 已证明该 logging 显著抬高 SGLang TTFT。因此早期"四 workload SGLang TTFT 全面更慢"(4.89×/3.20×/1.32×/1.33×)是 **instrumentation-confounded exploratory discovery signal**,不是 clean cross-framework final result。
2. **Clean Case A 验证了一个真实、可操作的 TTFT 贡献因素。** 无 KAPI / 无 profiler 的干净 benchmark 中,强制开启 prefill piecewise CUDA-graph coverage(`--enforce-piecewise-cuda-graph`)使 Case A(c=1)TTFT 显著降低(~19.2 → ~11.7 ms),**TPOT 基本不变**,0 failures,达到 vLLM TTFT 区间。`--enforce-piecewise-cuda-graph` 是 **testing lever,不是 production fix**;S2 CV ~10–12%,**不声称稳定优于/等于 vLLM**。
3. **Clean Case C 未显示 material gap,也无 Case-A-like 收益。** 旧的"稳定 1.32× batched gap"依赖 KAPI-污染的 SGLang 测量,已**撤回**。clean interleaved rerun:pooled S0 ~192.2 / S2 ~193.6 / vLLM ~189.8 ms —— 无 material median gap、无 Case-A-like S2 收益;S0/vLLM 有 ~17% session variance,故小幅效应未能解析(既非 strict parity,也非已证明的 gap)。
4. **Phase 4 trace 仍显示 shared GEMM absolute cost;机制与 production scope 需 clean follow-up。** 两框架 GPU time 主要由同类 FP8 GEMM 主导(shared absolute-cost,非已证明的 gap source);dispatch/graph/compile 机制有结构差异,值得 clean 验证。Case B/D 暂无 clean cross-framework baseline,不纳入 headline。

</aside>

<aside>
🐙

**GitHub**：https://github.com/bowenwan6/sglang-vllm-profiler

</aside>

## 摘要

本报告总结 SGLang 与 vLLM 在 `Qwen/Qwen3-VL-8B-Instruct` text-only serving 路径上的阶段性 profiling 结果。研究目标不是给出泛化 benchmark 排名，而是回答一个更工程化的问题：**SGLang 相对 vLLM 的 first-token 延迟差距来自哪个阶段、哪类系统路径，以及下一步应验证哪些优化假设**。

> ⚠️ **Methodology correction (2026-05-26):** 早期 SGLang 测量(Phase 1 baseline、Phase 2 Case C W500)开启了 SGLang-only KAPI logging,vLLM 无对应 instrumentation,显著抬高 SGLang TTFT。下文保留的早期 ratio 仅作 **provenance / exploratory**,不是 clean 结论。详见 `experiments/qwen3vl8b/methodology_correction.md`。

当前主实验已完成 Phase 0–4 并进入 Phase 5。早期 Phase 1/2 的"四 workload SGLang TTFT 全面更慢"(4.89×/3.20×/1.32×/1.33×)及 Case C"稳定 1.32× gap",**因 SGLang-only KAPI instrumentation 污染而降级为 exploratory discovery signal**,不能作为 clean cross-framework 结论。Phase 5 的 clean(无 KAPI/无 profiler)证据显示:**Case A** 强开 prefill piecewise CUDA-graph coverage 显著降 TTFT(TPOT 不变);**Case C** clean rerun 无 material gap、无 Case-A-like 收益。TPOT/throughput 近 parity 这一定性观察仍成立。

Phase 4 的主要贡献是把“哪个 GPU kernel 贡献最多绝对时间”与“什么导致跨框架 gap”区分开。trace 显示两边 GPU time 都主要消耗在同一类 `nvjet_sm90_*` FP8 GEMM kernel 上，因此 GEMM 是共享的 absolute-cost，不是解释 SGLang-vLLM gap 的首要差异源。主要 hypothesis(H1)是:SGLang 的 prefill 路径因 VLM auto-disable 而未落入 piecewise CUDA graph,留下额外 CPU launch / dispatch overhead。**Phase 5 clean Case A 干预已验证该方向**:强制开启 prefill piecewise graph 使 Case A TTFT 降低约 39%、TPOT 不变,达到 vLLM TTFT 区间。该结论限于 Case A(testing-lever、单 case、S2 CV 10–12%);Case C clean rerun **未见 Case-A-like 收益**,旧 1.32× gap 已撤回(见 methodology_correction.md)。

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
| Phase 5 | clean H1 validation | In progress | Case A clean H1 supported; Case C clean correction (no Case-A-like benefit); B/D clean baseline pending |

---

## 3. Workloads

| Case | Workload | Purpose |
| --- | --- | --- |
| A | 128 -> 128, c=1 | Short latency；最干净的 fixed-overhead case |
| B | 2048 -> 128, c=1 | Long prefill；观察长 prompt / chunk / bimodality |
| C | 512 -> 128, c=16 | Batched serving；更接近并发 serving |
| D | 512 -> 512, c=16 | Decode-heavy；用于 sanity check decode path |

---

## 4. Phase 1 Baseline (Exploratory / instrumented historical measurements)

> ⚠️ **KAPI-confounded — provenance only.** SGLang TTFT below was measured with SGLang-only KAPI
> logging; the ratios are **exploratory discovery signals**, not clean cross-framework results. See
> `methodology_correction.md`.

| Case | Workload | SGLang TTFT p50 | vLLM TTFT p50 | SGLang/vLLM (confounded) | TPOT | Note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A | 128 -> 128, c=1 | 61.8 ms | 12.6 ms | **4.89x** | parity | cleanest short-latency case |
| B | 2048 -> 128, c=1 | 66.7 ms | 20.8 ms | **3.20x** | parity | vLLM bimodal |
| C | 512 -> 128, c=16 | 247.5 ms | 187.9 ms | **1.32x** | parity | variance gate needed |
| D | 512 -> 512, c=16 | 253.0 ms | 189.7 ms | **1.33x** | parity | p99 tail / bimodal |

Phase 1 的**定性** discovery signal(confounded):TTFT 看似主要 gap、TPOT 近 parity;A→B prompt 16× 但 SGLang TTFT 仅 +4.9 ms,提示 first-token 前固定 dispatch overhead。这些**方向性观察**促成了 Phase 5,但其绝对 ratio 因 KAPI 污染**不能**作为 clean 结论。

---

## 5. Phase 2 Shaping / Variance Gate (Exploratory / instrumented historical measurements)

> ⚠️ **KAPI-confounded — provenance only.** SGLang TTFT below (incl. Case C W500) was measured with
> SGLang-only KAPI logging. The Case C **1.32× gap is SUPERSEDED** by the clean rerun (§ below / `methodology_correction.md`).

| Case | Winner Config | SGLang TTFT p50 (confounded) | vLLM Ref | Residual Gap | Phase 3 Protocol |
| --- | --- | ---: | ---: | ---: | --- |
| A | `--disable-overlap-schedule` | 19.6 ms, CV 3.2% | 12.6 ms | ~~1.56x~~ confounded | warmup 30, 3 reps |
| B | default | 30.3 ms, CV 68.4% | 21.5 ms, CV 85.9% | 1.41x (ceiling M) | warmup 300, 5 reps |
| C | default, W500 | 249.1 ms, CV 2.9% | 189.0 ms, CV 1.9% | ~~1.32x~~ **SUPERSEDED** | warmup 500, 5 reps |
| D | default, W30 | 206.2 ms, CV 3.3% | 189.7 ms | ~~1.09x~~ confounded | warmup 30, 3 reps |

**Case A.** `--disable-overlap-schedule` 将 SGLang TTFT 从 default 约 21.8 ms 降到 19.6 ms，说明 overlap scheduler 在 c=1 short-latency 场景确实有固定成本。但即使关闭该路径，仍有 1.56x residual gap，因此 Case A 是最干净的 Phase 4 / Phase 5 对象。

**Case C. ⚠️ 1.32× 已撤回(SUPERSEDED)。** W500 probe(CV 2.9%, 249.1 ms)是有效的 SGLang-internal 方差门(并纠正了 W100/W300 的 "SGLang faster / 0.79×" under-warmup artifact),但 249.1 ms 是 **KAPI-confounded**,**不构成** cross-framework gap。clean interleaved rerun 显示 SGLang ≈ vLLM ≈ 190 ms、无 material median gap(见 §11 Phase 5)。

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
| A | EXTEND/DECODE + vLLM complete | graph-on formal 中 SGLang coverage 不如 vLLM graph/compile 路径充分；residual gap 1.56x | strongest H1 evidence; clean Case A confirmation done (−39% TTFT) |
| C | EXTEND/DECODE + vLLM complete | c=16；早期 1.32x gap **SUPERSEDED**（KAPI-confounded） | clean rerun: 无 material gap、无 Case-A-like S2 收益 |
| B | DECODE + vLLM complete；EXTEND unavailable | 双框架 bimodal；长 prefill 结论受限 | ceiling M；deprioritize |
| D | EXTEND/DECODE + vLLM complete | decode-heavy sanity；gap 仅 1.09x | corroborating evidence |

### Phase 4 hypotheses

| ID | Hypothesis | Gap relevance | Impact | Confidence | Phase 5 action |
| --- | --- | --- | --- | --- | --- |
| H1 | SGLang prefill graph / compile coverage 相比 vLLM 不充分（VLM auto-disable piecewise graph），导致额外 CPU launch / dispatch overhead | primary gap candidate | High | **Strengthened (clean Case A)**；Case C pending | ✅ Case A: `--enforce-piecewise-cuda-graph` 降 TTFT ~39%（19.2→11.7ms），TPOT 不变。Next: Case A stability (reps=5) + Case C 泛化验证 |
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

## 11. Phase 5 Validation（进行中）

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

**剩余步骤:**
1. **Case B / Case D clean(无 KAPI/profiler)cross-framework baseline** —— 若要 four-workload headline 必须先补。
2. **Production-safe 设计讨论**:针对低并发 / text-only VLM 的 prefill graph enablement(Case-A locus),不用 testing lever、不宣称 global VLM fix(纯文档,本轮不改源码)。
3. **H2 并行、单独叙述。** PR #22392 / CUTLASS FP8 针对 absolute latency,不作 gap-closer。
