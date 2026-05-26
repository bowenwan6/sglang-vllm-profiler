# SGLang vs vLLM Profiling 当前状态报告

<aside>
📌

**Key findings**

1. **可操作的 TTFT 问题集中在低并发短请求 Case A。** 在 clean benchmark 中,SGLang default Case A TTFT 约为 **19.2 ms**,而 vLLM 处于 **13–14 ms** 区间;相比之下,clean Case C 中两者 median TTFT 均约为 **190 ms**,未观察到 material gap。

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

**1. Clean 现象。** 在干净 benchmark 中,**Case A(128→128, c=1)有明确 TTFT gap**:SGLang default ≈ 19.2 ms vs vLLM 13–14 ms;而 **Case C(512→128, c=16)无 material gap**:两者 median TTFT 均 ≈ 190 ms。

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

**Case A.** Phase-2 screening(instrumented)选择 `--disable-overlap-schedule` 作为 Case A baseline 配置;screening 中 default→no-overlap 约 21.8→19.6 ms 的差异属 **historical screening result**(KAPI-instrumented),除非另有 clean default-vs-no-overlap 验证,**不作为 clean root cause**。clean-validated 的发现是:在该 baseline 之上,**prefill piecewise graph coverage** 是 Case A TTFT 的可干预因素(§11)。

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
| graph-off mapping trace 不能证明真实 serving eager | H1 必须依赖 graph-on formal + Phase 5 CPU-gap 验证 |
| H1 尚未被 Phase 5 直接验证 | dispatch/graph hypothesis 仍是 Medium confidence |

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

---

## 12. Methodological Note（早期 instrumentation confound）

早期 Phase 1 baseline 与 Phase 2 Case C W500 的 SGLang 测量在 server 端开启了 KAPI logging
(`SGLANG_KERNEL_API_LOGLEVEL=1`),而 vLLM 无对应 instrumentation。Phase 5 Case A 证明该 logging 会显著
抬高 SGLang 的 eager-dispatch TTFT(clean Case A baseline 19.2 ms vs instrumented 53 ms)。因此:

- 旧的"四 workload SGLang TTFT 全面更慢"ratio(4.89× / 3.20× / 1.32× / 1.33×)与 Case C"稳定 1.32× gap"
  **是 instrumentation-confounded exploratory measurements**,仅作 provenance,**不是 clean 结论**。
- 本报告所有 clean 结论(§Key findings、§摘要、§11)均来自**无 KAPI、无 profiler** 的 benchmark。
- 原始 JSON / trace / log / scripts 全部保留不变;此 note 只修正 **conclusion strength**。
- 详细 correction 表:`experiments/qwen3vl8b/methodology_correction.md`。
- **Instrumentation policy(已写入 `plan.md`):** 任何 latency benchmark / clean validation **禁止** KAPI;
  KAPI 仅用于 crash/debug 的 targeted reproducer,且不得用于跨框架性能比较。

## 13. Historical / Exploratory Measurements

§4(Phase 1 baseline)与 §5(Phase 2 shaping)的表为 **historical / instrumentation-confounded /
provenance only**,保留以记录走过的路径;不应被读作 clean baseline。clean 数据见 §11。
