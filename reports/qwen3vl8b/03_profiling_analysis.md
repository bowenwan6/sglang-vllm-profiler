# Phase 4 — Trace Triage 分析报告（qwen3vl8b）

> 状态：**Phase 4（trace triage / 解释）已完成**。本文给出 evidence-backed **hypotheses**，不是最终
> PR 方案，也不是 Phase 5 验证结论。所有"建议"均为待验证假设（confidence H/M/L）。vLLM 仅作为
> reference baseline 用于佐证/证伪，**不输出 vLLM 优化建议**。

## 0. 概览

| Item | Value |
|---|---|
| Active run | `qwen3vl8b` |
| Model | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` |
| 硬件 | 单 H200 · TP=1 · bf16 · greedy（`temperature=0, top_p=1`）|
| SGLang / vLLM | `0.0.0.dev1+g0c8049d9b` / `0.21.0` · torch 2.11.0+cu130 / CUDA 13.0 |
| 工具 | `llm-torch-profiler-analysis` 的 `triage`（三表：kernel / overlap / fuse）+ catalog lookup |
| GPU 使用 | Phase 4 triage **离线**（不占 GPU）；仅 Case B EXTEND 重采尝试用过 GPU 1 |

**一句话结论**：四个 case、两个 stage（EXTEND/DECODE）的 GPU 时间都被**同一个 `nvjet_sm90_*` FP8 GEMM
家族**主导（72–86%），而 vLLM 跑的是**同一批 GEMM** —— 所以 GEMM 是最大开销类，但**不是跨框架差异源**。
真正的差异在 **dispatch / 编译方式**：SGLang eager `aten::mm`，vLLM 走 torch.compile/inductor（prefill）
+ CUDA graph（decode）。TTFT gap 是 **first-token 固定开销**效应，不是 per-token decode 落后（Phase-1
TPOT parity + Case D 小 gap 双重佐证）。

---

## 1. 逐 case 分析

### Case A — `caseA_short`（128→128, c=1, `--disable-overlap-schedule`）· **最高优先级**

| 维度 | 结果 |
|---|---|
| EXTEND triage | ✅ 成功（two-trace：mapping graph-off + formal graph-on）|
| DECODE triage | ✅ 成功（two-trace）|
| vLLM crosscheck | ✅ 成功（prefill_like + decode_like，single-trace）|
| 最大开销类别 | **GEMM** —— EXTEND ~84% / DECODE ~75% |
| 关键 kernel/op/source | `nvjet_sm90_*` FP8 GEMM，`aten::mm`，`srt/layers/quantization/unquant.py:138 apply`；attention：FlashInfer `BatchPrefillWithPagedKVCache` @ `flashinfer_backend.py:779/893` |

- **vLLM 对照**：vLLM 跑**同一个 nvjet GEMM 家族**，但全部经 **`cudaGraphLaunch`**（graph 捕获），而
  SGLang 是 eager `aten::mm`。→ 证伪"SGLang GEMM kernel 本身更慢"。SGLang top GEMM 行 overlap 表显示
  `excl 100% / hid 0%`（完全串行、无重叠）。
- **初步 observation**：128-token 短 prefill 下每个 kernel 都很小，**per-op CPU launch 开销 + 未融合
  epilogue** 是 1.56× residual gap 的首要嫌疑（OBS-A1）。Phase-1"prompt×16 → TTFT 仅 +4.9ms"佐证
  其为 dispatch-bound 而非 compute-bound。
- **confidence / caveat**：OBS-A1 impact **H** / confidence **M**（kernel-share 表能看到 dispatch *路径*，
  但看不到 launch-gap *时间*，需 CPU-gap 测量验证）；GPU-time 表不含 `scheduler/CPU gap` 类。
- **为什么最高优先级**：gap 最干净（1.56×，关掉 overlap scheduler 后的稳定 residual，CV 3.2%），
  且嫌疑（dispatch 开销）**不依赖 attention backend**（无 ceiling M），最 actionable。

### Case C — `caseC_batched`（512→128, c=16, default）

> ⚠️ **The earlier "stable 1.32× batched gap" is SUPERSEDED** (it relied on KAPI-confounded SGLang
> measurements). The structural trace observations below remain valid, but they do **not** explain a
> cross-framework latency gap — clean Case C shows no material median gap. See `methodology_correction.md`.

| 维度 | 结果 |
|---|---|
| EXTEND / DECODE / vLLM crosscheck | ✅ 全部成功 |
| 最大开销类别 | **GEMM** —— EXTEND ~73% / DECODE **~85%**（c=16 下 decode GEMM 更大，单个 coopB kernel 占 41%）|

- **与 Case A 的相同点**：仍是 GEMM-bound、同一 nvjet 家族、同一 `unquant.py:138` site、同一 PR #22392
  catalog 命中；attention 非主因。
- **与 Case A 的不同点**：(1) batched c=16 出现 **radix-cache / allocator 内存管理 kernel**
  （`allocator.py:159 free`、`radix_cache.py:360 match_prefix`、`:440 cache_finished_req`，~3.3%），
  但 overlap 表标 `low-roi-hidden`（86–96% 已被 compute 重叠），**非 gap 源**；(2) vLLM 这里更明确：
  prefill GEMM 在 **torch.compile / inductor AOT 编译区**（`inductor_cache/…call`），decode 经
  `cudaGraphLaunch`。
- **结构观察(仍有效)**：两框架 GEMM 分类几乎相同,差异在 GEMM 的**派发/编译方式**(SGLang eager vs
  vLLM compiled/graphed)。这是一个**结构差异**,值得验证 —— 但 Phase 5 clean Case C 显示该差异在 c=16
  **不**转化为 TTFT 收益(强开 piecewise graph 无效),所以它**不**解释一个 batched latency gap。
- **代表性(撤回)**：~~W500 后稳定 1.32× 是干净可信的 batched gap~~ —— **SUPERSEDED**:249.1 ms 是
  KAPI-confounded;clean rerun 显示 SGLang ≈ vLLM ≈ 190 ms,**无 material median gap**。
- **caveats**：attention ceiling M（FlashInfer vs FA3）；`scheduler/CPU gap` 不在 GPU-time 表;
  Case C c=16 有 ~17% session variance。

### Case B — `caseB_longprefill`（2048→128, c=1, default）· **仅作 ceiling M 辅助证据**

| 维度 | 结果 |
|---|---|
| DECODE triage | ✅ 成功（two-trace）|
| vLLM crosscheck | ✅ 成功（prefill_like 捕获到真实长 prefill + decode_like）|
| EXTEND triage | ❌ **不可用** |

- **EXTEND 不可用 / 重采失败的事实**：原 graph-on formal 早先 8 次采集失败；graph-off mapping 的 .gz
  **损坏**（`EOFError`）。Phase 4 中按指示在 GPU 1 重采 3 次：① 截断的 EXTEND、② 有效但 DECODE-stage、
  ③ 加 `--disable-radix-cache` 仍只得 DECODE。**根因**：2048-token prompt 被 prefix-cache 命中
  （`#new-token:1`），`--profile-by-stage` 的窗口落不到真实长 prefill，且大 trace 在 shutdown flush 时
  截断 —— 与原 8 次同源，属该 profiler 机制对长-prefill 的限制，修复需改 profiler/源码（超出 Phase 4）。
  失败产物已隔离保留（`CORRUPT_/TRUNC_/DECODEONLY_`，未删除）。
- **DECODE 结果**：nvjet FP8 78.3%（PR #22392），attention 12.9%（c=1 长 KV 的 decode attention 占比偏高）。
- **关键交叉发现（OBS-B1）**：vLLM 的 **2048 长 prefill GEMM 是 eager `aten::mm`，不是 graph/compiled**
  （与 Case A/C 不同）→ "graph 覆盖差异"**不是 Case B 的 gap 主因**；Case B gap 更可能是 **bimodality
  + c=1 固定开销**。
- **为什么只作 ceiling M 辅助**：Phase-2 两框架都 bimodal（CV 68%/86%），且 SGLang EXTEND trace 缺失 →
  所有 Case B 跨框架结论 **≤ M**。
- **结论边界（不要过度解释）**：Case B 不支持 high-confidence 的 kernel 级 claim；prefill-stage 仅有
  vLLM 侧证据，SGLang prefill timing 缺失。

### Case D — `caseD_decode`（512→512, c=16, default）· **decode-heavy sanity check**

| 维度 | 结果 |
|---|---|
| EXTEND / DECODE / vLLM crosscheck | ✅ 全部成功 |
| 最大开销类别 | **GEMM** —— EXTEND ~72% / DECODE ~85%（结构与 Case C 一致）|

- **为什么是 sanity check**：512-input c=16 的 kernel 结构与 Case C 完全同构；唯一区别是 output 512（更
  decode-heavy），residual gap 最小（**1.09×**，CV 3.3%）。
- **gap 小意味着什么**：512-token 长 decode 把 **first-token 固定开销摊薄**到很多 decode step 上 →
  相对 gap 收缩。这正是 OBS-D1：**佐证 gap 是 first-token/dispatch 固定开销效应，而非 per-token decode
  落后**（TPOT parity）。
- **对 H1/H2/H3 的支持/反证**：
  - **支持 H1**：dispatch 差异（eager vs graph/compiled）依旧存在；gap 随 decode 拉长而收缩，符合
    "固定开销"模型。
  - **支持 H2**：GEMM 仍主导且两框架共担。
  - **对 H3 中性**：attention 占比小，非主因（ceiling M）。
- **结论**：Case D 不引入新瓶颈，sanity check 通过。

---

## 2. 全局 hypotheses

| ID | 假设 | 是否解释跨框架 gap | impact | confidence | fairness 依赖 |
|---|---|---|---|---|---|
| **H1** | SGLang eager `aten::mm` dispatch vs vLLM torch.compile / CUDA graph | **是（主因候选）** | **H** | **M** | 否 |
| **H2** | nvjet FP8 GEMM 主导 GPU 时间；PR #22392 / CUTLASS-FP8 是**绝对加速**机会 | 否（仅绝对加速）| M（绝对）/ L（gap）| H（归因）/ L（gap）| 否 |
| **H3** | FlashInfer vs FA3 attention backend | 否 | L | M（capped）| **是** |
| **H4** | Case B long-prefill gap = bimodality + c=1 固定开销，非 graph 覆盖 | n/a（去优先级）| L | M | 否 |

- **H1（最重要）**：相同 nvjet GEMM 在 SGLang 是 eager（`unquant.py:138 apply` → `aten::mm`），在 vLLM
  是 compiled（inductor AOT，prefill）/ graphed（`cudaGraphLaunch`，decode）。短/批 prefill 下 per-op
  launch + 未融合 epilogue 是 Case A residual 的最佳解释(**clean Case A 已验证**);Case C 的旧 1.32× 已撤回(KAPI-confounded),clean Case C 无 material gap。**不依赖 attention backend**。
  confidence 只给 **M**，因 kernel-share 表只证 dispatch *路径*、未直接测 launch-gap *时间*。
- **H2**：nvjet FP8 GEMM 占 72–86%，是 SGLang 开放 PR #22392（CUTLASS scaled-MM 替换 nvjet，去 memset
  气泡/拷贝）的 Confirmed catalog 命中。但 **vLLM 也付同样代价** → 它是绝对加速、**不是 gap-closer**。
- **H3**：SGLang FlashInfer vs vLLM FA3，share 接近（4–13%），非主因；任何 attention 级结论带 **ceiling M**
  （fairness-dependent）。
- **H4**：vLLM 长 prefill 也是 eager，所以 graph 覆盖对 Case B prefill 不适用；Case B bimodal + EXTEND 缺
  失，不适合做 high-confidence kernel claim。

---

## 3. Phase 5 建议

| 优先级 | 假设 | Phase 5 动作 |
|---|---|---|
| 1 | **H1** | 在 Case A、C 上测 SGLang prefill 的 **CPU launch-gap**（GPU-time 表看不到的 `scheduler/CPU gap`），再测开启/扩展 SGLang CUDA-graph / piecewise-graph / torch.compile 覆盖能否收窄实测 TTFT gap |
| 2 | **H2** | 若 PR #22392 可合入，A/B CUTLASS-FP8 路径做**绝对加速**，与 gap 问题分开追踪 |
| 3（低）| **H3 / H4** | 暂不投入 kernel 级精力 —— H3 是 fairness-ceilinged（M）backend 差异，H4 是 bimodal + 无 SGLang EXTEND trace。除非后续证据改变 |

- **先验 H1**（最高杠杆、fairness-independent）。
- **H2 作并行绝对性能优化线**（不期待它关 gap）。
- **H3/H4 暂缓**。

---

## 4. Caveats（贯穿全程）

- **Case B SGLang EXTEND 不可用**：所有 Case B 结论 **≤ ceiling M**；prefill-stage 仅 vLLM 侧证据。
- **Attention backend mismatch**：FlashInfer 0.6.11 vs FlashAttention v3 → 任何 attention 级结论 ceiling M。
- **Phase 5 进展**：**H1 已在 clean Case A 中被 strengthened** —— 强制 prefill piecewise CUDA-graph coverage(`--enforce-piecewise-cuda-graph`,绕过 VLM auto-disable)使 Case A TTFT 降低约 39%(19.2→11.7ms)、TPOT 不变、0 errors,达到 vLLM TTFT 区间。**边界:** testing-lever(非 production fix)、单 case(A,c=1)、S2 CV 10.1%(稳定 parity 未确认);**Case C clean rerun 完成:无 Case-A-like 收益、旧 1.32× gap 撤回**(H1 不延伸到 c=16);Case B/D 无 clean baseline;H2 仍未验证。第一轮 GPU-3 intervention 因 KAPI logging 污染降级为 exploratory。详见 `experiments/qwen3vl8b/phase5/caseA_h1_confirmation/summary.md`。
- **每个 (framework, stage, case) 单一代表性 trace**（非多 reps）：share 稳定，但绝对时间是单窗口快照。

---

## 5. Artifacts

每个 case（`analysis/qwen3vl8b/{caseA_short,caseC_batched,caseB_longprefill,caseD_decode}/`）：

- `extend_triage.md` — SGLang EXTEND triage（Case B 为"不可用"说明 + 重采全过程）
- `decode_triage.md` — SGLang DECODE triage
- `breakdown.md` — category breakdown（按 `analysis/category_regex.md` 的 9 类）
- `vllm_crosscheck.md` — vLLM prefill/decode 佐证-证伪
- `preliminary_observations.md` — 逐 case 初步 observation（带 schema 字段）
- `*_raw.txt` — triage 脚本原始输出（可审计）

全局：

- `analysis/qwen3vl8b/hypotheses.md` — H1–H4 完整 schema
- `analysis/qwen3vl8b/ranked_recommendations.md` — 排序 + Phase 5 动作
- `analysis/category_regex.md` — 9 类共享分类定义
- `reports/qwen3vl8b/03_profiling_analysis.md` — 本报告

---

> 下一步：Phase 5 验证 H1（最高优先）。在此之前，H1/H2/H3/H4 均为 evidence-backed **hypotheses**，
> 未经验证，不作为最终优化结论。
