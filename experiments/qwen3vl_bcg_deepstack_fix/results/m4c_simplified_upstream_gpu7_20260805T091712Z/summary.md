> **Correction (2026-09-03) — model label.** The generated line below opens
> "Qwen3.5-4B" because the runner inherited its banner string from the
> Qwen3.5 sub-track it was forked from. **This attempt actually ran
> `Qwen/Qwen3-VL-8B-Instruct`** (`metadata.json:model_id`,
> `deepstack_visual_indexes = [8, 16, 24]`, width 12288). The verdict and all
> numbers are unaffected; only the model name in the prose is wrong. This
> distinction is load-bearing — Qwen3.5 ships an empty DeepStack index list and
> cannot exercise this path at all (verdict `NOT_APPLICABLE_QWEN35`), which is
> precisely why the reproduction had to move to Qwen3-VL.

Qwen3.5-4B BCG DeepStack attempt m4c_simplified_upstream_gpu7_20260805T091712Z: verdict PASS_BCG_CORRECT. bcg_normal matches eager_normal AND diverges from bcg_zero (so DeepStack was retained under BCG). PASS_BCG_CORRECT.
