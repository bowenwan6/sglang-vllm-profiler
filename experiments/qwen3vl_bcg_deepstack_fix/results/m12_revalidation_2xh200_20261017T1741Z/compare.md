Noise floor (pr/eager2 vs pr/eager, W1, first 8 positions): 0.0000; C1 tolerance 0.1000

## C1: graph arms against eager (W1 one at a time, W2 concurrent)

| Group | Arm | Backend | Graph batches W1 | W1 pass | W2 pass | First failures |
|---|---|---|---|---:|---:|---|
| dense, TP=1 | pr/eager2 | disabled | 0/17 | 16/16 | 19/24 | w2_b3: log-prob diff 0.2500 > 0.1000; w2_b7: log-prob diff 0.2500 > 0.1000; w2_c3: log-prob diff 0.2500 > 0.1000 |
| dense, TP=1 | pr/bcg | breakable | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5917 > 0.1000; w1_02_img720: log-prob diff 0.3616 > 0.1000; w1_03_text_again: log-prob diff 0.3750 > 0.1000 |
| dense, TP=1 | pr/full | full | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5917 > 0.1000; w1_02_img720: log-prob diff 0.3616 > 0.1000; w1_03_text_again: log-prob diff 0.3750 > 0.1000 |
| dense, TP=1 | main/default | disabled | 0/17 | 16/16 | 18/24 | w2_a3: log-prob diff 0.3747 > 0.1000; w2_a6: log-prob diff 0.2500 > 0.1000; w2_b3: log-prob diff 0.2500 > 0.1000 |
| dense, TP=1 | main/full | full | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5917 > 0.1000; w1_02_img720: tokens differ from position 1 (ref top-2 gap 0.500); w1_03_text_again: log-prob diff 0.3750 > 0.1000 |
| dense, TP=1 | allow/bcg | breakable | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5917 > 0.1000; w1_02_img720: tokens differ from position 1 (ref top-2 gap 0.500); w1_03_text_again: log-prob diff 0.3750 > 0.1000 |
| dense, chunk 2048 | pr/bcg-c2048 | breakable | 21/24 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5917 > 0.1000; w1_02_img720: log-prob diff 0.3616 > 0.1000; w1_03_text_again: log-prob diff 0.3750 > 0.1000 |
| MoE, TP=1 | pr/bcg | breakable | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5333 > 0.1000; w1_02_img720: tokens differ from position 4 (ref top-2 gap 0.125); w1_03_text_again: log-prob diff 0.5333 > 0.1000 |
| MoE, TP=1 | pr/full | full | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.5333 > 0.1000; w1_02_img720: tokens differ from position 4 (ref top-2 gap 0.125); w1_03_text_again: log-prob diff 0.5333 > 0.1000 |
| dense, TP=2 | pr/bcg-tp2 | breakable | 14/17 | 0/16 | 0/24 | w1_01_text: log-prob diff 0.3750 > 0.1000; w1_02_img720: log-prob diff 0.2477 > 0.1000; w1_03_text_again: log-prob diff 0.3750 > 0.1000 |

## Post-hoc reading (defined after the first arms; descriptive, not a pass rule)

Log probabilities come back in steps of 0.25 or less (bf16), so the C1 tolerance of 0.1 is below one
step. Text-only requests never touch DeepStack: their deviation is the backend's own numerical difference
from eager. A divergence is a near tie when both sides' top-2 gap at that position is at most 0.25.

| Group | Arm | Text max diff | Visual identical | Visual near-tie divergence | Visual other divergence | Visual max diff (aligned) |
|---|---|---:|---:|---:|---:|---:|
| dense, TP=1 | pr/eager2 | 0.375 | 30 | 1 | 0  | 0.253 |
| dense, TP=1 | pr/bcg | 0.592 | 24 | 7 | 0  | 0.500 |
| dense, TP=1 | pr/full | 0.592 | 23 | 8 | 0  | 0.500 |
| dense, TP=1 | main/default | 0.375 | 30 | 1 | 0  | 0.375 |
| dense, TP=1 | main/full | 0.592 | 2 | 0 | 29 w1_02_img720@1, w1_04_img448@1, w1_06_two_img@15, w1_07_three_img@3 | 9.610 |
| dense, TP=1 | allow/bcg | 0.592 | 2 | 0 | 29 w1_02_img720@1, w1_04_img448@0, w1_06_two_img@15, w1_07_three_img@3 | 9.610 |
| dense, chunk 2048 | pr/bcg-c2048 | 0.592 | 24 | 7 | 0  | 0.500 |
| MoE, TP=1 | pr/bcg | 1.247 | 14 | 15 | 2 w2_a5@0, w2_a7@23 | 1.125 |
| MoE, TP=1 | pr/full | 1.000 | 14 | 15 | 2 w2_a5@0, w2_a7@23 | 1.375 |
| dense, TP=2 | pr/bcg-tp2 | 0.499 | 28 | 3 | 0  | 0.500 |

**C3 positive control** (allow/bcg must fail on a visual request): detected

## C2: isolation inside each arm (W1)

| Model | Arm | w1_03_text_again = w1_01_text | w1_11_img720_again = w1_02_img720 | w1_15_video_long_again = w1_09_video_long |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | allow/bcg | same | same | same |
| Qwen3-VL-8B-Instruct | both/bcg | same | same | DIFFERENT |
| Qwen3-VL-8B-Instruct | both/bcg-c2048 | same | same | DIFFERENT |
| Qwen3-VL-8B-Instruct | both/eager | same | same | same |
| Qwen3-VL-8B-Instruct | both/eager-c2048 | same | same | same |
| Qwen3-VL-8B-Instruct | main/default | same | same | same |
| Qwen3-VL-8B-Instruct | main/full | same | same | same |
| Qwen3-VL-8B-Instruct | pr/bcg | same | same | DIFFERENT |
| Qwen3-VL-8B-Instruct | pr/bcg-c2048 | same | same | DIFFERENT |
| Qwen3-VL-8B-Instruct | pr/bcg-tp2 | same | same | same |
| Qwen3-VL-8B-Instruct | pr/eager | same | same | same |
| Qwen3-VL-8B-Instruct | pr/eager-c2048 | same | same | same |
| Qwen3-VL-8B-Instruct | pr/eager-tp2 | same | same | same |
| Qwen3-VL-8B-Instruct | pr/eager2 | same | same | same |
| Qwen3-VL-8B-Instruct | pr/full | same | same | DIFFERENT |
| Qwen3-VL-30B-A3B-Instruct | pr/bcg | same | DIFFERENT | same |
| Qwen3-VL-30B-A3B-Instruct | pr/eager | same | same | DIFFERENT |
| Qwen3-VL-30B-A3B-Instruct | pr/eager2 | same | same | DIFFERENT |
| Qwen3-VL-30B-A3B-Instruct | pr/full | same | DIFFERENT | same |

## C4: main/default against pr/eager (exact tokens)

- w1: 16/16 identical
- w2: 23/24 identical

## Speed (W4): median of per-block medians

| Arm | blocks | img720 TTFT ms | img1080 TTFT ms | two_img TTFT ms | video_long TTFT ms | text TTFT ms | conc req/s | conc median TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main/default | 3 | 77.0 | 139.7 | 65.6 | 1067.2 | 31.8 | 16.2459 | 395.6 |
| allow/bcg | 3 | 77.5 | 146.2 | 55.5 | 1063.5 | 14.3 | 16.0605 | 401.3 |
| pr/bcg | 3 | 75.2 | 145.4 | 56.2 | 1065.4 | 15.4 | 15.9984 | 402.3 |

- P-1 FAIL on text: pr/bcg 15.4 ms vs allow/bcg 14.3 ms (limit +1.0)
- **P-1** (slot cost on BCG TTFT): FAIL
- **P-2** (slot cost in memory, KV pool tokens): allow/bcg 708335, pr/bcg 708335, 0.00% fewer → pass

OVERALL: FAIL
