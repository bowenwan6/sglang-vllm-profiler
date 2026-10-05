# S1 runbook — server sessions

Operational reference for the sessions of [`PLAN.md`](PLAN.md) §4. Node rules and the access procedure
are in the workspace's `context/radix.md`; nothing here repeats them. Each session starts only after
Bowen has approved its plan.

## Session P1 — PR-A on real tasks (approved 2026-10-05)

What runs: `scripts/p1_node.sh` on the node (environment, step 0, then `scripts/p1_bench.py` for T1–T4).
Budget: one 1 h assignment plus one 1 h extension; nominal ≈ 70 min of node time.

| Step | Where | Command / content |
|---|---|---|
| 1. assign | Mac | `radix machines available`; `radix assign <machine> --gpus 1 --duration 1h --json < /dev/null`; write the returned host keys to `~/.radix/known_hosts.d/<ip>` |
| 2. copy | Mac | `tools/radix/setup_node.sh` → `~/sgl/`; `scripts/p1_node.sh`, `scripts/p1_bench.py` → `~/sgl/p1/` |
| 3. start | node, tmux `sgl-install:p1` | `EXPECT_SHA=<head of feat/bench-goodput> DEADLINE_EPOCH=<assignment end − 10 min> bash ~/sgl/p1/p1_node.sh` |
| 4. extend | Mac | `radix extend --by 1h` once step 0 has passed (the tasks do not fit the first hour). `DEADLINE_EPOCH` given at start already assumes this extension; if the extension is refused, `touch ~/sgl/logs/p1/STOP` before the first hour ends |
| 5. watch | Mac | `ssh … 'tail -5 ~/sgl/logs/p1/progress.log'`; stop cleanly with `touch ~/sgl/logs/p1/STOP` |
| 6. sync | Mac | `NODE=… KNOWN_HOSTS=… bash scripts/p1_sync.sh` during the run and at the end |
| 7. release | Mac | after the final sync lists everything: `radix release`, then `radix whoami` for the credit count |

`p1_node.sh` settings: `SGLANG_REF=origin/feat/bench-goodput`, `FLASHINFER_VER=0.7.0.post1`,
`MODEL=Qwen/Qwen3-8B`. Step 0 stops the session if the goodput unit file fails in the real
environment or the installed commit is not `EXPECT_SHA`.

Cells (`p1_bench.py`), all with `--output-details` and `--goodput`:

| Task | Server | Client |
|---|---|---|
| T1 probe | defaults | ShareGPT, closed loop, concurrency 256, 1500 prompts → capacity c0 (req/s) |
| T1 sweep | defaults | ShareGPT at {0.5, 0.8, 1.0, 1.25, 1.6, 2.0} × c0, 50 s of arrivals each; the best-goodput rate twice more with other seeds |
| T2 | `--max-running-requests` 32 / 128 / 512 | ShareGPT at 1.25 × c0 |
| T3 | `--schedule-policy` fcfs / hrrn | two clients at once: short (random 256-token prompts, 128 out, 0.6 × c0, `ttft:1000 tpot:100`) and long (random 12k-token prompts, 256 out, 2 req/s, `e2el:30000`) |
| T4 | `--max-running-requests 128`, `SGLANG_REQ_WAITING_TIMEOUT` unset / 2 / 10 | ShareGPT at 1.5 × c0 |

SLOs for T1, T2, T4: `ttft:2000 tpot:100 e2el:20000`. Other SLO sets are evaluated offline from the
per-request details; no rerun is needed for that.

Stop conditions: step 0 fails; a cell exceeds twice its budget (`STOP` file, then inspect); less than
four minutes left before the deadline (the runner skips the remaining cells by itself).

Outputs: node `~/sgl/logs/p1/` → Mac `results/raw/p1/` (ignored). Tracked: `results/pra/summary.jsonl`,
`results/pra/step0_summary.txt`, `results/pra_usage.md`.
