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

## Sessions P2 and P3 — PR-B (allowance from Bowen for 2026-10-06: 10 GPU-hours, 1–2 h per assignment)

What runs: `scripts/prb_node.sh` on the node (environment, the unit tests of D1, a background sweep of
the neighbouring unit files, then `scripts/prb_run.py` for the phases given). The plan of each session
is posted in chat before its node is assigned.

Builds: the node installs `bowenwan6/sglang` `exp/slo-node` (the pin `734cf3cf3b` + PR-B + the two
benchmark commits) and adds a second worktree `~/sgl/sglang-base` at `feat/bench-goodput` (the same
without PR-B). A server is started from the unpatched tree with `PYTHONPATH=~/sgl/sglang-base/python`;
the runner checks which tree each server imports before it starts it.

| Step | Where | Command / content |
|---|---|---|
| 1. assign | Mac | `radix assign <machine> --gpus 1 --duration 1h --json < /dev/null`; host keys → `~/.radix/known_hosts.d/<ip>` |
| 2. copy | Mac | `tools/radix/setup_node.sh` → `~/sgl/`; `scripts/{prb_node.sh,prb_run.py,p1_bench.py,prb_ladder.py,slo_client.py,stub_server.py}` → `~/sgl/prb/` |
| 3. start | node, tmux `sgl-install:prb` | `RUN_NAME=p2 EXPECT_SHA=… BASE_SHA=… DEADLINE_EPOCH=… PHASES=… bash ~/sgl/prb/prb_node.sh` |
| 4. extend | Mac | `radix extend --by 1h` once the environment is up, if the phases do not fit the first hour |
| 5. watch, sync | Mac | `tail ~/sgl/logs/<run>/progress.log`; `RUN_NAME=… NODE=… KNOWN_HOSTS=… bash scripts/prb_sync.sh`; stop with `touch ~/sgl/logs/<run>/STOP` |
| 6. release | Mac | after the final sync: `radix release`, `radix credits` |

Phases of `prb_run.py` (model `Qwen/Qwen3-8B`; every client request is greedy with `ignore_eos`):

| Phase | Server | What it does |
|---|---|---|
| `ladder` | `--max-running-requests 1`; dummy-weight `Qwen/Qwen3-0.6B`, then the real model, then the real model with `SGLANG_REQ_WAITING_TIMEOUT=2`, then the unpatched build | `prb_ladder.py`: a long request holds the slot; bounded requests behind it must be refused after their bound on all three endpoints, streaming and not; bad values get a 4xx; loose and unbounded requests are served; the unpatched build ignores the field |
| `calib` | `--max-running-requests 128` | closed-loop capacity of each class alone: c_chat (256-token prompt, 128 out), c_batch (1024, 256) |
| `check` | same | `slo_client.py` against `bench_serving --goodput` at 0.5 × c_chat, same request shape |
| `pilot` | cap 128 + `--enable-priority-scheduling --disable-priority-preemption` | U1 with one seed: `none`, `per_request`, `global_1.5` |
| `u3` | cap 128, a fresh process per arm | 0.8 of capacity for 90 s: unpatched twice, patched without the field, patched with a 3600 s bound |
| `u4` | cap 128 | chat alone at 1.3 × c_chat: the bound as a field against the bound as the global knob |
| `t4rep`, `t1rep` | as in P1 | PR-A: P1's T4 three times with the fixed benchmark; two more seeds at three T1 points |
| `u1` | as `pilot` | all arms of U1, two or three seeds |
| `u2` | cap 128, default order | the FCFS case |

P2 = `ladder,calib,check,pilot,u3,u4,t4rep,t1rep` (≈ 75 min of node time, one extension).
P3 = `u1,u2` with `C_CHAT` and `C_BATCH` taken from P2's `capacity.json` (≈ 95 min, one extension).

Stop conditions: the environment or the build check fails; the ladder fails on the real model (the
later phases would measure a broken patch); a cell runs past twice its budget. A failing unit file does
not stop the session — the ladder is the functional check — but is fixed before any PR is opened.

`scripts/stub_server.py` stands in for the server on the Mac: `prb_run.py --dry-run` drives every
phase against it (control flow only).
