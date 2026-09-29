# Q3 runbook — exact commands

Mac drives, node executes. Nothing on the node ever holds a credential; git push happens only on the
Mac. Times are budgets for the monitor: a step at 2× its budget is stalled.

## 0. Before assigning (Mac)

```bash
cd ~/Documents/Projects/SGL/sglang-vllm-profiler
git checkout exp/q3-vit-graph && git pull --ff-only          # the node clones this branch from GitHub
git log -1 --oneline
```

## 1. Assign and address the node (Mac) — only after the reviewer's OK

```bash
radix whoami                                                   # credits, session still valid
radix assign node-radixark-16-0001 --gpus 1 --duration 1h      # drops into ssh; type `exit` to come back
radix machines mine                                            # ssh target and time left
N=bowenwan6@<ip>                                               # copy from the line above
nsh() { ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "$@"; }
nsh "$N" 'hostname; nvidia-smi -L; df -h ~ | tail -1'
```

## 2. Bootstrap (≤ 12 min)

```bash
scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    tools/radix/setup_node.sh "${N}:setup_node.sh"
nsh "$N" 'mkdir -p ~/sgl ~/sgl/q3 && mv ~/setup_node.sh ~/sgl/ && tmux new-session -d -s sgl-graph -n setup \
          && tmux new-window -t sgl-graph -n models && tmux new-window -t sgl-graph -n run && tmux ls'
# environment at the pinned SGLang SHA (the patch was written against it)
nsh "$N" "tmux send-keys -t sgl-graph:setup 'SGLANG_REF=89e1316eae bash ~/sgl/setup_node.sh' Enter"
# pinned 8B snapshot in parallel (17 GB, ~2 min on this node)
nsh "$N" "tmux send-keys -t sgl-graph:models 'until [ -x ~/miniforge3/envs/sgl-profiler/bin/hf ]; do sleep 5; done; HF_HOME=~/hf ~/miniforge3/envs/sgl-profiler/bin/hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b && echo MODEL_DONE' Enter"
# wait for the build
nsh "$N" 'until [ -e ~/sgl/.setup_done ] || [ -e ~/sgl/.setup_failed ]; do sleep 15; done; ls -a ~/sgl | grep setup_; tail -4 ~/sgl/logs/setup_*.log'
# experiment branch + measurement patch
nsh "$N" 'cd ~/sgl/profiler && git fetch -q origin exp/q3-vit-graph && git checkout -q exp/q3-vit-graph && git log -1 --oneline \
          && cd ~/sgl/sglang && git apply ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/patches/q3_vit_instrumentation.patch && git diff --stat'
nsh "$N" 'tmux capture-pane -p -t sgl-graph:models | tail -2'   # expect MODEL_DONE
```

If `.setup_failed` appears: `nsh "$N" 'grep -n FAILED ~/sgl/logs/setup_*.log'`, fix, re-run the same
`setup_node.sh` line (it is idempotent).

## 3. Start the Mac-side backup loop (separate terminal, keep it running)

```bash
NODE=$N bash experiments/qwen3vl8b/q3_vit_graph/scripts/sync_from_node.sh        # every 120 s: rsync + commit + push
```

## 4. Run the stages in tmux window `run` (interactive bash ⇒ conda env auto-activated)

```bash
RUN='cd ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/scripts && mkdir -p ~/sgl/logs/q3'
nsh "$N" "tmux send-keys -t sgl-graph:run '$RUN && python3 run_q3.py check 2>&1 | tee -a ~/sgl/logs/q3/run.log' Enter"
nsh "$N" 'cat ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/results/preflight.json | head -30'   # verdict PASS
nsh "$N" "tmux send-keys -t sgl-graph:run 'python3 run_q3.py parity 2>&1 | tee -a ~/sgl/logs/q3/run.log && python3 run_q3.py pilot 2>&1 | tee -a ~/sgl/logs/q3/run.log' Enter"
```

Budgets: `check` ≤ 1 min · `parity` ≤ 8 min · `pilot` ≤ 25 min. Then read the gate:

```bash
nsh "$N" 'cat ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/results/gate_g1.json'
nsh "$N" 'grep -A9 "^\[.*\] PILOT" ~/sgl/logs/q3/run.log | tail -12'
```

`verdict: GO` ⇒ sweep and mixed (≈ 2.5 h; **extend the assignment first**, see §5):

```bash
nsh "$N" "tmux send-keys -t sgl-graph:run 'python3 run_q3.py sweep 2>&1 | tee -a ~/sgl/logs/q3/run.log; python3 run_q3.py mixed 2>&1 | tee -a ~/sgl/logs/q3/run.log; python3 report_q3.py' Enter"
```

`verdict: STOP` ⇒ skip to §6 (the pilot alone answers H2; write it up).

## 5. Monitor (Mac, every 10–15 min)

```bash
nsh "$N" 'cat ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/results/STATUS.json'     # stage, cell, done/total, eta_min
nsh "$N" 'tmux capture-pane -p -t sgl-graph:run | tail -15'
nsh "$N" 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -fc sglang.launch_server'
radix machines mine                                                                        # time left
```

Extend when `eta_min` exceeds the time left (1 h at a time, at most 6 h in total):

```bash
radix extend --by 1h
```

Stop rules (any one): a stage at 2× budget · `STATUS.json` unchanged for 15 min while a server is
running · two consecutive failed cells · G1 STOP · < 20 min left with no extension available.

```bash
nsh "$N" 'touch ~/sgl/q3/STOP'                     # runner exits after the current cell (≤ 8 min)
nsh "$N" 'tmux capture-pane -p -t sgl-graph:run | tail -5; pgrep -fc "run_q3|sglang.launch_server"'
```

## 6. Finish (complete or stopped) — before the assignment ends

```bash
nsh "$N" 'cd ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/scripts && python3 report_q3.py && cp ~/sgl/env_manifest.txt ../results/env_manifest.txt'
Q3_SYNC_TRACES=1 NODE=$N bash experiments/qwen3vl8b/q3_vit_graph/scripts/sync_from_node.sh 0      # one final pass incl. traces
git -C ~/Documents/Projects/SGL/sglang-vllm-profiler log -3 --oneline                             # summaries committed + pushed
radix release                                                                                    # frees the GPU, stops the clock
```

Then, on the Mac: add the dated **Outcome** section to `PLAN.md` (verdicts for H1–H3, gates, what
was cut), update `plan.md` §12.6, run `/handoff` in the SGL workspace, and update the Notion page.

## 7. Recovery

- Assignment expired mid-run: home is gone; re-assign (user's OK), redo §2, checkout the branch,
  `git pull` inside `~/sgl/profiler` brings back the synced `results/*.json`; the sweep resumes from
  the cells that exist (`raw/sweep/*.json` are not restored — the sweep re-runs missing cells).
- Server refuses to start: `nsh "$N" 'tail -40 ~/sgl/logs/q3/<tag>_server.log'`.
- GPU not idle after a kill: `nsh "$N" 'pkill -9 -f sglang; sleep 5; nvidia-smi'`.

## Exit codes of `run_q3.py`

| code | meaning |
|---|---|
| 0 | stage complete (pilot: gate GO) |
| 2 | preflight failed |
| 3 | stopped by the STOP file after the current cell |
| 4 | parity FAIL |
| 5 | gate G1 STOP |
| 6 | sweep aborted after two consecutive failed cells |
