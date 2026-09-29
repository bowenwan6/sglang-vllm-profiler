# Q3 runbook — exact commands (revised 2026-09-29 after review)

Mac drives, node executes. Nothing on the node ever holds a credential; git push happens only on the
Mac. Times are budgets for the monitor: a step at 2× its budget is stalled.

**Authorisation.** Every credit-spending step (`radix assign`, `radix extend`) is run only within
the scope Bowen gave in chat: 1 h assignments, extended one hour at a time, at most 6 GPU-hours in
total for this experiment. Anything beyond that — a fresh assignment after an expiry, more than
6 h, a change of scope — is asked first.

## 0. Before assigning (Mac)

```bash
cd ~/Documents/Projects/SGL/sglang-vllm-profiler
git checkout exp/q3-vit-graph && git pull --ff-only          # the node clones this branch from GitHub
git log -1 --oneline
```

## 1. Assign and address the node (Mac)

`radix assign` opens an interactive ssh session in the terminal it is run from; an agent's
non-interactive shell may not be able to drive that. If it fails, Bowen runs the assign line
himself and hands back the ssh target.

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
nsh "$N" 'mkdir -p ~/sgl ~/sgl/q3 ~/sgl/logs/q3 && mv ~/setup_node.sh ~/sgl/ && tmux new-session -d -s sgl-graph -n setup \
          && tmux new-window -t sgl-graph -n models && tmux new-window -t sgl-graph -n run && tmux ls'
# environment at the pinned SGLang SHA (the patch was written against it)
nsh "$N" "tmux send-keys -t sgl-graph:setup 'SGLANG_REF=89e1316eae bash ~/sgl/setup_node.sh' Enter"
# pinned 8B snapshot (17 GB) and the ShareGPT file the text control samples from, in parallel
nsh "$N" "tmux send-keys -t sgl-graph:models 'until [ -x ~/miniforge3/envs/sgl-profiler/bin/hf ]; do sleep 5; done; export HF_HOME=~/hf; HF=~/miniforge3/envs/sgl-profiler/bin/hf; \$HF download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b && \$HF download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset && echo MODEL_DONE' Enter"
# wait for the build (bounded: 15 min), then for the downloads
nsh "$N" 'for i in $(seq 1 60); do [ -e ~/sgl/.setup_done ] || [ -e ~/sgl/.setup_failed ] && break; sleep 15; done; ls -a ~/sgl | grep setup_; tail -4 ~/sgl/logs/setup_*.log'
nsh "$N" 'for i in $(seq 1 60); do tmux capture-pane -p -t sgl-graph:models | grep -q MODEL_DONE && break; sleep 10; done; tmux capture-pane -p -t sgl-graph:models | grep -c MODEL_DONE'
# experiment branch + measurement patch
nsh "$N" 'cd ~/sgl/profiler && git fetch -q origin exp/q3-vit-graph && git checkout -q exp/q3-vit-graph && git log -1 --oneline \
          && cd ~/sgl/sglang && git apply ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/patches/q3_vit_instrumentation.patch && git diff --stat'
```

If `.setup_failed` appears (or neither marker after 15 min): `nsh "$N" 'grep -n FAILED ~/sgl/logs/setup_*.log; tail -20 ~/sgl/logs/setup_*.log'`,
fix, re-run the same `setup_node.sh` line (it is idempotent).

## 3. Start the Mac-side backup loop (separate terminal, keep it running until §6)

```bash
N=bowenwan6@<ip>      # a new terminal does not inherit it
NODE=$N bash ~/Documents/Projects/SGL/sglang-vllm-profiler/experiments/qwen3vl8b/q3_vit_graph/scripts/sync_from_node.sh
```

Every 120 s it rsyncs `results/` (cells, summaries, raw), the node's setup and server logs, the
manifest and the profiler traces, and commits only the tracked summaries to `exp/q3-vit-graph`.

## 4. Run the stages in tmux window `run`

The `run` window was created before `setup_node.sh` wrote the conda block into `~/.bashrc`, so the
environment is activated explicitly; `set -o pipefail` keeps the exit code through `tee`.

```bash
RUN='source ~/miniforge3/etc/profile.d/conda.sh && conda activate sgl-profiler && set -o pipefail && cd ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/scripts && command -v python3'
nsh "$N" "tmux send-keys -t sgl-graph:run '$RUN && python3 run_q3.py check 2>&1 | tee -a ~/sgl/logs/q3/run.log' Enter"
nsh "$N" 'sleep 20; python3 -c "import json;d=json.load(open(\"/data/bowenwan6/home/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/results/preflight.json\"));print(d[\"verdict\"], d[\"hard_failures\"], d[\"warnings\"])"'
```

`check` PASS ⇒ **extend now** (the pilot alone takes 30–35 min and G1 is read at 45–55 min):

```bash
radix extend --by 1h
nsh "$N" "tmux send-keys -t sgl-graph:run 'python3 run_q3.py parity 2>&1 | tee -a ~/sgl/logs/q3/run.log && python3 run_q3.py pilot 2>&1 | tee -a ~/sgl/logs/q3/run.log' Enter"
```

Budgets: `check` ≤ 1 min · `parity` ≤ 8 min · `pilot` ≤ 35 min. Then read the gate:

```bash
nsh "$N" 'cat ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/results/gate_g1.json'
nsh "$N" 'grep -A9 "PILOT  workload" ~/sgl/logs/q3/run.log | tail -12'
```

`verdict: GO` ⇒ sweep and mixed (≈ 2.5 h; check the time left first, §5):

```bash
nsh "$N" "tmux send-keys -t sgl-graph:run 'python3 run_q3.py sweep 2>&1 | tee -a ~/sgl/logs/q3/run.log && python3 run_q3.py mixed 2>&1 | tee -a ~/sgl/logs/q3/run.log; python3 report_q3.py' Enter"
```

If `gate_g1.json` carries the D6 warning (TTFT differs between 16 and 128 output tokens), add
`--output-len 128` to the sweep line. `verdict: STOP` ⇒ skip to §6 (the pilot alone answers H2).

## 5. Monitor (Mac, every 10–15 min)

```bash
nsh "$N" 'cat ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/results/STATUS.json'     # stage, cell, blocks done/total, eta_min
nsh "$N" 'tmux capture-pane -p -t sgl-graph:run | grep -v "^$" | tail -12'
nsh "$N" 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -fc "sglang.launch_server"; date -u'
radix machines mine                                                                        # time left
```

Extend rule: extend by 1 h whenever `time left < remaining stage budget + 30 min`, within the
6 h total. Touch STOP at least 30 min before an expiry you cannot extend past.

Stop rules (any one): a stage at 2× budget · `STATUS.json` unchanged for 15 min while a server is
running · two consecutive failed cells · G1 STOP · < 30 min left with no extension available.

```bash
nsh "$N" 'touch ~/sgl/q3/STOP'                     # runner exits after the current block (≤ 8 min)
nsh "$N" 'tmux capture-pane -p -t sgl-graph:run | grep -v "^$" | tail -5; pgrep -fc "run_q3|sglang.launch_server"'
```

## 6. Finish (complete or stopped) — verified before anything irreversible

```bash
# 1. stop the sync loop (Ctrl-C in its terminal), then the report + manifest on the node
nsh "$N" 'cd ~/sgl/profiler/experiments/qwen3vl8b/q3_vit_graph/scripts && python3 report_q3.py && cp ~/sgl/env_manifest.txt ../results/env_manifest.txt'
# 2. final sync with traces
NODE=$N bash experiments/qwen3vl8b/q3_vit_graph/scripts/sync_from_node.sh 0
# 3. prove nothing is left behind: a dry run must list no pending files
RSH='ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
E=experiments/qwen3vl8b/q3_vit_graph/results
rsync -azn --itemize-changes -e "$RSH" "$N:sgl/profiler/$E/" "$E/" | grep -c '^[<>]' ; \
rsync -azn --itemize-changes -e "$RSH" "$N:sgl/logs/q3/" "$E/raw/logs/q3/" | grep -c '^[<>]' ; \
rsync -azn --itemize-changes -e "$RSH" "$N:sgl/traces/q3/" "$E/raw/traces/" | grep -c '^[<>]'      # all three: 0
# 4. the branch on GitHub equals the local HEAD
git -C ~/Documents/Projects/SGL/sglang-vllm-profiler fetch -q origin && \
[ "$(git -C ~/Documents/Projects/SGL/sglang-vllm-profiler rev-parse HEAD)" = "$(git -C ~/Documents/Projects/SGL/sglang-vllm-profiler rev-parse origin/exp/q3-vit-graph)" ] && echo pushed
# 5. only now
radix release
radix credits
```

Then, on the Mac: add the dated **Outcome** section to `PLAN.md` (verdicts for H1–H3, gates, what
was cut), update `plan.md` §12.6, run `/handoff` in the SGL workspace, and update the Notion page.

## 7. Recovery

- Assignment expired mid-run: home is gone. A fresh assignment needs Bowen's OK. Then redo §2;
  `git checkout exp/q3-vit-graph && git pull` inside `~/sgl/profiler` brings back the synced
  `results/cells/*.json`, and the sweep resumes at block granularity (a half-finished block is
  re-run in full, so no block mixes two assignments).
- Hotfix on the node: commit on the Mac, push, then `git -C ~/sgl/profiler stash -u && git pull --ff-only && git stash pop`
  (untracked results files would otherwise block the pull).
- Server refuses to start: `nsh "$N" 'tail -40 ~/sgl/logs/q3/<tag>_server.log'`.
- GPU not idle after a kill: `nsh "$N" 'pkill -9 -f sglang; sleep 5; nvidia-smi'`.

## Exit codes of `run_q3.py`

| code | meaning |
|---|---|
| 0 | stage complete (pilot: gate GO) |
| 2 | preflight failed |
| 3 | stopped by the STOP file after the current block |
| 4 | parity FAIL (or pilot started without a PASS) |
| 5 | gate G1 STOP |
| 6 | sweep aborted after two consecutive failed cells |
