#!/usr/bin/env bash
# Phase 2 chainer: B → C/D variance gate + vLLM recheck → summarize
# Run after Case A completes. All scripts exit non-zero on any failure.
set -e
cd /data/sglang-vllm-profiler

echo "[$(date -u +%H:%M:%S)] === Phase 2 BCD chain starting ==="

echo "[$(date -u +%H:%M:%S)] --- Case B shaping ---"
python3 experiments/qwen3vl8b/phase2/scripts/run_phase2_caseB.py

echo "[$(date -u +%H:%M:%S)] --- Case C/D variance gate + vLLM recheck ---"
python3 experiments/qwen3vl8b/phase2/scripts/run_phase2_variance.py

echo "[$(date -u +%H:%M:%S)] --- Summarizing Phase 2 ---"
python3 experiments/qwen3vl8b/phase2/scripts/summarize_phase2.py

echo "[$(date -u +%H:%M:%S)] === Phase 2 fully complete ==="
