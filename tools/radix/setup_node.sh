#!/usr/bin/env bash
# setup_node.sh — build the sgl-profiler environment on a RADIX community node.
#
# Idempotent: every phase checks for its own result and skips what is present.
#   bash setup_node.sh                 # conda env repos sglang flashinfer verify manifest
#   bash setup_node.sh models          # download $MODELS into $HF_HOME (run in its own tmux window)
#   bash setup_node.sh sglang verify   # re-run selected phases
#
# Why conda: the node ships only CUDA 12.8 (/usr/local/cuda-12.8) and no sudo,
# while SGLang main needs CUDA 13 (torch 2.13 cu13, flashinfer[cu13]). The env
# carries its own cuda-toolkit 13.0 so JIT builds see a matching nvcc, exactly
# like the official image (nvidia/cuda:13.0.3). flashinfer's prebuilt cubin and
# jit-cache wheels are installed the same way docker/Dockerfile does it.
set -euo pipefail

SGL_ROOT="${SGL_ROOT:-$HOME/sgl}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-sgl-profiler}"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY_VER="${PY_VER:-3.12}"
CUDA_TK="${CUDA_TK:-13.0}"                  # official image is CUDA 13.0.3
FLASHINFER_VER="${FLASHINFER_VER:-0.6.18}"  # keep aligned with python/pyproject.toml
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-Linux-x86_64.sh}"
PROFILER_REPO="${PROFILER_REPO:-https://github.com/bowenwan6/sglang-vllm-profiler.git}"
SGLANG_FORK="${SGLANG_FORK:-https://github.com/bowenwan6/sglang.git}"
SGLANG_UPSTREAM="${SGLANG_UPSTREAM:-https://github.com/sgl-project/sglang.git}"
SGLANG_REF="${SGLANG_REF:-upstream/main}"   # checked out as local branch main-upstream
SGLANG_EXTRAS="${SGLANG_EXTRAS:-dev}"       # dev = runtime + pytest; "all" also pulls diffusion deps
MODELS="${MODELS:-Qwen/Qwen3-VL-4B-Instruct}"
LOG_DIR="$SGL_ROOT/logs"
UV="$ENV_PREFIX/bin/uv"

mkdir -p "$SGL_ROOT" "$LOG_DIR" "$HF_HOME"
LOG="$LOG_DIR/setup_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
CUR=start
step() { CUR="$1"; printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }
trap 'echo "FAILED in phase $CUR (line $LINENO)"; touch "$SGL_ROOT/.setup_failed"' ERR
rm -f "$SGL_ROOT/.setup_done" "$SGL_ROOT/.setup_failed"

activate() { # conda-forge's ~cuda-nvcc_activate.sh reads unset vars, so drop -u around it
  set +u; # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"; set -u; }

phase_conda() {
  step "conda (Miniforge) -> $CONDA_ROOT"
  if [ -x "$CONDA_ROOT/bin/conda" ]; then echo "present: $("$CONDA_ROOT/bin/conda" --version)"; else
    curl -fsSL "$MINIFORGE_URL" -o "$SGL_ROOT/Miniforge3.sh"
    bash "$SGL_ROOT/Miniforge3.sh" -b -p "$CONDA_ROOT"
  fi
  if ! grep -q '# >>> sgl-profiler >>>' ~/.bashrc; then printf '\n' >> ~/.bashrc; cat >> ~/.bashrc <<EOF
# >>> sgl-profiler >>>
export HF_HOME="$HF_HOME"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate $ENV_NAME 2>/dev/null || true
# <<< sgl-profiler <<<
EOF
  echo "bashrc: added sgl-profiler block"; fi
}

phase_env() {
  step "conda env $ENV_NAME (python $PY_VER + cuda-toolkit $CUDA_TK)"
  if [ -x "$ENV_PREFIX/bin/python" ]; then echo "env present"; else
    "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" "python=$PY_VER"; fi
  if [ -x "$ENV_PREFIX/bin/nvcc" ]; then echo "toolkit present: $("$ENV_PREFIX/bin/nvcc" --version | tail -1)"; else
    "$CONDA_ROOT/bin/conda" install -y -n "$ENV_NAME" -c conda-forge "cuda-toolkit=$CUDA_TK" ninja; fi
  mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"
  cat > "$ENV_PREFIX/etc/conda/activate.d/zz-sgl-profiler.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export HF_HOME="${HF_HOME:-$HOME/hf}"
# radix-gpu-env writes the *physical* GPU index into ~/.bashrc, but the node already restricts this
# user to that GPU and CUDA enumerates it as device 0; an out-of-range index hides the GPU entirely.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "0" ] \
   && [ "$(nvidia-smi -L 2>/dev/null | wc -l)" = "1" ]; then export CUDA_VISIBLE_DEVICES=0; fi
EOF
  echo 'unset CUDA_HOME' > "$ENV_PREFIX/etc/conda/deactivate.d/zz-sgl-profiler.sh"
  [ -e "$ENV_PREFIX/lib64" ] || ln -s lib "$ENV_PREFIX/lib64"   # some build helpers look for lib64
  "$ENV_PREFIX/bin/python" -m pip install -q --upgrade pip uv
  # never --upgrade huggingface_hub here: transformers pins it (<2.0) and the sglang phase resolves it
  [ -x "$ENV_PREFIX/bin/hf" ] || "$ENV_PREFIX/bin/python" -m pip install -q "huggingface_hub<2"
  echo "uv: $("$UV" --version)  hf: $("$ENV_PREFIX/bin/hf" version 2>/dev/null | head -1)"
}

phase_repos() {
  step "repos -> $SGL_ROOT/{profiler,sglang}"
  if [ -d "$SGL_ROOT/profiler/.git" ]; then git -C "$SGL_ROOT/profiler" pull --ff-only
  else git clone "$PROFILER_REPO" "$SGL_ROOT/profiler"; fi
  if [ -d "$SGL_ROOT/sglang/.git" ]; then git -C "$SGL_ROOT/sglang" fetch --all --prune
  else
    git clone --filter=blob:none "$SGLANG_FORK" "$SGL_ROOT/sglang"
    git -C "$SGL_ROOT/sglang" remote add upstream "$SGLANG_UPSTREAM"
    git -C "$SGL_ROOT/sglang" fetch --filter=blob:none upstream main
  fi
  git -C "$SGL_ROOT/sglang" checkout -q -B main-upstream "$SGLANG_REF"
  echo "profiler @ $(git -C "$SGL_ROOT/profiler" log -1 --format='%h %cs %s')"
  echo "sglang   @ $(git -C "$SGL_ROOT/sglang" log -1 --format='%h %cs %s') [$SGLANG_REF]"
}

phase_models() {
  step "models -> $HF_HOME : $MODELS"
  # runs in its own tmux window in parallel with the main phases: wait for the env's hf CLI (<= 10 min)
  for _ in $(seq 1 120); do [ -x "$ENV_PREFIX/bin/hf" ] && break; sleep 5; done
  HF=("$ENV_PREFIX/bin/hf"); [ -x "${HF[0]}" ] || HF=("$HOME/.local/bin/uvx" --from huggingface_hub hf)
  for m in $MODELS; do "${HF[@]}" download "$m"; done
  du -sh "$HF_HOME"
}

phase_sglang() {
  step "sglang editable install python[$SGLANG_EXTRAS]"
  activate; cd "$SGL_ROOT/sglang"
  # The node has no cargo. The Rust extensions (RustServer, rust tree core, grpc, inkling image
  # processing) are all opt-in at runtime, so build without them; set SGLANG_BUILD_RUST_EXTS=all
  # after installing rustup (rust/rust-toolchain.toml pins channel 1.92) if one is ever needed.
  export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
  "$UV" pip install --python "$ENV_PREFIX/bin/python" --prerelease=allow -e "python[$SGLANG_EXTRAS]"
}

phase_flashinfer() {
  step "flashinfer prebuilt cubin + jit-cache $FLASHINFER_VER (cu130)"
  "$UV" pip install --python "$ENV_PREFIX/bin/python" --no-deps "flashinfer-cubin==$FLASHINFER_VER" --index-url https://flashinfer.ai/whl
  "$UV" pip install --python "$ENV_PREFIX/bin/python" --no-deps "flashinfer-jit-cache==$FLASHINFER_VER" --index-url https://flashinfer.ai/whl/cu130
}

phase_verify() {
  step verify
  activate
  echo "nvcc: $(nvcc --version | tail -1)"; echo "CUDA_HOME=$CUDA_HOME"
  python - <<'EOF'
import importlib.util as u
import torch, sglang, flashinfer, sgl_kernel
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")
print("sglang", sglang.__version__, "| flashinfer", flashinfer.__version__, "| sgl_kernel", getattr(sgl_kernel, "__version__", "?"))
print("flashinfer jit-cache:", u.find_spec("flashinfer_jit_cache") is not None, "| cubin:", u.find_spec("flashinfer_cubin") is not None)
EOF
}

phase_manifest() {
  step "manifest -> $SGL_ROOT/env_manifest.txt"
  activate
  {
    echo "generated: $(date -u +%FT%TZ) host=$(hostname) user=$(whoami)"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    echo "nvcc: $(nvcc --version | tail -1)"
    echo "profiler: $(git -C "$SGL_ROOT/profiler" rev-parse HEAD) $(git -C "$SGL_ROOT/profiler" branch --show-current)"
    echo "sglang: $(git -C "$SGL_ROOT/sglang" rev-parse HEAD) $(git -C "$SGL_ROOT/sglang" branch --show-current) (ref $SGLANG_REF)"
    echo "--- uv pip freeze ---"; "$UV" pip freeze --python "$ENV_PREFIX/bin/python"
    echo "--- conda list --explicit ---"; "$CONDA_ROOT/bin/conda" list -n "$ENV_NAME" --explicit
  } > "$SGL_ROOT/env_manifest.txt"
  echo "wrote $SGL_ROOT/env_manifest.txt ($(wc -l < "$SGL_ROOT/env_manifest.txt") lines)"
}

PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(conda env repos sglang flashinfer verify manifest)
for p in "${PHASES[@]}"; do "phase_$p"; done
touch "$SGL_ROOT/.setup_done"
step "done: ${PHASES[*]}   log: $LOG"
