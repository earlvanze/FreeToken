#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap a cacheable Vast Serverless worker from public, reviewable sources.
# The checkpoint and both git checkouts live under /workspace so an inactive
# worker can retain them on its billable storage volume.

workspace="${WORKSPACE_DIR:-/workspace}"
freetoken_dir="${TEKIZAI_FREETOKEN_DIR:-${workspace}/freetoken}"
model_dir="${TEKIZAI_FREETOKEN_MODEL_PATH:-${workspace}/models/GLM-5.3-Flash-NVFP4}"
worker_source_dir="${TEKIZAI_WORKER_SOURCE_DIR:-${workspace}/vast-pyworker}"
pyworker_uv_cache="${TEKIZAI_PYWORKER_UV_CACHE:-${workspace}/pyworker-uv-cache}"

freetoken_repo="${TEKIZAI_FREETOKEN_REPO:-https://github.com/earlvanze/FreeToken.git}"
freetoken_ref="${TEKIZAI_FREETOKEN_REF:-feat/glm53-flash}"
freetoken_expected_commit="${TEKIZAI_FREETOKEN_EXPECTED_COMMIT:-}"
worker_repo="${PYWORKER_REPO:-https://github.com/tekizai/tekizai-vast-setup.git}"
worker_ref="${PYWORKER_REF:-feat/glm53-flash-vast}"
worker_expected_commit="${TEKIZAI_WORKER_EXPECTED_COMMIT:-}"
model_repo="${TEKIZAI_MODEL_REPO:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
model_bench_dtype="${TEKIZAI_MODEL_BENCH_DTYPE:-nvfp4}"
model_launcher="${TEKIZAI_WORKER_LAUNCHER:-serverless/glm53_flash/start_freetoken.sh}"
pyworker_bootstrap_ref="${TEKIZAI_PYWORKER_BOOTSTRAP_REF:-2207a3f94b55a0921c1641520eeb83de5a0c1611}"
bootstrap="${workspace}/vast-pyworker-bootstrap.sh"
provision_marker="${TEKIZAI_PROVISION_MARKER:-${workspace}/.tekizai-freetoken-provisioned}"
provision_marker_value="${freetoken_expected_commit:-$freetoken_ref}|${worker_expected_commit:-$worker_ref}|${pyworker_bootstrap_ref}"

export DEBIAN_FRONTEND=noninteractive
export PATH="${HOME}/.local/bin:${PATH}"
export MODEL_LOG="${TEKIZAI_FREETOKEN_LOG:-${workspace}/logs/freetoken-glm53.log}"
export PYWORKER_REPO="$worker_repo"
export PYWORKER_REF="$worker_ref"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$workspace" "$(dirname "$MODEL_LOG")"

model_download_pid=""
pyworker_pid=""
model_launcher_pid=""
cleanup() {
  local pid
  for pid in "$model_launcher_pid" "$pyworker_pid" "$model_download_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_runtime() {
  echo "FREETOKEN_PROVISION_STAGE=pyworker_start"
  UV_CACHE_DIR="$pyworker_uv_cache" \
    USE_SYSTEM_PYTHON=true \
    ROTATE_MODEL_LOG=true \
    "$bootstrap" &
  pyworker_pid=$!

  echo "FREETOKEN_PROVISION_STAGE=model_start"
  "$worker_source_dir/$model_launcher" &
  model_launcher_pid=$!

  wait "$pyworker_pid"
}

checkout_matches_expected_commit() {
  local destination="$1"
  local expected="$2"
  [[ -d "$destination/.git" ]] || return 1
  [[ -z "$expected" ]] || \
    [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null)" == "$expected" ]]
}

if [[ -f "$provision_marker" ]] \
  && [[ "$(<"$provision_marker")" == "$provision_marker_value" ]] \
  && [[ -x "$bootstrap" ]] \
  && [[ -x "$freetoken_dir/.venv/bin/ft" ]] \
  && [[ -x "$worker_source_dir/$model_launcher" ]] \
  && [[ -s "$model_dir/config.json" ]] \
  && [[ -x /usr/local/cuda-13.0/bin/nvcc ]] \
  && checkout_matches_expected_commit "$freetoken_dir" "$freetoken_expected_commit" \
  && checkout_matches_expected_commit "$worker_source_dir" "$worker_expected_commit"; then
  echo "FREETOKEN_PROVISION_STAGE=fast_resume"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  start_runtime
  exit $?
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# Start the large checkpoint transfer before installing the build toolchain.
# Overlapping these independent stages keeps a cold worker inside Vast's model
# loading deadline, while `hf download --local-dir` still resumes in place.
download_model() {
  local attempt status
  for attempt in 1 2 3 4 5; do
    echo "FREETOKEN_PROVISION_STAGE=model_download attempt=${attempt}"
    if uvx --from huggingface-hub==1.29.0 hf download "$model_repo" --local-dir "$model_dir"; then
      echo "FREETOKEN_PROVISION_STAGE=model_download_complete"
      return 0
    else
      status=$?
    fi
    echo "FREETOKEN_PROVISION_STAGE=model_download_retry status=${status}"
    sleep "$((attempt * 10))"
  done
  return "$status"
}

download_model &
model_download_pid=$!

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential ca-certificates cuda-compiler-13-0 cuda-cudart-dev-13-0 \
    curl git libcurand-dev-13-0 ninja-build numactl python3.12-dev util-linux
  apt-get clean
fi

# The small CUDA base image reaches Vast's running state quickly enough for the
# Serverless autoscaler. Install only FreeToken's required build toolchain after
# rental, then point PyTorch's extension builder at that toolkit explicitly.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

checkout_ref() {
  local repo="$1"
  local ref="$2"
  local destination="$3"

  if [[ ! -d "$destination/.git" ]]; then
    mkdir -p "$destination"
    git -C "$destination" init
    git -C "$destination" remote add origin "$repo"
  fi
  git -C "$destination" fetch --depth 1 origin "$ref"
  git -C "$destination" checkout --detach --force FETCH_HEAD
}

checkout_ref "$freetoken_repo" "$freetoken_ref" "$freetoken_dir"
checkout_ref "$worker_repo" "$worker_ref" "$worker_source_dir"

verify_commit() {
  local destination="$1"
  local expected="$2"
  local actual

  [[ -z "$expected" ]] && return 0
  actual="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Expected commit %s at %s, got %s\n' \
      "$expected" "$destination" "$actual" >&2
    return 1
  fi
}

verify_commit "$freetoken_dir" "$freetoken_expected_commit"
verify_commit "$worker_source_dir" "$worker_expected_commit"

echo "FREETOKEN_PROVISION_STAGE=pyworker_bootstrap"
curl -fsSL \
  "https://raw.githubusercontent.com/vast-ai/pyworker/${pyworker_bootstrap_ref}/start_server.sh" \
  -o "$bootstrap"
chmod 0755 "$bootstrap"
UV_CACHE_DIR="$pyworker_uv_cache" \
  USE_SYSTEM_PYTHON=true \
  ROTATE_MODEL_LOG=true \
  "$bootstrap" &
pyworker_pid=$!

echo "FREETOKEN_PROVISION_STAGE=dependencies"
if [[ ! -x "$freetoken_dir/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$freetoken_dir/.venv"
fi
uv pip install --python "$freetoken_dir/.venv/bin/python" -e "$freetoken_dir[accel]"

wait "$model_download_pid"

# FreeToken's hardware profile upgrades the safe offload default to the faster
# CPU/GPU hybrid backend only when this exact worker clears the bandwidth gate.
echo "FREETOKEN_PROVISION_STAGE=bandwidth_check"
"$freetoken_dir/.venv/bin/ft" bench bw --dtype "$model_bench_dtype"

printf '%s\n' "$provision_marker_value" >"$provision_marker"

echo "FREETOKEN_PROVISION_STAGE=model_start"
"$worker_source_dir/$model_launcher" &
model_launcher_pid=$!

wait "$pyworker_pid"
