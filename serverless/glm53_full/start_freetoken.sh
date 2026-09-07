#!/usr/bin/env bash
set -Eeuo pipefail

export TEKIZAI_FREETOKEN_MODEL_PATH="${TEKIZAI_FREETOKEN_MODEL_PATH:-/workspace/models/GLM-5.3-NVFP4}"
export TEKIZAI_SERVED_MODEL="${TEKIZAI_SERVED_MODEL:-glm-5.3-nvfp4}"
export TEKIZAI_FREETOKEN_LOG="${TEKIZAI_FREETOKEN_LOG:-/workspace/logs/freetoken-glm53-full.log}"
export TEKIZAI_CPU_THREADS="${TEKIZAI_CPU_THREADS:-48}"
export TEKIZAI_MEMORY_RATIO="${TEKIZAI_MEMORY_RATIO:-0.95}"
export TEKIZAI_MAX_SEQ_LEN="${TEKIZAI_MAX_SEQ_LEN:-32768}"
export TEKIZAI_MAX_RUNNING_REQUESTS="${TEKIZAI_MAX_RUNNING_REQUESTS:-8}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/../glm53_flash/start_freetoken.sh"
