#!/usr/bin/env bash
set -Eeuo pipefail

# Full GLM-5.3 uses the same generic, cacheable worker lifecycle as Flash, but
# needs its own checkpoint, marker, log, and launcher defaults.
export TEKIZAI_FREETOKEN_MODEL_PATH="${TEKIZAI_FREETOKEN_MODEL_PATH:-/workspace/models/GLM-5.3-NVFP4}"
export TEKIZAI_MODEL_REPO="${TEKIZAI_MODEL_REPO:-LibertAIDAI/GLM-5.3-NVFP4}"
export TEKIZAI_SERVED_MODEL="${TEKIZAI_SERVED_MODEL:-glm-5.3-nvfp4}"
export TEKIZAI_FREETOKEN_LOG="${TEKIZAI_FREETOKEN_LOG:-/workspace/logs/freetoken-glm53-full.log}"
export TEKIZAI_PROVISION_MARKER="${TEKIZAI_PROVISION_MARKER:-/workspace/.tekizai-glm53-full-provisioned}"
export TEKIZAI_WORKER_LAUNCHER="${TEKIZAI_WORKER_LAUNCHER:-serverless/glm53_full/start_freetoken.sh}"
export TEKIZAI_CPU_THREADS="${TEKIZAI_CPU_THREADS:-48}"
export TEKIZAI_MEMORY_RATIO="${TEKIZAI_MEMORY_RATIO:-0.95}"
export TEKIZAI_MAX_SEQ_LEN="${TEKIZAI_MAX_SEQ_LEN:-32768}"
export TEKIZAI_MAX_RUNNING_REQUESTS="${TEKIZAI_MAX_RUNNING_REQUESTS:-8}"
export TEKIZAI_BENCHMARK_CONCURRENCY="${TEKIZAI_BENCHMARK_CONCURRENCY:-8}"
export TEKIZAI_ALLOW_PARALLEL_REQUESTS="${TEKIZAI_ALLOW_PARALLEL_REQUESTS:-1}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shared_provisioner="${script_dir}/../glm53_flash/provision_worker.sh"
if [[ ! -f "$shared_provisioner" ]]; then
  # Vast can download this entrypoint alone, before the worker checkout exists.
  # Fetch its sibling from the same immutable revision, never a moving branch.
  worker_ref="${PYWORKER_REF:-}"
  if [[ ! "$worker_ref" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Standalone provisioning requires PYWORKER_REF pinned to a full commit" >&2
    exit 2
  fi
  worker_repo="${PYWORKER_REPO:-https://github.com/tekizai/tekizai-vast-setup.git}"
  if [[ ! "$worker_repo" =~ ^https://github\.com/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+)$ ]]; then
    echo "Standalone provisioning requires a credential-free GitHub HTTPS repository" >&2
    exit 2
  fi
  worker_slug="${BASH_REMATCH[1]}/${BASH_REMATCH[2]%.git}"
  shared_provisioner="$(mktemp /tmp/tekizai-glm53-provision.XXXXXXXX.sh)"
  curl -fsSL "https://raw.githubusercontent.com/${worker_slug}/${worker_ref}/serverless/glm53_flash/provision_worker.sh" -o "$shared_provisioner"
fi
exec bash "$shared_provisioner"
