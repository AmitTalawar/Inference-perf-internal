#!/usr/bin/env bash
# Sweep concurrent_sessions across three CPU-offload eviction scenarios.
#
# For each concurrency in order, deploy scenarios one at a time:
#   (i)   gpu-lru + cpu-lru
#   (ii)  gpu-lru + cpu-prefix_pruning_on_sessions
#   (iii) gpu-prefix_pruning_on_sessions + cpu-prefix_pruning_on_sessions
#
# Lifecycle per (concurrency, scenario):
#   1. Stop any running vLLM server
#   2. Clean /dev/shm leftover OffloadingConnector mmaps
#   3. Unset GPU suffix-eviction env (then set it only for scenario iii)
#   4. Start vLLM (CPUOffloadingSpec only — no FS secondary tier)
#   5. Wait until /v1/models is ready
#   6. Run inference-perf with stage overrides
#   7. Stop server + clean /dev/shm again
#
# Artifacts:
#   <RESULTS_ROOT>/concurrency=<N>/<scenario>/{vllm.log,vllm_command.txt,
#     inference-perf.log, summary_lifecycle_metrics.json, ...}
#
# WARNING: runs `rm -rf /dev/shm/*`. Do not run other /dev/shm workloads on
# this host during the sweep.
#
# Scenarios (select with --scenarios; default runs all three in this order):
#   lru-lru            GPU native LRU + CPU LRU
#   lru-sessions       GPU native LRU + CPU prefix_pruning_on_sessions
#   sessions-sessions  GPU + CPU prefix_pruning_on_sessions
#
# Usage:
#   scripts/sweep_cpu_offload_eviction.sh
#   scripts/sweep_cpu_offload_eviction.sh --config free-inf-mistral.yaml
#   scripts/sweep_cpu_offload_eviction.sh --concurrencies 24 36 48
#   scripts/sweep_cpu_offload_eviction.sh --scenarios lru-lru lru-sessions
#   scripts/sweep_cpu_offload_eviction.sh --scenarios sessions-sessions --concurrencies 24
#   scripts/sweep_cpu_offload_eviction.sh --duration 900 --warmup 300
#   scripts/sweep_cpu_offload_eviction.sh --model mistralai/Ministral-3-8B-Reasoning-2512
#   scripts/sweep_cpu_offload_eviction.sh --cpu-bytes 214748364800 --block-size 256
#   scripts/sweep_cpu_offload_eviction.sh --cpu-cache-gb 200
#   scripts/sweep_cpu_offload_eviction.sh --dry-run

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${CONFIG:-${REPO_DIR}/free-inf-mistral.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/benchmarks/cpu-offload-eviction-sweep}"

# ---- inference-perf (traffic) ------------------------------------------------
if [[ -z "${INFER_CMD:-}" ]]; then
  if [[ -x "${HOME}/build/ip/bin/inference-perf" ]]; then
    INFER_CMD="${HOME}/build/ip/bin/inference-perf"
  elif [[ -x "${REPO_DIR}/.venv/bin/inference-perf" ]]; then
    INFER_CMD="${REPO_DIR}/.venv/bin/inference-perf"
  else
    INFER_CMD="inference-perf"
  fi
fi
_infer_bin="${INFER_CMD%% *}"
if [[ -z "${HELPER_PY:-}" && "${_infer_bin}" == */bin/inference-perf && -x "${_infer_bin%/inference-perf}/python3" ]]; then
  HELPER_PY="${_infer_bin%/inference-perf}/python3"
fi
HELPER_PY="${HELPER_PY:-python3}"

# ---- vLLM (server) -----------------------------------------------------------
VLLM_CODE_ROOT="${VLLM_CODE_ROOT:-${HOME}/dev/vllm-cpu-eviction}"
VLLM_VENV="${VLLM_VENV:-${HOME}/build/vllm-cpu-evict}"
VLLM_BIN="${VLLM_BIN:-${VLLM_VENV}/bin/vllm}"
VLLM_VENV_BIN="${VLLM_VENV_BIN:-$(dirname "${VLLM_BIN}")}"

MODEL="${MODEL:-mistralai/Ministral-3-8B-Reasoning-2512}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
READY_HOST="${READY_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-214748364800}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"

# Session / prefix-pruning defaults from vllm-cpu-eviction/server.sh
PREFIX_TIER_TOKENS="${PREFIX_TIER_TOKENS:-16384}"
PREFIX_TIER_STEP_TOKENS="${PREFIX_TIER_STEP_TOKENS:-16384}"
PREFIX_MAX_TIERS="${PREFIX_MAX_TIERS:-8}"
SESSION_DEAD_TTL_S="${SESSION_DEAD_TTL_S:-200}"
SESSION_DETECTOR="${SESSION_DETECTOR:-ewa}"
SESSION_EWA_ALPHA="${SESSION_EWA_ALPHA:-0.3}"
SESSION_DEAD_K="${SESSION_DEAD_K:-3}"
SESSION_MIN_GAPS="${SESSION_MIN_GAPS:-2}"
SESSION_QUANTILE="${SESSION_QUANTILE:-0.95}"
SESSION_GAP_WINDOW="${SESSION_GAP_WINDOW:-256}"
GPU_IDLE_TTL_S="${GPU_IDLE_TTL_S:-60}"

# Traffic stage defaults (override via CLI; else inherit from YAML first stage
# for any key not set here).
DURATION="${DURATION:-}"
WARMUP="${WARMUP:-}"
NUM_SESSIONS="${NUM_SESSIONS:-}"

READY_URL="http://${READY_HOST}:${PORT}/v1/models"
SERVE_TIMEOUT_S="${SERVE_TIMEOUT_S:-1800}"
# Large cpu_bytes_to_use (hundreds of GiB) makes OffloadingConnector mmap +
# CPU-tensor alloc dominate startup; vLLM's default 600s is often too low.
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
GPU_FREE_WAIT_S="${GPU_FREE_WAIT_S:-30}"

CONCURRENCIES=(24 30 36 48 64 80)
# Scenario order is intentional and fixed.
SCENARIOS=(lru-lru lru-sessions sessions-sessions)
DRY_RUN=0

usage() {
  sed -n '2,34p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)          shift; CONFIG="$1"; shift ;;
    --results-root)    shift; RESULTS_ROOT="$1"; shift ;;
    --model)           shift; MODEL="$1"; shift ;;
    --cpu-bytes)       shift; CPU_BYTES_TO_USE="$1"; shift ;;
    --cpu-cache-gb)
      shift
      CPU_BYTES_TO_USE="$(awk -v g="$1" 'BEGIN { printf "%.0f", g * 1024 * 1024 * 1024 }')"
      shift
      ;;
    --block-size)      shift; BLOCK_SIZE="$1"; shift ;;
    --duration)        shift; DURATION="$1"; shift ;;
    --warmup)          shift; WARMUP="$1"; shift ;;
    --num-sessions)    shift; NUM_SESSIONS="$1"; shift ;;
    --gpu-memory-utilization) shift; GPU_MEM_UTIL="$1"; shift ;;
    --max-model-len)   shift; MAX_MODEL_LEN="$1"; shift ;;
    --port)            shift; PORT="$1"; READY_URL="http://${READY_HOST}:${PORT}/v1/models"; shift ;;
    --gpu)             shift; GPU="$1"; shift ;;
    --vllm-bin)        shift; VLLM_BIN="$1"; VLLM_VENV_BIN="$(dirname "${VLLM_BIN}")"; shift ;;
    --vllm-code-root)  shift; VLLM_CODE_ROOT="$1"; shift ;;
    --infer-cmd)       shift; INFER_CMD="$1"; shift ;;
    --concurrencies)
      shift
      CONCURRENCIES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CONCURRENCIES+=("$1")
        shift
      done
      ;;
    --scenarios)
      shift
      SCENARIOS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SCENARIOS+=("$1")
        shift
      done
      ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for req in "${CONFIG}" "${VLLM_BIN}" "${VLLM_CODE_ROOT}"; do
  if [[ ! -e "${req}" ]]; then
    echo "missing required path: ${req}" >&2
    exit 1
  fi
done

mkdir -p "${RESULTS_ROOT}"

# ---- helpers ----------------------------------------------------------------

clear_gpu_eviction_env() {
  unset VLLM_SUFFIX_PRESERVING_EVICTION \
        VLLM_SUFFIX_EVICTION_TIER_TOKENS \
        VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS \
        VLLM_SUFFIX_EVICTION_MAX_TIERS \
        VLLM_SUFFIX_EVICTION_IDLE_TTL_S \
        VLLM_SUFFIX_SESSION_DEAD_TTL_S \
        VLLM_SUFFIX_SESSION_DETECTOR \
        VLLM_SUFFIX_SESSION_EWA_ALPHA \
        VLLM_SUFFIX_SESSION_DEAD_K \
        VLLM_SUFFIX_SESSION_MIN_GAPS \
        VLLM_SUFFIX_SESSION_QUANTILE \
        VLLM_SUFFIX_SESSION_GAP_WINDOW \
        || true
}

set_gpu_session_eviction_env() {
  export VLLM_SUFFIX_PRESERVING_EVICTION=1
  export VLLM_SUFFIX_EVICTION_TIER_TOKENS="${PREFIX_TIER_TOKENS}"
  export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS="${PREFIX_TIER_STEP_TOKENS}"
  export VLLM_SUFFIX_EVICTION_MAX_TIERS="${PREFIX_MAX_TIERS}"
  export VLLM_SUFFIX_EVICTION_IDLE_TTL_S="${GPU_IDLE_TTL_S}"
  export VLLM_SUFFIX_SESSION_DEAD_TTL_S="${SESSION_DEAD_TTL_S}"
  export VLLM_SUFFIX_SESSION_DETECTOR="${SESSION_DETECTOR}"
  export VLLM_SUFFIX_SESSION_EWA_ALPHA="${SESSION_EWA_ALPHA}"
  export VLLM_SUFFIX_SESSION_DEAD_K="${SESSION_DEAD_K}"
  export VLLM_SUFFIX_SESSION_MIN_GAPS="${SESSION_MIN_GAPS}"
  export VLLM_SUFFIX_SESSION_QUANTILE="${SESSION_QUANTILE}"
  export VLLM_SUFFIX_SESSION_GAP_WINDOW="${SESSION_GAP_WINDOW}"
}

# CPUOffloadingSpec only (no TieringOffloadingSpec / no FS secondary tier).
kv_transfer_config_for() {
  local scenario="$1"
  local cpu_policy
  case "${scenario}" in
    lru-lru)           cpu_policy="lru" ;;
    lru-sessions)      cpu_policy="prefix_pruning_on_sessions" ;;
    sessions-sessions) cpu_policy="prefix_pruning_on_sessions" ;;
    *)
      echo "unknown scenario: ${scenario}" >&2
      return 2
      ;;
  esac

  if [[ "${cpu_policy}" == "lru" ]]; then
    cat <<JSON
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": ${CPU_BYTES_TO_USE},
    "block_size": ${BLOCK_SIZE},
    "eviction_policy": "lru"
  }
}
JSON
  else
    cat <<JSON
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": ${CPU_BYTES_TO_USE},
    "block_size": ${BLOCK_SIZE},
    "eviction_policy": "prefix_pruning_on_sessions",
    "first_tier_tokens": ${PREFIX_TIER_TOKENS},
    "tier_step_tokens": ${PREFIX_TIER_STEP_TOKENS},
    "max_tiers": ${PREFIX_MAX_TIERS},
    "session_dead_ttl_s": ${SESSION_DEAD_TTL_S},
    "session_ewa_alpha": ${SESSION_EWA_ALPHA},
    "session_dead_k": ${SESSION_DEAD_K},
    "session_min_gaps": ${SESSION_MIN_GAPS},
    "session_detector": "${SESSION_DETECTOR}",
    "session_quantile": ${SESSION_QUANTILE},
    "session_gap_window": ${SESSION_GAP_WINDOW}
  }
}
JSON
  fi
}

stage_override_for() {
  local conc="$1"
  CONFIG="${CONFIG}" CONC="${conc}" \
    DURATION="${DURATION}" WARMUP="${WARMUP}" NUM_SESSIONS="${NUM_SESSIONS}" \
    "${HELPER_PY}" - <<'PY'
import json
import os
import sys

import yaml

config_path = os.environ["CONFIG"]
conc = int(os.environ["CONC"])
duration = os.environ.get("DURATION") or None
warmup = os.environ.get("WARMUP") or None
num_sessions = os.environ.get("NUM_SESSIONS") or None

with open(config_path) as fh:
    cfg = yaml.safe_load(fh)

stages = (cfg.get("load") or {}).get("stages") or []
if not stages:
    print("base config has no load.stages", file=sys.stderr)
    sys.exit(1)

stage = dict(stages[0])
stage["concurrent_sessions"] = conc
if duration is not None:
    stage["duration"] = int(duration)
if warmup is not None:
    stage["warmup"] = int(warmup)
if num_sessions is not None:
    stage["num_sessions"] = int(num_sessions)

print(json.dumps([stage]))
PY
}

port_busy() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"
}

wait_ready() {
  local deadline=$((SECONDS + SERVE_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -sf "${READY_URL}" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${SERVE_PID:-}" ]] && ! kill -0 "${SERVE_PID}" 2>/dev/null; then
      echo "vLLM exited before becoming ready" >&2
      return 1
    fi
    sleep 5
  done
  echo "vLLM did not become ready within ${SERVE_TIMEOUT_S}s" >&2
  return 1
}

wait_vllm_gone() {
  local deadline=$((SECONDS + 120))
  while pgrep -f "${VLLM_BIN} serve" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "warning: vLLM still running after 120s" >&2
      return 1
    fi
    sleep 1
  done
  return 0
}

stop_server() {
  if [[ -n "${SERVE_PID:-}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    kill -- -"${SERVE_PID}" 2>/dev/null || kill -TERM "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
  unset SERVE_PID
  wait_vllm_gone || true
  sleep "${GPU_FREE_WAIT_S}"
}

clean_shm() {
  echo "    clean slate: rm -rf /dev/shm/*"
  rm -rf /dev/shm/* 2>/dev/null || true
}

start_server() {
  local scenario="$1"
  local logfile="$2"
  local cmdfile="$3"
  local kv_cfg
  kv_cfg="$(kv_transfer_config_for "${scenario}")"

  clear_gpu_eviction_env
  if [[ "${scenario}" == "sessions-sessions" ]]; then
    set_gpu_session_eviction_env
  fi

  local serve_argv=(
    "${VLLM_BIN}" serve "${MODEL}"
    --host "${VLLM_HOST}"
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
    --kv-transfer-config "${kv_cfg}"
  )

  {
    echo "# vLLM launch for concurrency/scenario run"
    echo "# timestamp: $(date -Is)"
    echo "# scenario: ${scenario}"
    echo "# cwd: $(pwd)"
    echo "# stdout/stderr -> $(basename "${logfile}")"
    echo "# VLLM_CODE_ROOT=${VLLM_CODE_ROOT} (on PYTHONPATH)"
    echo "# PATH prefixed with: ${VLLM_VENV_BIN}"
    echo "# CUDA_VISIBLE_DEVICES=${GPU}"
    if [[ "${scenario}" == "sessions-sessions" ]]; then
      echo "# GPU eviction env:"
      echo "#   VLLM_SUFFIX_PRESERVING_EVICTION=${VLLM_SUFFIX_PRESERVING_EVICTION}"
      echo "#   VLLM_SUFFIX_EVICTION_TIER_TOKENS=${VLLM_SUFFIX_EVICTION_TIER_TOKENS}"
      echo "#   VLLM_SUFFIX_SESSION_DEAD_TTL_S=${VLLM_SUFFIX_SESSION_DEAD_TTL_S}"
      echo "#   VLLM_SUFFIX_SESSION_DETECTOR=${VLLM_SUFFIX_SESSION_DETECTOR}"
    else
      echo "# GPU eviction env: cleared (native GPU LRU)"
    fi
    echo
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "${GPU}" "${VLLM_CODE_ROOT}"
    printf 'PATH=%q:"$PATH" ' "${VLLM_VENV_BIN}"
    printf 'PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=%q VLLM_KV_CONNECTOR_HIT_LOG=1 PYTHONHASHSEED=0 VLLM_SERVER_DEV_MODE=1' \
      "${VLLM_ENGINE_READY_TIMEOUT_S}"
    if [[ "${scenario}" == "sessions-sessions" ]]; then
      printf ' VLLM_SUFFIX_PRESERVING_EVICTION=1'
      printf ' VLLM_SUFFIX_EVICTION_TIER_TOKENS=%q' "${PREFIX_TIER_TOKENS}"
      printf ' VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=%q' "${PREFIX_TIER_STEP_TOKENS}"
      printf ' VLLM_SUFFIX_EVICTION_MAX_TIERS=%q' "${PREFIX_MAX_TIERS}"
      printf ' VLLM_SUFFIX_EVICTION_IDLE_TTL_S=%q' "${GPU_IDLE_TTL_S}"
      printf ' VLLM_SUFFIX_SESSION_DEAD_TTL_S=%q' "${SESSION_DEAD_TTL_S}"
      printf ' VLLM_SUFFIX_SESSION_DETECTOR=%q' "${SESSION_DETECTOR}"
      printf ' VLLM_SUFFIX_SESSION_EWA_ALPHA=%q' "${SESSION_EWA_ALPHA}"
      printf ' VLLM_SUFFIX_SESSION_DEAD_K=%q' "${SESSION_DEAD_K}"
      printf ' VLLM_SUFFIX_SESSION_MIN_GAPS=%q' "${SESSION_MIN_GAPS}"
      printf ' VLLM_SUFFIX_SESSION_QUANTILE=%q' "${SESSION_QUANTILE}"
      printf ' VLLM_SUFFIX_SESSION_GAP_WINDOW=%q' "${SESSION_GAP_WINDOW}"
    fi
    printf ' %s' "$(printf '%q ' "${serve_argv[@]}")"
    echo
  } >"${cmdfile}"

  echo "    starting vLLM scenario=${scenario} model=${MODEL} port=${PORT}"
  echo "    vLLM command -> ${cmdfile}"
  echo "    vLLM output  -> ${logfile}"

  if (( DRY_RUN )); then
    echo "    DRY_RUN=1: not launching server"
    return 0
  fi

  # Same pattern that live-updates in Cursor: single truncating redirect
  # `> logfile 2>&1` (not append, not stdbuf). Header + server share one fd.
  # Export PYTHONUNBUFFERED so EngineCore children inherit unbuffered stdio.
  export CUDA_VISIBLE_DEVICES="${GPU}"
  export PYTHONPATH="${VLLM_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export PATH="${VLLM_VENV_BIN}:${PATH}"
  export PYTHONUNBUFFERED=1
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}"
  export VLLM_KV_CONNECTOR_HIT_LOG=1
  export PYTHONHASHSEED=0
  export VLLM_SERVER_DEV_MODE=1
  setsid bash -c '
    cat "$1"
    echo "----- vLLM output below -----"
    shift
    exec "$@"
  ' bash "${cmdfile}" "${serve_argv[@]}" >"${logfile}" 2>&1 &
  SERVE_PID=$!
}

trap 'stop_server; clean_shm; clear_gpu_eviction_env' EXIT

# ---- preamble ---------------------------------------------------------------

echo "config:          ${CONFIG}"
echo "results root:    ${RESULTS_ROOT}"
echo "concurrencies:   ${CONCURRENCIES[*]}"
echo "scenarios:       ${SCENARIOS[*]}"
echo "infer cmd:       ${INFER_CMD}"
echo "vllm bin:        ${VLLM_BIN}"
echo "vllm code:       ${VLLM_CODE_ROOT}"
echo "vllm venv bin:   ${VLLM_VENV_BIN}"
echo "model:           ${MODEL}"
echo "cpu_bytes:       ${CPU_BYTES_TO_USE}"
echo "block_size:      ${BLOCK_SIZE}"
echo "max_model_len:   ${MAX_MODEL_LEN}"
echo "gpu_mem_util:    ${GPU_MEM_UTIL}"
echo "duration override: ${DURATION:-<(from yaml)>}"
echo "warmup override:   ${WARMUP:-<(from yaml)>}"
echo "num_sessions override: ${NUM_SESSIONS:-<(from yaml)>}"
echo "dry_run:         ${DRY_RUN}"
echo

# ---- sweep: concurrency outer, scenarios inner ------------------------------

for CONC in "${CONCURRENCIES[@]}"; do
  for SCENARIO in "${SCENARIOS[@]}"; do
    out="${RESULTS_ROOT}/concurrency=${CONC}/${SCENARIO}"

    if [[ -f "${out}/summary_lifecycle_metrics.json" ]]; then
      echo "=== skip concurrency=${CONC} scenario=${SCENARIO} (already complete) ==="
      continue
    fi

    mkdir -p "${out}"
    stages_json="$(stage_override_for "${CONC}")"

    echo "=== concurrency=${CONC} scenario=${SCENARIO} starting ==="
    echo "    stages override: ${stages_json}"
    echo "    artifacts:       ${out}"

    stop_server
    if port_busy; then
      echo "port ${PORT} still in use after teardown; refusing to start another server" >&2
      exit 1
    fi
    clean_shm

    start_server "${SCENARIO}" "${out}/vllm.log" "${out}/vllm_command.txt"

    if (( DRY_RUN )); then
      echo "    DRY_RUN=1: would run inference-perf with --load.stages ${stages_json}"
      echo "=== concurrency=${CONC} scenario=${SCENARIO} dry-run done ==="
      echo
      continue
    fi

    if ! wait_ready; then
      echo "concurrency=${CONC} scenario=${SCENARIO}: vLLM failed to become ready; see ${out}/vllm.log" >&2
      exit 1
    fi

    echo "    running inference-perf"
    # shellcheck disable=SC2086
    (
      cd "${REPO_DIR}"
      PYTHONUNBUFFERED=1 ${INFER_CMD} -c "${CONFIG}" \
        --load.stages "${stages_json}" \
        --storage.local_storage.path "${out}"
    ) 2>&1 | tee "${out}/inference-perf.log"

    echo "    stopping vLLM"
    stop_server
    clean_shm
    clear_gpu_eviction_env

    if [[ ! -f "${out}/summary_lifecycle_metrics.json" ]]; then
      echo "concurrency=${CONC} scenario=${SCENARIO}: missing summary_lifecycle_metrics.json" >&2
      echo "see ${out}/inference-perf.log" >&2
      exit 1
    fi
    echo "=== concurrency=${CONC} scenario=${SCENARIO} done ==="
    echo
  done
done

echo "Sweep complete. Reports under ${RESULTS_ROOT}"
