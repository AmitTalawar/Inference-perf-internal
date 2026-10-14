#!/usr/bin/env bash
# Sweep load.stages[].concurrent_sessions while replaying weka-modified-vm.yaml,
# restarting vLLM with a clean CPU + storage slate before every concurrency run.
#
# What this does, per concurrency value (8 10 12 16 24 32 40 48 64):
#   1. Tear down any running vLLM server.
#   2. Clean slate (in this exact order), BEFORE starting vLLM:
#        rm -rf /dev/shm/*                        (offloading connector leaks shm)
#        rm -rf <OFFLOAD_FS_DIR>/*                (secondary fs storage tier)
#   3. Start vLLM (poolside/Laguna-XS-2.1, OffloadingConnector + fs tier) and
#      wait until /v1/models is ready.
#   4. Run inference-perf for exactly this one concurrency.
#   5. Tear down the server, then repeat the clean-slate order for the next run.
#
# Only concurrent_sessions changes between runs. num_sessions,
# num_dataset_entries and duplicate_sessions_target stay at whatever the base
# config has (100 / 100 / 100 right now). duration / warmup / every other stage
# field is inherited from the base config's first stage, so the time-boxed TTFT
# window set in weka-modified-vm.yaml is preserved.
#
# Artifacts are isolated per concurrency under:
#     <RESULTS_ROOT>/concurrency=<N>/
#   e.g. benchmarks/concurrency-sweep/concurrency=8/{summary_lifecycle_metrics.json,
#        vllm_command.txt, vllm.log, inference-perf.log}
#   vllm_command.txt = exact copy-pasteable vLLM launch command for that run.
#   vllm.log         = that command line followed by all vLLM stdout/stderr.
# A concurrency whose summary_lifecycle_metrics.json already exists is skipped
# (resumable sweep).
#
# WARNING: this runs `rm -rf /dev/shm/*`. Do not run other /dev/shm-dependent
# workloads on this host during the sweep.
#
# Usage:
#   scripts/sweep_concurrent_sessions.sh
#   scripts/sweep_concurrent_sessions.sh --concurrencies 8 16 32
#   scripts/sweep_concurrent_sessions.sh --results-root /tmp/my-sweep
#   scripts/sweep_concurrent_sessions.sh --config /path/to/weka-modified-vm.yaml

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${CONFIG:-${REPO_DIR}/weka-modified-vm.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/benchmarks/concurrency-sweep}"

# ---- inference-perf (traffic) ------------------------------------------------
# Defaults to the inference-perf venv at ~/amit/build/ip; override with
# INFER_CMD, e.g. INFER_CMD="uv run inference-perf".
INFER_CMD="${INFER_CMD:-${HOME}/amit/build/ip/bin/inference-perf}"

# Python used only for the tiny stage-override helper (needs PyYAML). Prefer the
# python from the same venv as INFER_CMD when INFER_CMD is a plain binary path.
_infer_bin="${INFER_CMD%% *}"
if [[ -z "${HELPER_PY:-}" && "${_infer_bin}" == */bin/inference-perf && -x "${_infer_bin%/inference-perf}/python3" ]]; then
  HELPER_PY="${_infer_bin%/inference-perf}/python3"
fi
HELPER_PY="${HELPER_PY:-python3}"

# ---- vLLM (server) -----------------------------------------------------------
VLLM_BIN="${VLLM_BIN:-${HOME}/amit/build/venv/bin/vllm}"
# The vLLM venv's bin dir must be on PATH: FlashInfer JIT-compiles CUDA kernels
# at startup via `ninja` (installed as the venv's `ninja` console script). When
# vLLM is launched by absolute path without activating the venv, `ninja` is not
# found and EngineCore crashes. Prepend it here so JIT build tools resolve.
# Derived from VLLM_BIN so it stays correct if VLLM_BIN is overridden.
VLLM_VENV_BIN="${VLLM_VENV_BIN:-$(dirname "${VLLM_BIN}")}"
MODEL="${MODEL:-poolside/Laguna-XS-2.1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
OFFLOAD_FS_DIR="${OFFLOAD_FS_DIR:-${HOME}/amit/dev/benchmark_state/offload_fs}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-483183820800}"
OFFLOAD_BLOCK_SIZE="${OFFLOAD_BLOCK_SIZE:-256}"

READY_URL="http://${VLLM_HOST}:${PORT}/v1/models"
SERVE_TIMEOUT_S="${SERVE_TIMEOUT_S:-1800}"   # max wait for /v1/models to come up
GPU_FREE_WAIT_S="${GPU_FREE_WAIT_S:-30}"     # settle time after server exits

CONCURRENCIES=(8 10 12 16 24 32 40 48 64)

usage() {
  sed -n '2,36p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        shift; CONFIG="$1"; shift ;;
    --results-root)  shift; RESULTS_ROOT="$1"; shift ;;
    --concurrencies)
      shift
      CONCURRENCIES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CONCURRENCIES+=("$1")
        shift
      done
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for req in "${CONFIG}" "${VLLM_BIN}"; do
  if [[ ! -e "${req}" ]]; then
    echo "missing required path: ${req}" >&2
    exit 1
  fi
done

mkdir -p "${RESULTS_ROOT}" "${OFFLOAD_FS_DIR}"

# kv-transfer-config JSON. block_size and cpu_bytes_to_use are overridable via env.
KV_TRANSFER_CONFIG="$(cat <<JSON
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": ${CPU_BYTES_TO_USE},
    "block_size": ${OFFLOAD_BLOCK_SIZE},
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "fs",
        "root_dir": "${OFFLOAD_FS_DIR}",
        "n_read_threads": 32,
        "n_write_threads": 16
      }
    ]
  }
}
JSON
)"

# ---- helpers ----------------------------------------------------------------

# Build the --load.stages JSON override for one concurrency value by reading the
# first stage of the base config and swapping only concurrent_sessions.
stage_override_for() {
  local conc="$1"
  CONFIG="${CONFIG}" CONC="${conc}" "${HELPER_PY}" - <<'PY'
import json
import os
import sys

import yaml

config_path = os.environ["CONFIG"]
conc = int(os.environ["CONC"])

with open(config_path) as fh:
    cfg = yaml.safe_load(fh)

stages = (cfg.get("load") or {}).get("stages") or []
if not stages:
    print("base config has no load.stages", file=sys.stderr)
    sys.exit(1)

# Carry over every field of the first stage; swap only concurrent_sessions.
stage = dict(stages[0])
stage["concurrent_sessions"] = conc
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
    # Kill the whole process group started via setsid.
    kill -- -"${SERVE_PID}" 2>/dev/null || kill -TERM "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
  unset SERVE_PID
  wait_vllm_gone || true
  sleep "${GPU_FREE_WAIT_S}"
}

# Clean slate performed BEFORE every server start, in the required order:
#   1. rm -rf /dev/shm/*          (OffloadingConnector shm not removed on exit)
#   2. rm -rf <OFFLOAD_FS_DIR>/*  (fs secondary storage tier)
clean_slate() {
  echo "    clean slate: rm -rf /dev/shm/*"
  rm -rf /dev/shm/* 2>/dev/null || true
  echo "    clean slate: rm -rf ${OFFLOAD_FS_DIR}/*"
  rm -rf "${OFFLOAD_FS_DIR:?}/"* 2>/dev/null || true
  mkdir -p "${OFFLOAD_FS_DIR}"
}

# Launch vLLM. Logs the exact command line to <cmdfile> and all server
# stdout/stderr to <logfile> (both live in the per-concurrency artifact dir).
start_server() {
  local logfile="$1"
  local cmdfile="$2"

  # The literal argv we exec, as a copy-pasteable one-liner. %q quotes each
  # token (incl. the JSON kv-transfer-config) so it can be re-run verbatim.
  local serve_argv=(
    "${VLLM_BIN}" serve "${MODEL}"
    --host "${VLLM_HOST}"
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
  )
  {
    echo "# vLLM launch for concurrency run"
    echo "# timestamp: $(date -Is)"
    echo "# cwd: $(pwd)"
    echo "# stdout/stderr -> $(basename "${logfile}")"
    echo "# PATH prefixed with: ${VLLM_VENV_BIN} (for FlashInfer ninja JIT)"
    echo
    printf 'PATH=%q:"$PATH" ' "${VLLM_VENV_BIN}"
    printf 'PYTHONHASHSEED=0 VLLM_SERVER_DEV_MODE=1'
    printf ' %s' "$(printf '%q ' "${serve_argv[@]}")"
    echo
  } >"${cmdfile}"

  echo "    starting vLLM (model=${MODEL}, port=${PORT}, offload_fs=${OFFLOAD_FS_DIR})"
  echo "    vLLM command -> ${cmdfile}"
  echo "    vLLM output  -> ${logfile}"

  # Mirror the command line at the top of the server log too, so vllm.log is
  # self-contained. setsid => own process group so stop_server can signal the
  # whole tree. PATH is extended with the vLLM venv bin so FlashInfer's ninja
  # JIT build at startup can find its tools.
  cat "${cmdfile}" >"${logfile}"
  echo "----- vLLM output below -----" >>"${logfile}"
  PYTHONHASHSEED=0 VLLM_SERVER_DEV_MODE=1 PATH="${VLLM_VENV_BIN}:${PATH}" setsid "${serve_argv[@]}" \
    >>"${logfile}" 2>&1 &
  SERVE_PID=$!
}

# Always tear the server down + clean up on any exit.
trap 'stop_server; clean_slate' EXIT

# ---- preamble ---------------------------------------------------------------

echo "config:          ${CONFIG}"
echo "results root:    ${RESULTS_ROOT}"
echo "concurrencies:   ${CONCURRENCIES[*]}"
echo "infer cmd:       ${INFER_CMD}"
echo "vllm bin:        ${VLLM_BIN}"
echo "vllm venv bin:   ${VLLM_VENV_BIN} (prepended to PATH for ninja JIT)"
echo "model:           ${MODEL}"
echo "offload fs dir:  ${OFFLOAD_FS_DIR}"
echo

# ---- sweep ------------------------------------------------------------------

for CONC in "${CONCURRENCIES[@]}"; do
  out="${RESULTS_ROOT}/concurrency=${CONC}"

  if [[ -f "${out}/summary_lifecycle_metrics.json" ]]; then
    echo "=== skip concurrency=${CONC} (already complete) ==="
    continue
  fi

  mkdir -p "${out}"
  stages_json="$(stage_override_for "${CONC}")"

  echo "=== concurrency=${CONC} starting ==="
  echo "    stages override: ${stages_json}"
  echo "    artifacts:       ${out}"

  # 1. Ensure no stale server, then 2. clean slate, then 3. start fresh server.
  stop_server
  if port_busy; then
    echo "port ${PORT} still in use after teardown; refusing to start another server" >&2
    exit 1
  fi
  clean_slate

  # Log the exact launch command (vllm_command.txt) + server output (vllm.log).
  start_server "${out}/vllm.log" "${out}/vllm_command.txt"
  if ! wait_ready; then
    echo "concurrency=${CONC}: vLLM failed to become ready; see ${out}/vllm.log" >&2
    exit 1
  fi

  # 4. Run inference-perf for this one concurrency.
  echo "    running inference-perf"
  # shellcheck disable=SC2086
  (
    cd "${REPO_DIR}"
    ${INFER_CMD} -c "${CONFIG}" \
      --load.stages "${stages_json}" \
      --storage.local_storage.path "${out}"
  ) 2>&1 | tee "${out}/inference-perf.log"

  # 5. Tear down; clean_slate runs at the top of the next iteration (and the
  #    EXIT trap covers the final iteration / any early exit).
  echo "    stopping vLLM"
  stop_server

  if [[ ! -f "${out}/summary_lifecycle_metrics.json" ]]; then
    echo "concurrency=${CONC}: inference-perf did not write summary_lifecycle_metrics.json" >&2
    echo "see ${out}/inference-perf.log" >&2
    exit 1
  fi
  echo "=== concurrency=${CONC} done ==="
  echo
done

echo "Sweep complete. Reports under ${RESULTS_ROOT}"
