#!/usr/bin/env bash
# Sweep vLLM --block-size while replaying weka-modified-vm.yaml.
#
# Requirements this script is meant to satisfy:
#   * Restart vLLM for each block size (startup flag).
#   * Serve from ~/build/vllm-offloading; traffic from ~/build/inference-perf.
#   * Leave weka default_block_size=64 so compiled traces stay identical.
#   * Run inference-perf to completion per size.
#   * Time-series scrape of the full /metrics endpoint (histograms included).
#   * Isolate reports per block size; smoke results do not mark a full run done.
#   * After each serve, unlink leaked /dev/shm/vllm_offload_*.mmap (and
#     vllm_ec_*.mmap) so the next block size can allocate CPU offload.
#
# Usage:
#   scripts/sweep_block_size.sh              # 16..2048
#   scripts/sweep_block_size.sh --smoke      # block-size 16, 3 sessions
#   scripts/sweep_block_size.sh --block-sizes 16 32

set -euo pipefail

BUILD_DIR="${HOME}/build"
VLLM_BIN="${BUILD_DIR}/vllm-offloading/bin/vllm"
INFER_BIN="${BUILD_DIR}/inference-perf/bin/inference-perf"
SCRAPE_PY="${BUILD_DIR}/inference-perf/bin/python3"

INFER_DIR="/home/amit/dev/offloading/traces/inference-perf"
CONFIG="${INFER_DIR}/weka-modified-vm.yaml"
SCRAPE_SCRIPT="${INFER_DIR}/scripts/scrape_vllm_metrics.py"
RESULTS_ROOT="${INFER_DIR}/benchmarks/block-size-sweep"

MODEL="poolside/Laguna-XS-2.1"
PORT=8000
READY_URL="http://127.0.0.1:${PORT}/v1/models"
METRICS_URL="http://127.0.0.1:${PORT}/metrics"
SERVE_TIMEOUT_S=1800
GPU_FREE_WAIT_S=30
SCRAPE_INTERVAL_S=10

BLOCK_SIZES=(16 32 64 128 256 512 1024 2048)
SMOKE=0
INFER_EXTRA=()

usage() {
  sed -n '2,16p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      BLOCK_SIZES=(16)
      SCRAPE_INTERVAL_S=5
      # Weka load is session-based. 3 sessions is the smallest complete replay
      # that still exercises the same path as the full 40-session run.
      INFER_EXTRA=(
        --load.stages '[{"concurrent_sessions":3,"num_sessions":3}]'
        --data.weka_trace_replay.duplicate_sessions_target 3
      )
      shift
      ;;
    --block-sizes)
      shift
      BLOCK_SIZES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        BLOCK_SIZES+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for req in "${VLLM_BIN}" "${INFER_BIN}" "${SCRAPE_PY}" "${SCRAPE_SCRIPT}" "${CONFIG}"; do
  if [[ ! -e "${req}" ]]; then
    echo "missing required path: ${req}" >&2
    exit 1
  fi
done

mkdir -p "${RESULTS_ROOT}"

wait_ready() {
  local deadline=$((SECONDS + SERVE_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -sf "${READY_URL}" >/dev/null; then
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

stop_scraper() {
  if [[ -n "${SCRAPE_PID:-}" ]] && kill -0 "${SCRAPE_PID}" 2>/dev/null; then
    kill -TERM "${SCRAPE_PID}" 2>/dev/null || true
    wait "${SCRAPE_PID}" 2>/dev/null || true
  fi
  unset SCRAPE_PID
}

wait_vllm_gone() {
  local deadline=$((SECONDS + 120))
  while pgrep -f "${VLLM_BIN} serve" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "warning: vLLM still running after 120s; shm cleanup may race" >&2
      return 1
    fi
    sleep 1
  done
  return 0
}

# Native CPU offload maps /dev/shm/vllm_offload_<engine_id>.mmap
# (--kv-offloading-size 200 => ~200 GiB). SIGTERM often skips the
# SharedOffloadRegion destructor, so the file leaks and the next serve
# cannot allocate another 200 GiB in tmpfs.
cleanup_vllm_shm() {
  shopt -s nullglob
  local files=(/dev/shm/vllm_offload_*.mmap /dev/shm/vllm_ec_*.mmap)
  shopt -u nullglob
  if ((${#files[@]} == 0)); then
    echo "no leaked vLLM /dev/shm mmap files"
    return 0
  fi
  local f
  for f in "${files[@]}"; do
    echo "removing leaked shm $(ls -lh "${f}" | awk '{print $5}') ${f}"
    rm -f -- "${f}"
  done
}

wait_shm_capacity() {
  # Need room for the next --kv-offloading-size 200 plus a little headroom.
  local need_gib="${1:-210}"
  local deadline=$((SECONDS + 120))
  local avail
  while (( SECONDS < deadline )); do
    avail="$(df --output=avail -B1G /dev/shm | tail -n1 | tr -d ' ')"
    if (( avail >= need_gib )); then
      echo "/dev/shm free ${avail} GiB (need ${need_gib} GiB)"
      return 0
    fi
    echo "waiting for /dev/shm: ${avail} GiB free, need ${need_gib} GiB"
    sleep 2
  done
  echo "warning: /dev/shm only ${avail:-?} GiB free after wait" >&2
  return 1
}

stop_server() {
  stop_scraper
  if [[ -n "${SERVE_PID:-}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    kill -- -"${SERVE_PID}" 2>/dev/null || kill -TERM "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
  unset SERVE_PID
  wait_vllm_gone || true
  cleanup_vllm_shm
  wait_shm_capacity 210 || true
  sleep "${GPU_FREE_WAIT_S}"
}

port_busy() {
  ss -ltn | awk '{print $4}' | grep -qE ":${PORT}\$"
}

trap stop_server EXIT

echo "vllm binary:     ${VLLM_BIN}"
echo "inference-perf:  ${INFER_BIN}"
echo "block sizes:     ${BLOCK_SIZES[*]}"
echo "smoke:           ${SMOKE}"
echo "results:         ${RESULTS_ROOT}"

for X in "${BLOCK_SIZES[@]}"; do
  if (( SMOKE )); then
    out="${RESULTS_ROOT}/smoke-BS=${X}"
  else
    out="${RESULTS_ROOT}/BS=${X}"
  fi

  if [[ -f "${out}/summary_lifecycle_metrics.json" && -s "${out}/vllm_metrics.jsonl" ]]; then
    echo "=== skip $(basename "${out}") (already complete) ==="
    continue
  fi

  if port_busy; then
    echo "port ${PORT} is already in use; refusing to start another server" >&2
    exit 1
  fi

  # Previous serve may have leaked an offload mmap even if this shell
  # did not start it (e.g. a killed smoke run).
  cleanup_vllm_shm
  wait_shm_capacity 210 || true

  mkdir -p "${out}/prom"
  echo "=== BS=${X} starting serve ==="
  setsid "${VLLM_BIN}" serve "${MODEL}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --kv-offloading-backend native \
    --kv-offloading-size 200 \
    --block-size "${X}" \
    >"${out}/vllm.log" 2>&1 &
  SERVE_PID=$!

  if ! wait_ready; then
    echo "BS=${X} failed to start; see ${out}/vllm.log" >&2
    stop_server
    exit 1
  fi

  echo "=== BS=${X} scraping ${METRICS_URL} every ${SCRAPE_INTERVAL_S}s ==="
  "${SCRAPE_PY}" "${SCRAPE_SCRIPT}" \
    --url "${METRICS_URL}" \
    --interval "${SCRAPE_INTERVAL_S}" \
    --block-size "${X}" \
    --out "${out}/vllm_metrics.jsonl" \
    --raw-dir "${out}/prom" \
    >"${out}/scrape.log" 2>&1 &
  SCRAPE_PID=$!

  echo "=== BS=${X} running inference-perf ==="
  (
    cd "${INFER_DIR}"
    "${INFER_BIN}" -c "${CONFIG}" \
      --storage.local_storage.path "${out}" \
      "${INFER_EXTRA[@]}"
  ) | tee "${out}/inference-perf.log"

  echo "=== BS=${X} stopping scraper + server ==="
  stop_server

  if [[ ! -s "${out}/vllm_metrics.jsonl" ]]; then
    echo "BS=${X}: scraper wrote no samples" >&2
    exit 1
  fi
  if [[ ! -f "${out}/summary_lifecycle_metrics.json" ]]; then
    echo "BS=${X}: inference-perf did not write summary_lifecycle_metrics.json" >&2
    exit 1
  fi
  echo "=== BS=${X} done ==="
done

echo "Sweep complete. Reports under ${RESULTS_ROOT}"
