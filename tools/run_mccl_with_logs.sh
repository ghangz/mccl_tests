#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_NUM="${1:-4}"
BENCH_NAME="${2:-all_reduce_perf}"
LOG_ROOT="${MCCL_LOG_DIR:-${ROOT_DIR}/logs}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${BENCH_NAME}_g${GPU_NUM}"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"

mkdir -p "${RUN_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root_dir=${ROOT_DIR}"
  echo "gpu_num=${GPU_NUM}"
  echo "bench_name=${BENCH_NAME}"
  echo "maca_path=${MACA_PATH:-/opt/maca}"
  echo "ld_library_path=${LD_LIBRARY_PATH:-}"
  echo "dry_run=${MCCL_DRY_RUN:-0}"
} > "${RUN_DIR}/metadata.env"

set +e
bash "${ROOT_DIR}/mccl.sh" "${GPU_NUM}" "${BENCH_NAME}" > "${RUN_DIR}/stdout.log" 2> "${RUN_DIR}/stderr.log"
status=$?
set -e

echo "${status}" > "${RUN_DIR}/exit_code.txt"
echo "MCCL logs written to: ${RUN_DIR}"
exit "${status}"
