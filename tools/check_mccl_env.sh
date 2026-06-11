#!/usr/bin/env bash
set -euo pipefail

MACA_PATH="${MACA_PATH:-/opt/maca}"
TEST_DIR="${MACA_PATH}/samples/mccl_tests/perf/mccl_perf"

exit_code=0
check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "[OK] ${label}: ${path}"
  else
    echo "[MISS] ${label}: ${path}"
    exit_code=1
  fi
}

check_path "MACA_PATH" "${MACA_PATH}"
check_path "MACA library path" "${MACA_PATH}/lib"
check_path "OMPI library path" "${MACA_PATH}/ompi/lib"
check_path "mpirun" "${MACA_PATH}/ompi/bin/mpirun"
check_path "mccl perf directory" "${TEST_DIR}"
check_path "render devices" "/dev/dri"

for bench in all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf alltoall_perf; do
  check_path "${bench}" "${TEST_DIR}/${bench}"
done

exit "${exit_code}"
