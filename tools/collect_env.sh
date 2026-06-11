#!/usr/bin/env bash
set -euo pipefail

MACA_PATH="${MACA_PATH:-/opt/maca}"
TEST_DIR="${TEST_DIR:-${MACA_PATH}/samples/mccl_tests/perf/mccl_perf}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

json_string() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}

command_output() {
  local cmd="$1"
  local exe="${cmd%% *}"
  local output
  if [[ -x "${exe}" ]] || command -v "${exe}" >/dev/null 2>&1; then
    if ! output="$(bash -c "${cmd}" 2>&1 | head -n 5)"; then
      :
    fi
    printf '%s' "${output}" | json_string
  else
    printf 'null'
  fi
}

cat <<JSON
{
  "maca_path": $(printf '%s' "${MACA_PATH}" | json_string),
  "test_dir": $(printf '%s' "${TEST_DIR}" | json_string),
  "maca_path_exists": $(if [[ -d "${MACA_PATH}" ]]; then echo true; else echo false; fi),
  "mpi_exists": $(if [[ -x "${MACA_PATH}/ompi/bin/mpirun" ]]; then echo true; else echo false; fi),
  "all_reduce_exists": $(if [[ -x "${TEST_DIR}/all_reduce_perf" ]]; then echo true; else echo false; fi),
  "mxcc_version": $(command_output "${MACA_PATH}/mxgpu_llvm/bin/mxcc --version"),
  "mpirun_version": $(command_output "${MACA_PATH}/ompi/bin/mpirun --version")
}
JSON
