#!/usr/bin/env bash
set -euo pipefail

MACA_PATH="${MACA_PATH:-/opt/maca}"
TEST_DIR="${TEST_DIR:-${MACA_PATH}/samples/mccl_tests/perf/mccl_perf}"
BENCH="${BENCH:-all_reduce_perf}"
GPU_NUM="${GPU_NUM:-1}"
MIN_BYTES="${MIN_BYTES:-1K}"
MAX_BYTES="${MAX_BYTES:-16M}"
ITERS="${ITERS:-2}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
DATATYPE="${DATATYPE:-bfloat16}"

export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export FORCE_ACTIVE_WAIT="${FORCE_ACTIVE_WAIT:-2}"

if [[ ! -x "${MACA_PATH}/ompi/bin/mpirun" ]]; then
  echo "mpirun not found under ${MACA_PATH}/ompi/bin" >&2
  exit 1
fi

if [[ ! -x "${TEST_DIR}/${BENCH}" ]]; then
  echo "benchmark not found: ${TEST_DIR}/${BENCH}" >&2
  exit 1
fi

echo "MACA_PATH=${MACA_PATH}"
echo "TEST_DIR=${TEST_DIR}"
echo "BENCH=${BENCH}"
echo "GPU_NUM=${GPU_NUM}"
echo "MIN_BYTES=${MIN_BYTES}"
echo "MAX_BYTES=${MAX_BYTES}"
echo "ITERS=${ITERS}"
echo "WARMUP_ITERS=${WARMUP_ITERS}"

"${MACA_PATH}/ompi/bin/mpirun" \
  -np "${GPU_NUM}" \
  --allow-run-as-root \
  -mca pml ^ucx \
  -mca osc ^ucx \
  -mca btl ^openib \
  "${TEST_DIR}/${BENCH}" \
  -b "${MIN_BYTES}" \
  -e "${MAX_BYTES}" \
  -d "${DATATYPE}" \
  -f 2 \
  -g 1 \
  -n "${ITERS}" \
  -w "${WARMUP_ITERS}"
