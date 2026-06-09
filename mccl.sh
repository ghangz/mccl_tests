#!/bin/bash
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib

export FORCE_ACTIVE_WAIT=2

GPU_NUM=4
if [[ $1 -gt 0 && $1 -lt 65 ]]; then
  GPU_NUM=$1
fi

TEST_DIR=${MACA_PATH}/samples/mccl_tests/perf/mccl_perf
#BENCH_NAMES="all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf alltoall_perf"
BENCH_NAMES=all_reduce_perf

if [ -n "$2" ]; then
  if [ "$2" = "all" ]; then
    BENCH_NAMES="all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf alltoall_perf"
  else
    if [[ "${MCCL_DRY_RUN:-0}" = "1" || -e "$TEST_DIR/$2" ]]; then
      BENCH_NAMES=$2
    else
      echo "$TEST_DIR/$2 dose not exist!"
      exit 1
    fi
  fi
fi

MPI_PROCESS_NUM=${GPU_NUM}
MPI_RUN_OPT="--allow-run-as-root -mca pml ^ucx -mca osc ^ucx -mca btl ^openib"

for BENCH in ${BENCH_NAMES}; do
echo -n "The test is ${BENCH}, the maca version is " && realpath ${MACA_PATH}
CMD="${MACA_PATH}/ompi/bin/mpirun -np ${MPI_PROCESS_NUM} ${MPI_RUN_OPT} ${TEST_DIR}/${BENCH} -b 1K -e 1G -d bfloat16 -f 2 -g 1 -n 10"
echo "${CMD}"
if [[ "${MCCL_DRY_RUN:-0}" = "1" ]]; then
  continue
fi
${CMD}
done
