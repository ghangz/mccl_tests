#!/bin/bash
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

MACA_PATH="${MACA_PATH:-/opt/maca}"

HOST_IP=10.2.122.225:8,10.2.122.226:8
GPU_NUM=16

TEST_DIR=$MACA_PATH/samples/mccl_tests/perf/mccl_perf
#BENCH_NAMES="all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf alltoall_perf"
BENCH_NAMES="all_reduce_perf"

if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
  echo "Use the default ip addr. Run with parameters for custom ip addr, for example: bash cluster.sh ip_1:proc_count,ip_2:proc_count gpu_num test_name"
else
  HOST_IP=$1
  GPU_NUM=$2

  if [ "$3" = "all" ]; then
    BENCH_NAMES="all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf alltoall_perf"
  else
    if [ -e "$TEST_DIR/$3" ]; then
      BENCH_NAMES=$3
    else
      echo "$TEST_DIR/$3 dose not exist!"
      exit 1
    fi
  fi
fi

IP_MASK="$(echo "$HOST_IP" | cut -d. -f1-3).0/24"
IB_PORT=mlx5_0,mlx5_1

PERF_ENV="-x FORCE_ACTIVE_WAIT=2"
LIB_PATH_ENV="-x MACA_PATH=${MACA_PATH} -x LD_LIBRARY_PATH=${MACA_PATH}/lib:/${MACA_PATH}/ompi/lib:/${MACA_PATH}/ucx/lib"
ENV_VAR="-x MCCL_IB_HCA=${IB_PORT} -x MCCL_CROSS_NIC=1 ${PERF_ENV} ${LIB_PATH_ENV}"

MPI_PROCESS_NUM=${GPU_NUM}
MPI_RUN_OPT="--allow-run-as-root -mca btl_tcp_if_include ${IP_MASK} -mca oob_tcp_if_include ${IP_MASK} -mca pml ^ucx -mca osc ^ucx -mca btl ^openib"

for BENCH in ${BENCH_NAMES}; do
  echo -n "The test is ${BENCH}, the maca version is " && realpath ${MACA_PATH}
  ${MACA_PATH}/ompi/bin/mpirun -np ${MPI_PROCESS_NUM} ${MPI_RUN_OPT} -host ${HOST_IP} ${ENV_VAR} ${TEST_DIR}/${BENCH} -b 1K -e 1G -d float -f 2 -g 1 -n 10
done
