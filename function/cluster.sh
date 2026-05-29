#!/bin/bash
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

MACA_PATH=/opt/maca

HOST_IP=10.2.122.225:8,10.2.122.226:8
IP_MASK=10.2.122.0/24
GPU_NUM=16

if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" ]]; then
  echo "Use the default ip addr. Run with parameters for custom ip addr, for example: bash cluster.sh ip_1 ip_2 ip_mask gpu_num"
else
  HOST_IP=$1,$2
  IP_MASK=$3
  GPU_NUM=$4
fi

IB_PORT=mlx5_0,mlx5_1

TEST_DIR=/opt/maca/samples/mccl_tests/perf/mccl_perf
#BENCH_NAMES="all_reduce_perf all_gather_perf reduce_scatter_perf sendrecv_perf alltoall_perf"
BENCH_NAMES="all_reduce_perf"

PERF_ENV="-x FORCE_ACTIVE_WAIT=2"
LIB_PATH_ENV="-x LD_LIBRARY_PATH=${MACA_PATH}/lib:/${MACA_PATH}/ompi/lib"
ENV_VAR="-x MCCL_IB_HCA=${IB_PORT} -x MCCL_CROSS_NIC=1 ${PERF_ENV} ${LIB_PATH_ENV}"

MPI_PROCESS_NUM=${GPU_NUM}
MPI_RUN_OPT="--allow-run-as-root -mca btl_tcp_if_include ${IP_MASK} -mca oob_tcp_if_include ${IP_MASK} -mca pml ^ucx -mca osc ^ucx -mca btl ^openib"

for BENCH in ${BENCH_NAMES}; do
echo -n "The test is ${BENCH}, the maca version is " && realpath ${MACA_PATH}
TEST="${TEST_DIR}/${BENCH} -b 1K -e 1G -d float -f 2 -g 1 -n 10"
${MACA_PATH}/ompi/bin/mpirun -np ${MPI_PROCESS_NUM} ${MPI_RUN_OPT} -host ${HOST_IP} ${ENV_VAR} /opt/maca/samples/mccl_tests/perf/function/per_rank.sh "${TEST}" 0
done
