#!/bin/bash
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

VISIBLE_DEVICE_PER_RANK=1
MAX_GPUS_PER_NODE=8

if [[ -n "$2" ]]; then
  VISIBLE_DEVICE_PER_RANK=$2
fi

rank=${OMPI_COMM_WORLD_RANK}
node=$(expr ${rank} / ${MAX_GPUS_PER_NODE})
if [[ ${VISIBLE_DEVICE_PER_RANK} -eq 1 ]]; then
  export NCCL_TESTS_DEVICE=0
  export MACA_VISIBLE_DEVICES=$(expr ${rank} % ${MAX_GPUS_PER_NODE})
else
  isOdd=$(expr ${node} % 2)
  if [[ ${isOdd} -eq 0 ]]; then
    export MACA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  else
    export MACA_VISIBLE_DEVICES=5,7,6,4,3,0,2,1
  fi
fi

echo "Rank ${rank} Node ${node} on ${HOSTNAME} MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES}"
$1
