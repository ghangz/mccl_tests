#!/bin/bash
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

export MACA_PATH=/opt/maca
export CUDA_PATH=${MACA_PATH}/tools/cu-bridge

cd src

make MPI=1 MPI_HOME=${MACA_PATH}/ompi CUDA_HOME=${MACA_PATH}/tools/cu-bridge
