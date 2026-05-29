#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# See LICENSE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/cucc
CUDARTLIB ?= cudart

CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
INSTALL_DIRS ?= ${BUILDROOT}/samples/mccl_tests
PERF_INSTALL_DIRS ?= ${INSTALL_DIRS}/perf

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_80,code=sm_80 \
                -gencode=arch=compute_80,code=compute_80
else
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_70,code=compute_70
endif

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11 -I$(CUDA_INC)
NVCUFLAGS  += --offload-arch=xcore1500 --offload-arch=xcore1000

LDFLAGS    := -L${CUDA_LIB} -lcudart -lrt
NVLDFLAGS  := -L${CUDA_LIB} -l${CUDARTLIB} -lrt -lcupti

ifdef PREBUILTLIB
NVCUFLAGS += -DJSON_SUPPORT -I$(PREBUILTLIB)/$(ARCH)/include/
NVLDFLAGS += -L$(PREBUILTLIB)/$(ARCH)/lib/ -ljsoncpp
endif

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3 -g
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif

.PHONY: build clean

BUILD_DIR ?= ../build
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILD_DIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILE ?= all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube alltoallv alltoalld
BIN_FILES_LIST := $(BIN_FILE)
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILD_DIR)/verifiable
include ../verifiable/verifiable.mk

$(info "NVCUFLAGS is ${NVCUFLAGS}")
$(info "NVLDFLAGS is ${NVLDFLAGS}")
${DST_DIR}/%.o: %.cu common.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o ${DST_DIR}/topoCheck.o $(TEST_VERIFIABLE_OBJS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

install:
	mkdir -p ${PERF_INSTALL_DIRS}/mccl_perf
	cp ${DST_DIR}/*_perf  ${PERF_INSTALL_DIRS}/mccl_perf/
	cp ${BUILD_DIR}/../README.md  ${PERF_INSTALL_DIRS}/
	cp ${BUILD_DIR}/../mccl.sh  ${PERF_INSTALL_DIRS}/
	cp ${BUILD_DIR}/../cluster.sh  ${PERF_INSTALL_DIRS}/
	cp ${BUILD_DIR}/../dragonfly.sh  ${PERF_INSTALL_DIRS}/
	cp -r ${BUILD_DIR}/../nccl_perf  ${PERF_INSTALL_DIRS}/
	cp -r ${BUILD_DIR}/../function  ${PERF_INSTALL_DIRS}/