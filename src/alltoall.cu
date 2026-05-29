/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mc_runtime_api.h"
#include "common.h"
extern bool isAlltoAll;

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount,
                              size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count,
                              int nranks)
{
    *sendcount         = (count / nranks) * nranks;
    *recvcount         = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount        = count / nranks;
}

testResult_t AlltoAllInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op,
                              int root, int rep, int in_place)
{
    size_t sendcount = args->sendBytes / wordSize(type);
    size_t recvcount = args->expectedBytes / wordSize(type);
    int nranks       = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        int rank = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
        void *data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
        for (int j = 0; j < nranks; j++) {
            size_t partcount = sendcount / nranks;
            TESTCHECK(InitData((char *)args->expected[i] + j * partcount * wordSize(type),
                               partcount, rank * partcount, type, ncclSum, 33 * rep + j, 1, 0));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
    // We don't support in-place alltoall
    args->reportErrors = in_place ? 0 : 1;
    isAlltoAll         = true;
    return testSuccess;
}

void AlltoAllGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / bwUnit / sec;

    *algBw        = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)(nranks));
    *busBw        = baseBw * factor;
}

static int mcclAlltoAllOpt = -1;
testResult_t AlltoAllRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type,
                             ncclRedOp_t op, int root, ncclComm_t comm, mcStream_t stream, struct extendCollArg* extendArgs=nullptr)
{
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);
    if (count == 0)
        return testSuccess;

    if (mcclAlltoAllOpt == -1) {
        if (getenv("MCCL_OPTIMIZATION_A2A")) {
            mcclAlltoAllOpt = atoi(getenv("MCCL_OPTIMIZATION_A2A"));
        } else {
            mcclAlltoAllOpt = 1;
        }
    }
#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n",
           NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
    NCCLCHECK(ncclGroupStart());
    if (mcclAlltoAllOpt == 1) {
        if (enable_extend_api)
            NCCLCHECK(mcclAllToAllExt(sendbuff, recvbuff, count, type, comm, stream));
        else
            NCCLCHECK(mcclAllToAll(sendbuff, recvbuff, count, type, comm, stream));
    } else {
        for (int r = 0; r < nRanks; r++) {
            if (enable_extend_api) {
                NCCLCHECK(
                    mcclSendExt(((char *)sendbuff) + r * rankOffset, count, type, r, comm, stream));
                NCCLCHECK(
                    mcclRecvExt(((char *)recvbuff) + r * rankOffset, count, type, r, comm, stream));
            } else {
                NCCLCHECK(
                    ncclSend(((char *)sendbuff) + r * rankOffset, count, type, r, comm, stream));
                NCCLCHECK(
                    ncclRecv(((char *)recvbuff) + r * rankOffset, count, type, r, comm, stream));
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
#endif
}

struct testColl alltoAllTest = {"AlltoAll", AlltoAllGetCollByteCount, AlltoAllInitData,
                                AlltoAllGetBw, AlltoAllRunColl};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset,
                             &recvInplaceOffset, count, nranks);
}

testResult_t AlltoAllRunTest(struct threadArgs *args, int root, ncclDataType_t type,
                             const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &alltoAllTest;
    ncclDataType_t *run_types;
    const char **run_typenames;
    int type_count;

    if ((int)type != -1) {
        type_count    = 1;
        run_types     = &type;
        run_typenames = &typeName;
    } else {
        type_count    = test_typenum;
        run_types     = test_types;
        run_typenames = test_typenames;
    }

    for (int i = 0; i < type_count; i++) {
        TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
    }
    return testSuccess;
}

struct testEngine alltoAllEngine = {AlltoAllGetBuffSize, AlltoAllRunTest};

// #pragma weak ncclTestEngine=alltoAllEngine
__attribute__((weak)) testEngine ncclTestEngine = alltoAllEngine;