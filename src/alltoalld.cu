
/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "mc_runtime_api.h"
#include "common.h"
#include <vector>

extern bool isAlltoAlld;
#define ARRAY_SHIFT_LEFT(arr1, arr2, nrank) {         \
    int tmp = arr2[0];                                \
    for (int i = 0; i < nrank - 1; i++) {             \
        arr2[i] = arr2[i + 1];                        \
        arr1[i] += arr2[i];                           \
    }                                                 \
    arr2[nrank - 1] = tmp;                            \
    arr1[nrank - 1] += arr2[nrank - 1];               \
}

void AlltoAlldGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset,
                               size_t *recvInplaceOffset, size_t count, int nranks)
{
    size_t splitcount  = nranks*(nranks+1)/2;
    *sendcount         = (count / nranks) * nranks;
    *recvcount         = (count / splitcount) * splitcount; // Represents the actual received data size of the rank.
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount        = count / nranks;
}
testResult_t AlltoAlldInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op,
                               int root, int rep, int in_place)
{
    size_t sendcount = args->sendBytes / wordSize(type);
    size_t recvcount = args->expectedBytes / wordSize(type);
    size_t count     = args->sendBytes / wordSize(type);
    int nranks       = args->nProcs*args->nThreads*args->nGpus;
    size_t splitcount = nranks*(nranks+1)/2;
    sendcount = recvcount = (sendcount / splitcount)*splitcount;
    count = (count / splitcount)*splitcount;
    for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
        CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
        void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
        std::vector<int> offset(nranks);
        std::vector<int> offsetcount(nranks);
        // compute expect offset
        for(int k=0;k<nranks;++k){
            offsetcount[k] = offset[k] = count / splitcount * (k+1);
        }
        for (int k=0; k<rank-1;++k) {
            ARRAY_SHIFT_LEFT(offsetcount, offset, nranks);
        }

        for (int j=0; j<nranks; j++) {
            args->collArgs[i]->sendcounts[j] = count*((rank+j)%nranks+1) / splitcount;
            args->collArgs[i]->sdispls[j] = (j == 0 ? 0:args->collArgs[i]->sendcounts[j-1] + args->collArgs[i]->sdispls[j-1]);
            args->collArgs[i]->sendbuffs[j] = (char *)data + args->collArgs[i]->sdispls[j]*wordSize(type);
            args->collArgs[i]->recvcounts[j] = count*((rank+j)%nranks+1) / splitcount;
            args->collArgs[i]->rdispls[j] = (j == 0 ? 0:args->collArgs[i]->recvcounts[j-1] + args->collArgs[i]->rdispls[j-1]);
            args->collArgs[i]->recvbuffs[j] = (char *)args->recvbuffs[i] + args->collArgs[i]->rdispls[j]*wordSize(type);
            int offsets = (rank==0) ? 0:offsetcount[j];
            TESTCHECK(InitData((char*)args->expected[i] + args->collArgs[i]->rdispls[j]*wordSize(type),
                               args->collArgs[i]->recvcounts[j], offsets, type, ncclSum, 33*rep + j, 1, 0));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
    // We don't support in-place alltoalld
    args->reportErrors = in_place ? 0 : 1;
    isAlltoAlld = true;
    return testSuccess;
}
void AlltoAlldGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / bwUnit / sec;
    *algBw        = baseBw;
    double factor = ((double)(nranks-1))/((double)(nranks));
    *busBw        = baseBw * factor;
}
static int mcclAlltoAlldOpt = -1;
testResult_t AlltoAlldRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op,
                              int root, ncclComm_t comm, mcStream_t stream, struct extendCollArg* extendArgs=nullptr)
{
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);
    if (count == 0)
        return testSuccess;
    if (mcclAlltoAlldOpt == -1) {
        if (getenv("MCCL_OPTIMIZATION_A2A")) {
            mcclAlltoAlldOpt = atoi(getenv("MCCL_OPTIMIZATION_A2A"));
        } else {
            mcclAlltoAlldOpt = 1;
        }
    }
#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for AlltoAlld. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
    NCCLCHECK(ncclGroupStart());
    if (mcclAlltoAlldOpt == 1) {
        if (enable_extend_api)
            NCCLCHECK(mcclAllToAlldExt((const void **)extendArgs->sendbuffs, extendArgs->sendcounts,
                                     extendArgs->recvbuffs, extendArgs->recvcounts, type, comm, stream));
        else
            NCCLCHECK(mcclAllToAlld((const void **)extendArgs->sendbuffs, extendArgs->sendcounts,
                                     extendArgs->recvbuffs, extendArgs->recvcounts, type, comm, stream));
    } else {
        for (int r=0; r<nRanks; r++) {
            if (enable_extend_api) {
                NCCLCHECK(mcclSendExt(extendArgs->sendbuffs[r], extendArgs->sendcounts[r], type, r, comm, stream));
                NCCLCHECK(mcclRecvExt(extendArgs->recvbuffs[r], extendArgs->recvcounts[r], type, r, comm, stream));
            } else {
                NCCLCHECK(mcclSend(extendArgs->sendbuffs[r], extendArgs->sendcounts[r], type, r, comm, stream));
                NCCLCHECK(mcclRecv(extendArgs->recvbuffs[r], extendArgs->recvcounts[r], type, r, comm, stream));
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
#endif
}
struct testColl AlltoAlldTest = {"AlltoAlld", AlltoAlldGetCollByteCount, AlltoAlldInitData,
                                AlltoAlldGetBw, AlltoAlldRunColl};

void AlltoAlldGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AlltoAlldGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AlltoAlldRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName)
{
    args->collTest = &AlltoAlldTest;
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
    for (int i=0; i<type_count; i++) {
        TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
    }
    return testSuccess;
}
struct testEngine AlltoAlldEngine = {AlltoAlldGetBuffSize, AlltoAlldRunTest};

// #pragma weak ncclTestEngine=AlltoAlldEngine
__attribute__((weak)) testEngine ncclTestEngine=AlltoAlldEngine;