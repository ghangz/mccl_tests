/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "topoCheck.h"
#include <pthread.h>
#include <cstdio>
#include <type_traits>
#include <getopt.h>
#include <libgen.h>
#include "maca_fp16.h"
#include "__clang_maca_runtime_wrapper.h"
#include "../verifiable/verifiable.h"
#include "profiler.h"
#include <pwd.h>
#include <unistd.h>
#include <mutex>

#ifdef JSON_SUPPORT
#include <json/json.h>
#endif

int test_ncclVersion    = 0; // init'd with ncclGetVersion()
bool enable_extend_api  = false;
int ENHANCE_TEST        = 0; // enhance test switch
static int nProcess     = 1;
static float lat_4k     = 0;
static float bw_1G      = 0;
static float tolerance  = 0;
static bool find_proper = false;
static std::mutex getLatBw_mtx;

//  metaTopo process thread gpu_perthread  lat(msg_block=4k)  bw(msg_block=1G)
std::vector<topoExpectPerformance_t> topoExpectPerformanceTable_COMMON;
std::vector<topoExpectPerformance_t> topoExpectPerformanceTable_DRAGONFLY;
std::vector<topoExpectPerformance_t> topoExpectPerformanceTable_OAM;
std::vector<topoExpectPerformance_t> topoExpectPerformanceTable_CASCADE;
static char *mcclDefaultDebugFile;

#if NCCL_MAJOR >= 2
ncclDataType_t test_types[ncclNumTypes] = {ncclInt8,
                                           ncclUint8,
                                           ncclInt32,
                                           ncclUint32,
                                           ncclInt64,
                                           ncclUint64,
                                           ncclHalf,
                                           ncclFloat,
                                           ncclDouble
#if defined(__CUDA_BF16_TYPES_EXIST__)
                                           ,
                                           ncclBfloat16
#endif
};
const char *test_typenames[ncclNumTypes] = {"int8",
                                            "uint8",
                                            "int32",
                                            "uint32",
                                            "int64",
                                            "uint64",
                                            "half",
                                            "float",
                                            "double"
#if defined(__CUDA_BF16_TYPES_EXIST__)
                                            ,
                                            "bfloat16"
#endif
};
int test_typenum = -1;

const char *test_opnames[] = {"sum", "prod", "max", "min", "avg", "mulsum"};
ncclRedOp_t test_ops[]     = {
    ncclSum,
    ncclProd,
    ncclMax,
    ncclMin
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    ,
    ncclAvg
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
    ,
    ncclNumOps // stand in for ncclRedOpCreatePreMulSum() created on-demand
#endif
};
int test_opnum = -1;
#else
ncclDataType_t test_types[ncclNumTypes]  = {ncclChar,   ncclInt,   ncclHalf,  ncclFloat,
                                           ncclDouble, ncclInt64, ncclUint64};
const char *test_typenames[ncclNumTypes] = {"char",   "int",   "half",  "float",
                                            "double", "int64", "uint64"};
int test_typenum                         = 7;
const char *test_opnames[]               = {"sum", "prod", "max", "min"};
ncclRedOp_t test_ops[]                   = {ncclSum, ncclProd, ncclMax, ncclMin};
int test_opnum                           = 4;
#endif
#define MCCL_LOGFILE_PREFIX "/tmp/result"
#define MCCL_LOGFILE_END    ".txt"
int is_main_proc                = 0;
thread_local int is_main_thread = 0;
thread_local double latency     = 0;
thread_local double high_algBw  = 0;
thread_local double high_busBw  = 0;
bool isSendRecv                 = false;
bool isAlltoAll                 = false;
bool isAlltoAllv                = false;
bool isAlltoAlld                = false;

// Command line parameter defaults
static int nThreads          = 1;
static int nGpus             = 1;
static size_t minBytes       = 32 * 1024 * 1024;
static size_t maxBytes       = 32 * 1024 * 1024;
static size_t stepBytes      = 1 * 1024 * 1024;
static size_t stepFactor     = 1;
static int datacheck         = 1;
static int warmup_iters      = 5;
static int iters             = 20;
static int agg_iters         = 1;
static int ncclop            = ncclSum;
static int nccltype          = ncclFloat;
static int ncclroot          = 0;
static int parallel_init     = 0;
static int blocking_coll     = 0;
static int cudaGraphLaunches = 0;
static int useVmmMemory      = 0;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 1;

double bwUnit = 1.0E9;
char bwUnitStr[8] = {0};

#define NUM_BLOCKS 32

static double parsesize(const char *value)
{
    long long int units;
    double size;
    char size_lit;

    int count = sscanf(value, "%lf %c", &size, &size_lit);

    switch (count) {
    case 2:
        switch (size_lit) {
        case 'G':
        case 'g':
            units = 1024 * 1024 * 1024;
            break;
        case 'M':
        case 'm':
            units = 1024 * 1024;
            break;
        case 'K':
        case 'k':
            units = 1024;
            break;
        default:
            return -1.0;
        };
        break;
    case 1:
        units = 1;
        break;
    default:
        return -1.0;
    }

    return size * units;
}

testResult_t CheckDelta(void *results, void *expected, size_t count, size_t offset,
                        ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks,
                        int64_t *wrongEltN)
{
    ncclVerifiableVerify(results, expected, count, (int)type, (int)op, nranks, seed, offset,
                         wrongEltN, cudaStreamDefault);
    CUDACHECK(cudaDeviceSynchronize());
    return testSuccess;
}

testResult_t InitDataReduce(void *data, const size_t count, const size_t offset,
                            ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks)
{
    ncclVerifiablePrepareExpected(data, count, (int)type, (int)op, nranks, seed, offset,
                                  cudaStreamDefault);
    return testSuccess;
}

testResult_t InitData(void *data, const size_t count, size_t offset, ncclDataType_t type,
                      ncclRedOp_t op, uint64_t seed, int nranks, int rank)
{
    ncclVerifiablePrepareInput(data, count, (int)type, (int)op, nranks, rank, seed, offset,
                               cudaStreamDefault);
    return testSuccess;
}

void Barrier(struct threadArgs *args)
{
    thread_local int epoch         = 0;
    static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
    static pthread_cond_t cond[2]  = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
    static int counter[2]          = {0, 0};

    pthread_mutex_lock(&lock[epoch]);
    if (++counter[epoch] == args->nThreads)
        pthread_cond_broadcast(&cond[epoch]);

    if (args->thread + 1 == args->nThreads) {
        while (counter[epoch] != args->nThreads)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
#ifdef MPI_SUPPORT
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        counter[epoch] = 0;
        pthread_cond_broadcast(&cond[epoch]);
    } else {
        while (counter[epoch] != 0)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
    }
    pthread_mutex_unlock(&lock[epoch]);
    epoch ^= 1;
}

// Inter-thread/process barrier+allreduce. The quality of the return value
// for average=0 (which means broadcast from rank=0) is dubious. The returned
// value will actually be the result of process-local broadcast from the local thread=0.
template <typename T> void Allreduce(struct threadArgs *args, T *value, int average)
{
    thread_local int epoch         = 0;
    static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
    static pthread_cond_t cond[2]  = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
    static T accumulator[2];
    static int counter[2] = {0, 0};

    pthread_mutex_lock(&lock[epoch]);
    if (counter[epoch] == 0) {
        if (average != 0 || args->thread == 0)
            accumulator[epoch] = *value;
    } else {
        switch (average) {
        case /*r0*/ 0:
            if (args->thread == 0)
                accumulator[epoch] = *value;
            break;
        case /*avg*/ 1:
            accumulator[epoch] += *value;
            break;
        case /*min*/ 2:
            accumulator[epoch] = std::min<T>(accumulator[epoch], *value);
            break;
        case /*max*/ 3:
            accumulator[epoch] = std::max<T>(accumulator[epoch], *value);
            break;
        case /*sum*/ 4:
            accumulator[epoch] += *value;
            break;
        }
    }

    if (++counter[epoch] == args->nThreads)
        pthread_cond_broadcast(&cond[epoch]);

    if (args->thread + 1 == args->nThreads) {
        while (counter[epoch] != args->nThreads)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);

#ifdef MPI_SUPPORT
        if (average != 0) {
            static_assert(std::is_same<T, long long>::value || std::is_same<T, double>::value,
                          "Allreduce<T> only for T in {long long, double}");
            MPI_Datatype ty = std::is_same<T, long long>::value ? MPI_LONG_LONG
                              : std::is_same<T, double>::value  ? MPI_DOUBLE
                                                                : MPI_Datatype();
            MPI_Op op       = average == 1   ? MPI_SUM
                              : average == 2 ? MPI_MIN
                              : average == 3 ? MPI_MAX
                              : average == 4 ? MPI_SUM
                                             : MPI_Op();
            MPI_Allreduce(MPI_IN_PLACE, (void *)&accumulator[epoch], 1, ty, op, MPI_COMM_WORLD);
        }
#endif

        if (average == 1)
            accumulator[epoch] /= args->totalProcs * args->nThreads;
        counter[epoch] = 0;
        pthread_cond_broadcast(&cond[epoch]);
    } else {
        while (counter[epoch] != 0)
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
    }
    pthread_mutex_unlock(&lock[epoch]);

    *value = accumulator[epoch];
    epoch ^= 1;
}

testResult_t CheckData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root,
                       int in_place, int64_t *wrongElts)
{
    int nranks   = args->nProcs * args->nGpus * args->nThreads;
    size_t count = args->expectedBytes / wordSize(type);

    int64_t *wrongPerGpu = nullptr;
    CUDACHECK(
        cudaHostAlloc((void **)&wrongPerGpu, args->nGpus * sizeof(int64_t), cudaHostAllocMapped));

    for (int i = 0; i < args->nGpus; i++) {
        int rank = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        void *data =
            in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset * rank))
                     : args->recvbuffs[i];

        TESTCHECK(
            CheckDelta(data, args->expected[i], count, 0, type, op, 0, nranks, wrongPerGpu + i));

        const char *debug_error = getenv("DEBUG_PRINT");
        if ((NULL != debug_error) && (0 == strcmp(debug_error, "ON"))) {
            if ((isSendRecv || isAlltoAll || isAlltoAllv || isAlltoAlld) && in_place) {
                continue;
            }
            if (args->reportErrors && wrongPerGpu[i] != 0) {
                printf("rank=%d #wrong=%d\n", rank, (int)wrongPerGpu[i]);
                char *expectedHost = (char *)malloc(args->expectedBytes);
                char *dataHost     = (char *)malloc(args->expectedBytes);
                int eltsz          = wordSize(type);
                cudaMemcpy(expectedHost, args->expected[i], args->expectedBytes,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(dataHost, data, args->expectedBytes, cudaMemcpyDeviceToHost);

                for (int j = 0; j < args->expectedBytes / eltsz; j++) {
                    unsigned long long want, got;
                    want = 0;
                    memcpy(&want, expectedHost + j * eltsz, eltsz);
                    got = 0;
                    memcpy(&got, dataHost + j * eltsz, eltsz);
                    if (want != got) {
                        printf(" rank=%d elt[%d]: want=0x%llx got=0x%llx\n", rank, j, want, got);
                    }
                }
                free(expectedHost);
                free(dataHost);
            }
        }
    }

    *wrongElts = 0;
    for (int i = 0; i < args->nGpus; i++)
        *wrongElts += wrongPerGpu[i];
    cudaFreeHost(wrongPerGpu);

    if (args->reportErrors && *wrongElts)
        args->errors[0]++;
    return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, mcStream_t *streams, ncclComm_t *comms)
{
    cudaError_t cudaErr;
    int remaining = ngpus;
    int *done     = (int *)malloc(sizeof(int) * ngpus);
    memset(done, 0, sizeof(int) * ngpus);
    while (remaining) {
        int idle = 1;
        for (int i = 0; i < ngpus; i++) {
            if (done[i])
                continue;

            cudaErr = cudaStreamQuery(streams[i]);
            if (cudaErr == cudaSuccess) {
                done[i] = 1;
                remaining--;
                idle = 0;
                continue;
            }

            if (cudaErr != cudaErrorNotReady)
                CUDACHECK(cudaErr);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 4, 0)
            if (test_ncclVersion >= NCCL_VERSION(2, 4, 0) && comms) {
                ncclResult_t ncclAsyncErr;
                NCCLCHECK(ncclCommGetAsyncError(comms[i], &ncclAsyncErr));
                if (ncclAsyncErr != ncclSuccess) {
                    // An asynchronous error happened. Stop the operation and destroy
                    // the communicator
                    for (int i = 0; i < ngpus; i++)
                        NCCLCHECK(ncclCommAbort(comms[i]));
                    // Abort the perf test
                    NCCLCHECK(ncclAsyncErr);
                }
            }
#endif
        }

        // We might want to let other threads (including NCCL threads) use the CPU.
        if (idle)
            sched_yield();
    }
    free(done);
    return testSuccess;
}

testResult_t startColl(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t opIndex, int root,
                       int in_place, int iter)
{
    size_t count = args->nbytes / wordSize(type);

    // Try to change offset for each iteration so that we avoid cache effects and catch race
    // conditions in ptrExchange
    size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
    size_t steps       = totalnbytes ? args->maxbytes / totalnbytes : 1;
    size_t shift       = totalnbytes * (iter % steps);

    if (args->nGpus > 1)
        NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
        CUDACHECK(cudaSetDevice(args->gpus[i]));
#endif
        int rank       = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        char *recvBuff = ((char *)args->recvbuffs[i]) + shift;
        char *sendBuff = ((char *)args->sendbuffs[i]) + shift;
        ncclRedOp_t op;

        if (opIndex < ncclNumOps) {
            op = opIndex;
        }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
        else {
            union {
                int8_t i8;
                uint8_t u8;
                int32_t i32;
                uint32_t u32;
                int64_t i64;
                uint64_t u64;
                half f16;
                float f32;
                double f64;
#if defined(__CUDA_BF16_TYPES_EXIST__)
                __nv_bfloat16 bf16;
#endif
            };
            switch (type) {
            case ncclInt8:
                i8 = ncclVerifiablePremulScalar<int8_t>(rank);
                break;
            case ncclUint8:
                u8 = ncclVerifiablePremulScalar<uint8_t>(rank);
                break;
            case ncclInt32:
                i32 = ncclVerifiablePremulScalar<int32_t>(rank);
                break;
            case ncclUint32:
                u32 = ncclVerifiablePremulScalar<uint32_t>(rank);
                break;
            case ncclInt64:
                i64 = ncclVerifiablePremulScalar<int64_t>(rank);
                break;
            case ncclUint64:
                u64 = ncclVerifiablePremulScalar<uint64_t>(rank);
                break;
            case ncclFloat16:
                f16 = ncclVerifiablePremulScalar<half>(rank);
                break;
            case ncclFloat32:
                f32 = ncclVerifiablePremulScalar<float>(rank);
                break;
            case ncclFloat64:
                f64 = ncclVerifiablePremulScalar<double>(rank);
                break;
#if defined(__CUDA_BF16_TYPES_EXIST__)
            case ncclBfloat16:
                bf16 = ncclVerifiablePremulScalar<__nv_bfloat16>(rank);
                break;
#endif
            default:
                break;
            }
            NCCLCHECK(
                ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate, args->comms[i]));
        }
#endif

        TESTCHECK(args->collTest->runColl(
            (void *)(in_place ? recvBuff + args->sendInplaceOffset * rank : sendBuff),
            (void *)(in_place ? recvBuff + args->recvInplaceOffset * rank : recvBuff), count, type,
            op, root, args->comms[i], args->streams[i], args->collArgs[i]));

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
        if (opIndex >= ncclNumOps) {
            NCCLCHECK(ncclRedOpDestroy(op, args->comms[i]));
        }
#endif
    }
    if (args->nGpus > 1)
        NCCLCHECK(ncclGroupEnd());

    if (blocking_coll) {
        // Complete op before returning
        TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
    }
    if (blocking_coll)
        Barrier(args);
    return testSuccess;
}

testResult_t completeColl(struct threadArgs *args)
{
    if (blocking_coll)
        return testSuccess;

    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
    return testSuccess;
}

testResult_t BenchTime(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root,
                       int in_place, int size_id, info_log &persize)
{
    size_t count = args->nbytes / wordSize(type);
    if (datacheck) {
        // Initialize sendbuffs, recvbuffs and expected
        TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
    }

    // Sync
    TESTCHECK(startColl(args, type, op, root, in_place, 0));
    TESTCHECK(completeColl(args));

    Barrier(args);

#if CUDART_VERSION >= 11030
    std::vector<cudaGraph_t> graphs(args->nGpus);
    std::vector<cudaGraphExec_t> graphExec(args->nGpus);
    if (cudaGraphLaunches >= 1) {
        // Begin cuda graph capture
        for (int i = 0; i < args->nGpus; i++) {
            // Thread local mdoe is needed for:
            // - Multi-thread mode: where graph capture and instantiation can happen concurrently
            // across threads
            // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during graph
            // capture.
            //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
            CUDACHECK(cudaStreamBeginCapture(args->streams[i], cudaStreamCaptureModeThreadLocal));
        }
    }
#endif

    // Performance Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        if (agg_iters > 1)
            NCCLCHECK(ncclGroupStart());
        for (int aiter = 0; aiter < agg_iters; aiter++) {
            TESTCHECK(startColl(args, type, op, root, in_place, iter * agg_iters + aiter));
        }
        if (agg_iters > 1)
            NCCLCHECK(ncclGroupEnd());
    }

#if CUDART_VERSION >= 11030
    if (cudaGraphLaunches >= 1) {
        // End cuda graph capture
        for (int i = 0; i < args->nGpus; i++) {
            CUDACHECK(cudaStreamEndCapture(args->streams[i], &graphs[i]));
        }
        // Instantiate cuda graph
        for (int i = 0; i < args->nGpus; i++) {
            CUDACHECK(cudaGraphInstantiate(&graphExec[i], graphs[i], NULL, NULL, 0));
        }
        // Resync CPU, restart timing, launch cuda graph
        Barrier(args);
        start = std::chrono::high_resolution_clock::now();
        for (int l = 0; l < cudaGraphLaunches; l++) {
            for (int i = 0; i < args->nGpus; i++) {
                CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
            }
        }
    }
#endif

    TESTCHECK(completeColl(args));

    auto delta      = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    deltaSec        = deltaSec / (iters * agg_iters);
    if (cudaGraphLaunches >= 1)
        deltaSec = deltaSec / cudaGraphLaunches;
    Allreduce(args, &deltaSec, average);

#if CUDART_VERSION >= 11030
    if (cudaGraphLaunches >= 1) {
        // destroy cuda graph
        for (int i = 0; i < args->nGpus; i++) {
            CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
            CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
    }
#endif

    double algBw, busBw;
    args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw,
                          args->nProcs * args->nThreads * args->nGpus);

    Barrier(args);

    int64_t wrongElts       = 0;
    static __thread int rep = 0;
    rep++;
    if (datacheck) {
        // Initialize sendbuffs, recvbuffs and expected
        TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

#if CUDART_VERSION >= 11030
        if (cudaGraphLaunches >= 1) {
            // Begin cuda graph capture for data check
            for (int i = 0; i < args->nGpus; i++) {
                CUDACHECK(cudaStreamBeginCapture(
                    args->streams[i], args->nThreads > 1 ? cudaStreamCaptureModeThreadLocal
                                                         : cudaStreamCaptureModeGlobal));
            }
        }
#endif

        // test validation in single itertion, should ideally be included into the multi-iteration
        // run
        TESTCHECK(startColl(args, type, op, root, in_place, 0));

#if CUDART_VERSION >= 11030
        if (cudaGraphLaunches >= 1) {
            // End cuda graph capture
            for (int i = 0; i < args->nGpus; i++) {
                CUDACHECK(cudaStreamEndCapture(args->streams[i], &graphs[i]));
            }
            // Instantiate cuda graph
            for (int i = 0; i < args->nGpus; i++) {
                CUDACHECK(cudaGraphInstantiate(&graphExec[i], graphs[i], NULL, NULL, 0));
            }
            // Launch cuda graph
            for (int i = 0; i < args->nGpus; i++) {
                CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
            }
        }
#endif

        TESTCHECK(completeColl(args));

#if CUDART_VERSION >= 11030
        if (cudaGraphLaunches >= 1) {
            // destroy cuda graph
            for (int i = 0; i < args->nGpus; i++) {
                CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
                CUDACHECK(cudaGraphDestroy(graphs[i]));
            }
        }
#endif

        TESTCHECK(CheckData(args, type, op, root, in_place, &wrongElts));

        // aggregate delta from all threads and procs
        long long wrongElts1 = wrongElts;
        Allreduce(args, &wrongElts1, /*sum*/ 4);
        wrongElts = wrongElts1;
    }

    int mcptiFlag           = false;
    const char *enableMcpti = getenv("MX_TRACER_ENABLED_MCPTI");
    if ((NULL != enableMcpti) && (0 == strcmp(enableMcpti, "ON"))) {
        mcptiFlag = true;
    }

    FILE *fp2;
    if (is_main_thread)
        fp2 = fopen(mcclDefaultDebugFile, "a+");
    double timeUsec = deltaSec * 1.0E6;
    char timeStr[100];
    if (timeUsec >= 10000.0) {
        sprintf(timeStr, "%9.2f", timeUsec);
    } else if (timeUsec >= 100.0) {
        sprintf(timeStr, "%8.2f", timeUsec);
    } else {
        sprintf(timeStr, "%7.2f", timeUsec);
    }
    if ((NULL != debugLog) && (0 == strcmp(debugLog, "ON"))) {
        printf("getpid=%d,is_main_thread=%d,in_place=%d,timeUsec=%7.2f\n", getpid(), is_main_thread,
               in_place, timeUsec);
    }
    if (is_main_thread) {
        if (in_place) {
            persize.inTime  = std::string(timeStr);
            persize.inAlgbw = algBw;
            persize.inBusbw = busBw;
            persize.inWrong = wrongElts;
        } else {
            persize.outTime  = std::string(timeStr);
            persize.outAlgbw = algBw;
            persize.outBusbw = busBw;
            persize.outWrong = wrongElts;
        }
    }
    if (args->reportErrors) {
        FPRINTF(fp2,
                "  -                                                  %10s  %6.5f  %6.5f  %5g \n",
                timeStr, algBw, busBw, (double)wrongElts);
        if (mcptiFlag == false)
            sprintf(in_place == true ? args->resultInPlace : args->resultOutPlace,
                    "%10s  %6.2f  %6.2f  %5g", timeStr, algBw, busBw, (double)wrongElts);
    } else {
        FPRINTF(fp2,
                " --                                                                               "
                "%10s  %6.2f  %6.2f  %5s \n",
                timeStr, algBw, busBw, "N/A");
        if (mcptiFlag == false)
            sprintf(in_place == true ? args->resultInPlace : args->resultOutPlace,
                    "%10s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");
    }

    if (is_main_thread && (in_place == false)) {
        high_algBw = std::fmax(high_algBw, algBw);
        high_busBw = std::fmax(high_busBw, busBw);
        if (args->nbytes == (1 << 12)) { // 4KB
            latency = timeUsec;
        }
    }

    if (is_main_thread)
        fclose(fp2);
    args->bw[0] += busBw;
    args->bw_count[0]++;
    if (ENHANCE_TEST && find_proper && in_place == 0) {
        float lat = std::stof(timeStr);
        if (persize.size == 4096 && lat > lat_4k * (1 + tolerance)) {
            PRINT("# expected lat is %.0fus, current lat is %fus. NOT PASS\n", lat_4k, lat);
            return testInternalError;
        }

        if (persize.size == 1073741824 && busBw < bw_1G * (1 - tolerance)) {
            PRINT("# expected bw is %.0fGB/s, current bw is %fGB/s. NOT PASS\n", bw_1G, busBw);
            return testInternalError;
        }
    }
    return testSuccess;
}

void setupArgs(size_t size, ncclDataType_t type, struct threadArgs *args)
{
    int nranks = args->nProcs * args->nGpus * args->nThreads;
    size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

    count = size / wordSize(type);
    args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset,
                                     &recvInplaceOffset, (size_t)count, (size_t)nranks);

    args->nbytes            = paramCount * wordSize(type);
    args->sendBytes         = sendCount * wordSize(type);
    args->expectedBytes     = recvCount * wordSize(type);
    args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
    args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

#ifdef JSON_SUPPORT
void getLatBw(const char* caseName)
{
    std::lock_guard<std::mutex> lock(getLatBw_mtx);
    const char *platform = getenv("TEST_PLATFORM");
    if ((NULL != platform) && (0 == strcmp(platform, "n300_chip")))
        return;
    if (ENHANCE_TEST && !find_proper) {
        // get expected performance data from enhance_test.json
        char cwd[1024];
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        getcwd(cwd, sizeof(cwd));
        std::string full_path = std::string(cwd) + "/configs/enhance_test.json";
        std::ifstream file(full_path);
        if (!file.is_open()) {
            printf("rank %d can't open enhance_test.json\n", rank);
            return;
        }
        Json::CharReaderBuilder readerBuilder;
        Json::Value root;
        std::string errs;

        if (!Json::parseFromStream(readerBuilder, file, &root, &errs)) {
            PRINT("parse enhance_test.json failed\n");
            return;
        }

        file.close();

        tolerance = root["tolerance"].asFloat();
        tolerance = tolerance / 100;

        const char *onlyTestMetaxEth = getenv("ONLY_TEST_METAX_ETH");
        if ((NULL != onlyTestMetaxEth) && (0 == strcmp(onlyTestMetaxEth, "1"))) {
            // get bw and lat for c600/n300 2 nodes test of metax eth rdma function
            std::vector<topoExpectPerformanceEth_t> topoExpectPerformanceTable;
            Json::Value data = root["expected"];
            for (int i = 0; i < data.size(); i++) {
                Json::Value elem = data[i];
                topoExpectPerformanceTable.push_back({elem["collective"].asString(),
                                                      elem["rank"].asUInt(), elem["lat"].asFloat(),
                                                      elem["bw"].asFloat()});
            }

            for (std::vector<topoExpectPerformanceEth_t>::iterator it =
                     topoExpectPerformanceTable.begin();
                 it != topoExpectPerformanceTable.end(); it++) {
                topoExpectPerformanceEth_t tmp = *it;
                if (tmp.case_name == caseName && tmp.rank == nProcess * nThreads * nGpus) {
                    printf("rank %d finded proper data\n", rank);
                    find_proper = true;
                    lat_4k      = tmp.lat;
                    bw_1G       = tmp.bw;
                    break;
                }
            }
        } else {
            // get bw and lat for c500 single node
            std::vector<topoExpectPerformance_t> *topoExpectPerformanceTable = nullptr;
            // checktopo and get lat and bw
            PcieTopo_t pcie_topo;
            MetaxTopo_t metax_topo;
            ncclResult_t res = getMetaxTopo(&pcie_topo, &metax_topo);
            if (res != ncclSuccess)
                return;
            if (pcie_topo == PCIE_TOPO_COMMON) {
                Json::Value data = root["expected"]["COMMON"];
                for (int i = 0; i < data.size(); i++) {
                    Json::Value elem = data[i];
                    topoExpectPerformanceTable_COMMON.push_back(
                        {(MetaxTopo_t)elem["topo"].asUInt(), elem["process"].asUInt(),
                         elem["thread"].asUInt(), elem["gpu"].asUInt(), elem["lat"].asFloat(),
                         elem["bw"].asFloat()});
                }
                topoExpectPerformanceTable = &topoExpectPerformanceTable_COMMON;
            }
            // TODO: other pcie topo

            for (std::vector<topoExpectPerformance_t>::iterator it =
                     topoExpectPerformanceTable->begin();
                 it != topoExpectPerformanceTable->end(); it++) {
                topoExpectPerformance_t tmp = *it;
                if (tmp.metax == metax_topo && tmp.process == nProcess && tmp.thread == nThreads &&
                    tmp.gpu_perthread == nGpus) {
                    printf("rank %d finded proper data\n", rank);
                    find_proper = true;
                    lat_4k      = tmp.lat;
                    bw_1G       = tmp.bw;
                    break;
                }
            }
        }
    }
}
#endif

testResult_t TimeTest(struct threadArgs *args, ncclDataType_t type, const char *typeName,
                      ncclRedOp_t op, const char *opName, int root)
{
    int mcptiFlag           = false;
#ifdef JSON_SUPPORT
    getLatBw(args->collTest->name);
#endif
    const char *enableMcpti = getenv("MX_TRACER_ENABLED_MCPTI");
    if ((NULL != enableMcpti) && (0 == strcmp(enableMcpti, "ON"))) {
        mcptiFlag = true;
    }
    // Warm-up for large size
    setupArgs(args->maxbytes, type, args);
    for (int iter = 0; iter < warmup_iters; iter++) {
        TESTCHECK(startColl(args, type, op, root, 0, iter));
    }
    TESTCHECK(completeColl(args));

    // Warm-up for small size
    setupArgs(args->minbytes, type, args);
    for (int iter = 0; iter < warmup_iters; iter++) {
        TESTCHECK(startColl(args, type, op, root, 0, iter));
    }
    TESTCHECK(completeColl(args));

    FILE *fp2;
    if (is_main_thread)
        fp2 = fopen(mcclDefaultDebugFile, "a+");
    // Benchmark
    for (size_t size = args->minbytes; size <= args->maxbytes;) {
        setupArgs(size, type, args);
        char rootName[100];
        sprintf(rootName, "%5i", root);
        FPRINTF(fp2, "#%12li  %12li  %8s  %6s  %6s", max(args->sendBytes, args->expectedBytes),
                args->nbytes / wordSize(type), typeName, opName, rootName);
        info_log persize;
        if (mcptiFlag == false) {
            sprintf(args->headStr, "%12li  %12li  %8s  %6s  %6s",
                    max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type),
                    typeName, opName, rootName);
        }
        persize.size  = max(args->sendBytes, args->expectedBytes);
        persize.count = args->nbytes / wordSize(type);
        persize.type  = (char *)typeName;
        persize.redop = (char *)opName;
        persize.root  = rootName;

        static int error_try = 0;
        testResult_t res     = BenchTime(args, type, op, root, 0, size_id, persize);
        if (persize.size == 4096 && res != testSuccess && error_try++ < 100) {
            PRINT("last test has failed and try again\n");
            continue;
        }
        TESTCHECK(res);
        TESTCHECK(BenchTime(args, type, op, root, 1, size_id, persize));
        if (mcptiFlag == false) {
            PRINT("%s %s %s\n", args->headStr, args->resultOutPlace, args->resultInPlace);
        }
        FPRINTF(fp2, "\n");
        size_id++;
        info_per_size.push_back(persize);
        size = (args->stepfactor > 1) ? size * args->stepfactor : size + args->stepbytes;
    }
    if (is_main_thread)
        fclose(fp2);

    if (mcptiFlag == true) {
        enableTracerMcpti(args, type, typeName, op, opName, root);

        if (is_main_thread) {
            printTracerMcpti();

            for (int i = 0; i < size_id; i++) {
                PRINT("#%12li  %12li  %8s  %6s  %5s", info_per_size[i].size, info_per_size[i].count,
                      info_per_size[i].type, info_per_size[i].redop, info_per_size[i].root);
                printKernelAvg(i);
                args->reportErrors = datacheck;
                if (args->reportErrors) {
                    PRINT("%11s  %6.2f  %6.2f  %5g", info_per_size[i].outTime.c_str(),
                          info_per_size[i].outAlgbw, info_per_size[i].outBusbw,
                          info_per_size[i].outWrong);
                    if (isSendRecv || isAlltoAll || isAlltoAllv || isAlltoAlld) {
                        PRINT("%11s  %6.2f  %6.2f  %5s", info_per_size[i].inTime.c_str(),
                              info_per_size[i].inAlgbw, info_per_size[i].inBusbw, "N/A");
                    } else {
                        PRINT("%11s  %6.2f  %6.2f  %5g", info_per_size[i].inTime.c_str(),
                              info_per_size[i].inAlgbw, info_per_size[i].inBusbw,
                              info_per_size[i].inWrong);
                    }
                } else {
                    PRINT("%11s  %6.2f  %6.2f  %5s", info_per_size[i].outTime.c_str(),
                          info_per_size[i].outAlgbw, info_per_size[i].outBusbw, "N/A");
                    PRINT("%11s  %6.2f  %6.2f  %5s", info_per_size[i].inTime.c_str(),
                          info_per_size[i].inAlgbw, info_per_size[i].inBusbw, "N/A");
                }
                PRINT("\n");
            }
        }
    }
    return testSuccess;
}

testResult_t threadRunTests(struct threadArgs *args)
{
    // Set device to the first of our GPUs. If we don't do that, some operations
    // will be done on the current GPU (by default : 0) and if the GPUs are in
    // exclusive mode those operations will fail.
    CUDACHECK(cudaSetDevice(args->gpus[0]));
    TESTCHECK(ncclTestEngine.runTest(args, ncclroot, (ncclDataType_t)nccltype,
                                     test_typenames[nccltype], (ncclRedOp_t)ncclop,
                                     test_opnames[ncclop]));
    return testSuccess;
}

testResult_t threadInit(struct threadArgs *args)
{
    char hostname[1024];
    getHostName(hostname, 1024);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    // set main thread again
    is_main_thread = (is_main_proc == 0 && args->thread == 0) ? 1 : 0;

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
        int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        NCCLCHECK(ncclCommInitRank(args->comms + i, nranks, args->ncclId, rank));
    }
    NCCLCHECK(ncclGroupEnd());

    TESTCHECK(threadRunTests(args));

    for (int i = 0; i < args->nGpus; i++) {
        NCCLCHECK(ncclCommDestroy(args->comms[i]));
    }
    return testSuccess;
}

void *threadLauncher(void *thread_)
{
    struct testThread *thread = (struct testThread *)thread_;
    thread->ret               = thread->func(&thread->args);
    return NULL;
}
testResult_t threadLaunch(struct testThread *thread)
{
    pthread_create(&thread->thread, NULL, threadLauncher, thread);
    return testSuccess;
}

struct vmmInfo {
    size_t size;
    CUmemGenericAllocationHandle handle;
};

std::map<CUdeviceptr, vmmInfo> vmmMap = {};

testResult_t AllocateVmm(void **buff, size_t nbytes, int dev)
{
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop{};
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity = 0;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    nbytes = ((nbytes + granularity - 1) / granularity) * granularity;

    CUCHECK(cuMemCreate(&handle, nbytes, &prop, 0));
    CUCHECK(cuMemAddressReserve((CUdeviceptr *)buff, nbytes, 0, 0, 0));
    CUCHECK(cuMemMap(*(CUdeviceptr *)buff, nbytes, 0, handle, 0));
    vmmInfo info{};
    info.size   = nbytes;
    info.handle = handle;
    vmmMap.insert({*(CUdeviceptr *)buff, info});

    return testSuccess;
}

testResult_t FreeVmm(void *buff)
{
    auto it = vmmMap.find((CUdeviceptr)buff);
    if (it == vmmMap.end()) {
        return testInternalError;
    }

    vmmInfo info = it->second;
    CUCHECK(cuMemUnmap((CUdeviceptr)buff, info.size));
    CUCHECK(cuMemAddressFree((CUdeviceptr)buff, info.size));
    CUCHECK(cuMemRelease(info.handle));
    vmmMap.erase((CUdeviceptr)buff);

    return testSuccess;
}

testResult_t FreeBuffs(void *sendbuff, void *recvbuff, void *expected, int datacheck)
{
    if (useVmmMemory) {
        if (sendbuff)
            TESTCHECK(FreeVmm(sendbuff));
        if (recvbuff)
            TESTCHECK(FreeVmm(recvbuff));
    } else {
        if (sendbuff)
            CUDACHECK(cudaFree(sendbuff));
        if (recvbuff)
            CUDACHECK(cudaFree(recvbuff));
    }
    if (datacheck && expected)
        CUDACHECK(cudaFree(expected));
    return testSuccess;
}

testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes,
                           void **expected, size_t nbytes, int dev)
{
    if (useVmmMemory) {
        TESTCHECK(AllocateVmm(sendbuff, nbytes, dev));
        TESTCHECK(AllocateVmm(recvbuff, nbytes, dev));
    } else {
        CUDACHECK(cudaMalloc(sendbuff, nbytes));
        CUDACHECK(cudaMalloc(recvbuff, nbytes));
    }

    if (datacheck)
        CUDACHECK(cudaMalloc(expected, recvBytes));
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char *argv[])
{
    // Make sure everyline is flushed so that we see the progress of the test
    setlinebuf(stdout);

    memcpy(bwUnitStr, "(GB/s)", sizeof("(GB/s)"));

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 4, 0)
    ncclGetVersion(&test_ncclVersion);
#else
    test_ncclVersion = NCCL_VERSION_CODE;
#endif
// printf("# NCCL_VERSION_CODE=%d ncclGetVersion=%d\n", NCCL_VERSION_CODE, test_ncclVersion);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 0, 0)
    test_opnum   = 4;
    test_typenum = 9;
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0) && test_ncclVersion >= NCCL_VERSION(2, 10, 0)) {
        test_opnum++; // ncclAvg
#if defined(__CUDA_BF16_TYPES_EXIST__)
        test_typenum++; // bfloat16
#endif
    }
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0) && test_ncclVersion >= NCCL_VERSION(2, 11, 0)) {
        test_opnum++; // PreMulSum
    }
#endif

    // Parse args
    double parsed;
    int longindex;
    static struct option longopts[] = {{"nthreads", required_argument, 0, 't'},
                                       {"ngpus", required_argument, 0, 'g'},
                                       {"minbytes", required_argument, 0, 'b'},
                                       {"maxbytes", required_argument, 0, 'e'},
                                       {"stepbytes", required_argument, 0, 'i'},
                                       {"stepfactor", required_argument, 0, 'f'},
                                       {"iters", required_argument, 0, 'n'},
                                       {"agg_iters", required_argument, 0, 'm'},
                                       {"warmup_iters", required_argument, 0, 'w'},
                                       {"parallel_init", required_argument, 0, 'p'},
                                       {"check", required_argument, 0, 'c'},
                                       {"op", required_argument, 0, 'o'},
                                       {"datatype", required_argument, 0, 'd'},
                                       {"root", required_argument, 0, 'r'},
                                       {"blocking", required_argument, 0, 'z'},
                                       {"cudagraph", required_argument, 0, 'G'},
                                       {"average", required_argument, 0, 'a'},
                                       {"bw_unit", required_argument, 0, 'u'},
                                       {"enhance", no_argument, 0, 'E'},
                                       {"extend", no_argument, 0, 'x'},
                                       {"help", no_argument, 0, 'h'},
                                       {}};

int proc = 0;
#ifdef MPI_SUPPORT
    int argCount = argc;
    MPI_Init(&argCount, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
#endif
    while (1) {
        int c;
        c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:p:c:o:d:r:z:v:hG:a:u:Ex", longopts, &longindex);

        if (c == -1)
            break;

        switch (c) {
        case 't':
            nThreads = strtol(optarg, NULL, 0);
            break;
        case 'g':
            nGpus = strtol(optarg, NULL, 0);
            break;
        case 'b':
            parsed = parsesize(optarg);
            if (parsed < 0) {
                FPRINTF(stderr, "invalid size specified for 'minbytes'\n");
                return -1;
            }
            minBytes = (size_t)parsed;
            break;
        case 'e':
            parsed = parsesize(optarg);
            if (parsed < 0) {
                FPRINTF(stderr, "invalid size specified for 'maxbytes'\n");
                return -1;
            }
            maxBytes = (size_t)parsed;
            break;
        case 'i':
            stepBytes = strtol(optarg, NULL, 0);
            break;
        case 'f':
            stepFactor = strtol(optarg, NULL, 0);
            break;
        case 'n':
            iters = (int)strtol(optarg, NULL, 0);
            break;
        case 'm':
#if NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 2)
            agg_iters = (int)strtol(optarg, NULL, 0);
#else
            FPRINTF(stderr, "Option -m not supported before NCCL 2.2. Ignoring\n");
#endif
            break;
        case 'w':
            warmup_iters = (int)strtol(optarg, NULL, 0);
            break;
        case 'c':
            datacheck = (int)strtol(optarg, NULL, 0);
            break;
        case 'p':
            parallel_init = (int)strtol(optarg, NULL, 0);
            break;
        case 'o':
            ncclop = ncclstringtoop(optarg);
            break;
        case 'd':
            nccltype = ncclstringtotype(optarg);
            break;
        case 'r':
            ncclroot = strtol(optarg, NULL, 0);
            break;
        case 'z':
            blocking_coll = strtol(optarg, NULL, 0);
            break;
        case 'v':
            useVmmMemory = strtol(optarg, NULL, 0);
            break;
        case 'G':
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && CUDART_VERSION >= 11030
            cudaGraphLaunches = strtol(optarg, NULL, 0);
#else
            printf("Option -G (CUDA graph) not supported before NCCL 2.9 + CUDA 11.3. Ignoring\n");
#endif
            break;
        case 'a':
            average = (int)strtol(optarg, NULL, 0);
            break;
        case 'E':
            ENHANCE_TEST = 1;
            break;
        case 'u': {
            int unit = (int)strtol(optarg, NULL, 0);
            switch (unit) {
            case 0:
                bwUnit = 1.0E9;
                memcpy(bwUnitStr, "(GB/s)", sizeof("(GB/s)"));
                break;
            case 1:
                bwUnit = 1.0E6;
                memcpy(bwUnitStr, "(MB/s)", sizeof("(MB/s)"));
                break;
            case 2:
                bwUnit = 1.0E3;
                memcpy(bwUnitStr, "(KB/s)", sizeof("(KB/s)"));
                break;
            default:
                break;
            }
            break;
        }
        case 'x':
            enable_extend_api = true;
            break;
        case 'h':
        default:
            if(proc == 0) {
                if (c != 'h') printf("invalid option '%c'\n", c);
                printf("USAGE: %s \n\t"
                    "[-t,--nthreads <num threads>] \n\t"
                    "[-g,--ngpus <gpus per thread>] \n\t"
                    "[-b,--minbytes <min size in bytes>] \n\t"
                    "[-e,--maxbytes <max size in bytes>] \n\t"
                    "[-i,--stepbytes <increment size>] \n\t"
                    "[-f,--stepfactor <increment factor>] \n\t"
                    "[-n,--iters <iteration count>] \n\t"
                    "[-m,--agg_iters <aggregated iteration count>] \n\t"
                    "[-w,--warmup_iters <warmup iteration count>] \n\t"
                    "[-p,--parallel_init <0/1>] \n\t"
                    "[-c,--check <0/1>] \n\t"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
                    "[-o,--op <sum/prod/min/max/avg/mulsum/all>] \n\t"
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
                    "[-o,--op <sum/prod/min/max/avg/all>] \n\t"
#else
                    "[-o,--op <sum/prod/min/max/all>] \n\t"
#endif
                    "[-d,--datatype <nccltype/all>] \n\t"
                    "[-r,--root <root>] \n\t"
                    "[-z,--blocking <0/1>] \n\t"
                    "[-v,--use_vmm <0/1>] \n\t"
                    "[-G,--cudagraph <num graph launches>] \n\t"
                    "[-a,--average <0/1/2/3> report average iteration time "
                    "<0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
                    "[-E,--enhance enhance test] \n\t"
                    "[-u,--bw_unit: 0-GB/s, 1-MB/s, 2-KB/s] \n\t"
                    "[-x,--extend API test] \n\t"
                    "[-h,--help]\n",
                    basename(argv[0]));
                is_main_thread = 1;
                std::string defaultDebugFile = getDefaultDebugFile();
                mcclDefaultDebugFile = (char *)defaultDebugFile.c_str();
                FILE *fp = fopen(mcclDefaultDebugFile, "w+");
                FPRINTF(fp, "# arg: %c Unrecognized \n", c);
                FPRINTF(fp, "# Out of bounds values : 1 FAILED\n");
                fclose(fp);
            }
            return 1;
        }
    }
    gDataCheck = datacheck;
    gIters     = iters;
    gAggIters  = agg_iters;

    if (minBytes > maxBytes) {
        FPRINTF(stderr, "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n",
                (unsigned long long)minBytes, (unsigned long long)maxBytes);
        return -1;
    }

    TESTCHECK(run());
    return 0;
}

testResult_t run()
{
    int totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0, color = 0;
    int localRank = 0;
    char hostname[1024];
    getHostName(hostname, 1024);

    FILE *fp;
#ifdef MPI_SUPPORT
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    std::vector<uint64_t> hostHashs(totalProcs);
    hostHashs[proc] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs.data(), sizeof(uint64_t), MPI_BYTE,
                  MPI_COMM_WORLD);
    for (int p = 0; p < totalProcs; p++) {
        if (p == proc)
            break;
        if (hostHashs[p] == hostHashs[proc])
            localRank++;
    }

    char *str     = getenv("NCCL_TESTS_SPLIT_MASK");
    uint64_t mask = str ? strtoul(str, NULL, 16) : 0;
    MPI_Comm mpi_comm;
    color = proc & mask;
    MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
    MPI_Comm_size(mpi_comm, &ncclProcs);
    MPI_Comm_rank(mpi_comm, &ncclProc);
    nProcess = totalProcs;
#endif
    debugLog       = getenv("MCCL_TEST_LOG");
    is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;
    if (is_main_thread) {
        main_process = getpid();
        printf("main_process = %d\n", main_process);
    }
    std::string defaultDebugFile = getDefaultDebugFile();
    mcclDefaultDebugFile = (char *)defaultDebugFile.c_str();
    if (is_main_thread)
        fp = fopen(mcclDefaultDebugFile, "w+");
    //================================================================================================================================================//
    FPRINTF(fp, "======================================================\n");
    FPRINTF(fp,
            "# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: "
            "%d agg iters: %d validation: %d graph: %d\n",
            nThreads, nGpus, minBytes, maxBytes, (stepFactor > 1) ? stepFactor : stepBytes,
            (stepFactor > 1) ? "factor" : "bytes", warmup_iters, iters, agg_iters, datacheck,
            cudaGraphLaunches);
    PRINT("===============================\n");
    PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: "
          "%d agg iters: %d validation: %d graph: %d\n",
          nThreads, nGpus, minBytes, maxBytes, (stepFactor > 1) ? stepFactor : stepBytes,
          (stepFactor > 1) ? "factor" : "bytes", warmup_iters, iters, agg_iters, datacheck,
          cudaGraphLaunches);
    if (blocking_coll)
        PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
    if (parallel_init)
        PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
    PRINT("#\n");

    FPRINTF(fp, "# Using devices\n");
    PRINT("# Using devices\n");
#define MAX_LINE 2048
    char line[MAX_LINE];
    int len       = 0;
    size_t maxMem = ~0;
    char *envdev  = getenv("NCCL_TESTS_DEVICE");
    int gpu0      = envdev ? atoi(envdev) : -1;
    for (int i = 0; i < nThreads * nGpus; i++) {
        int cudaDev = (gpu0 != -1 ? gpu0 : localRank * nThreads * nGpus) + i;
        int rank    = proc * nThreads * nGpus + i;
        cudaDeviceProp prop;
        CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
        len += snprintf(line + len, MAX_LINE - len,
                        "#   Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n", rank, getpid(),
                        hostname, cudaDev, prop.pciBusID, prop.name);
        maxMem = std::min(maxMem, prop.totalGlobalMem);
        FPRINTF(fp, "#   Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n", rank, getpid(),
                hostname, cudaDev, prop.pciBusID, prop.name);
    }
#if MPI_SUPPORT
    char *lines = (proc == 0) ? (char *)malloc(totalProcs * MAX_LINE) : NULL;
    // Gather all output in rank order to root (0)
    MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (proc == 0) {
        for (int p = 0; p < totalProcs; p++)
            PRINT("%s", lines + MAX_LINE * p);
        free(lines);
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
    PRINT("%s", line);
#endif

    // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
    size_t memMaxBytes = (maxMem - (1 << 30)) / (datacheck ? 3 : 2);
    if (maxBytes > memMaxBytes) {
        maxBytes = memMaxBytes;
        if (proc == 0)
            printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
    }

    ncclUniqueId ncclId;
    if (ncclProc == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
#ifdef MPI_SUPPORT
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpi_comm);
    MPI_Barrier(mpi_comm);
#endif
    std::vector<int> gpus(nGpus * nThreads);
    std::vector<mcStream_t> streams(nGpus * nThreads);
    std::vector<void *> sendbuffs(nGpus * nThreads);
    std::vector<void *> recvbuffs(nGpus * nThreads);
    std::vector<void *> expected(nGpus * nThreads);
    size_t sendBytes, recvBytes;

    ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes,
                               (size_t)ncclProcs * nGpus * nThreads);

    // add alltoallv/alltoalld parameter
    std::vector<extendCollArg *> extendCollArgs(nGpus * nThreads);
    size_t nranks = ncclProcs*nGpus*nThreads;
    for (int i = 0; i < nGpus * nThreads; i++) {
        gpus[i] = (gpu0 != -1 ? gpu0 : localRank * nThreads * nGpus) + i;
        CUDACHECK(cudaSetDevice(gpus[i]));
        TESTCHECK(AllocateBuffs(&sendbuffs[i], sendBytes, &recvbuffs[i], recvBytes, &expected[i],
                                (size_t)maxBytes, gpus[i]));
        CUDACHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        CUDACHECK(cudaHostAlloc(&extendCollArgs[i], sizeof(extendCollArg), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDACHECK(cudaHostAlloc(&extendCollArgs[i]->sendcounts, nranks*sizeof(size_t), cudaHostAllocMapped));
        CUDACHECK(cudaHostAlloc(&extendCollArgs[i]->recvcounts, nranks*sizeof(size_t), cudaHostAllocMapped));
        CUDACHECK(cudaHostAlloc(&extendCollArgs[i]->sdispls, nranks*sizeof(size_t), cudaHostAllocMapped));
        CUDACHECK(cudaHostAlloc(&extendCollArgs[i]->rdispls, nranks*sizeof(size_t), cudaHostAllocMapped));
        CUDACHECK(cudaHostAlloc(&extendCollArgs[i]->sendbuffs, nranks*sizeof(void *), cudaHostAllocMapped));
        CUDACHECK(cudaHostAlloc(&extendCollArgs[i]->recvbuffs, nranks*sizeof(void *), cudaHostAllocMapped));
    }

    // if parallel init is not selected, use main thread to initialize NCCL
    ncclComm_t *comms = (ncclComm_t *)malloc(sizeof(ncclComm_t) * nThreads * nGpus);
    if (!parallel_init) {
        if (ncclProcs == 1) {
            NCCLCHECK(ncclCommInitAll(comms, nGpus * nThreads, gpus.data()));
        } else {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus * nThreads; i++) {
                CUDACHECK(cudaSetDevice(gpus[i]));
                NCCLCHECK(ncclCommInitRank(comms + i, ncclProcs * nThreads * nGpus, ncclId,
                                           ncclProc * nThreads * nGpus + i));
            }
            NCCLCHECK(ncclGroupEnd());
        }
    }

    std::vector<int> errors(nThreads);
    std::vector<double> bw(nThreads);
    double *delta;
    CUDACHECK(cudaHostAlloc(&delta, sizeof(double) * nThreads * NUM_BLOCKS,
                            cudaHostAllocPortable | cudaHostAllocMapped));
    std::vector<int> bw_count(nThreads);
    for (int t = 0; t < nThreads; t++) {
        bw[t]     = 0.0;
        errors[t] = bw_count[t] = 0;
    }

    const char *enableMcpti = getenv("MX_TRACER_ENABLED_MCPTI");
    if ((NULL != enableMcpti) && (0 == strcmp(enableMcpti, "ON"))) {
        PRINT("#\n");
        PRINT("#  %10s  %12s  %8s  %6s  %6s    kernel       ┌----- out-of-place ------┐        "
              "┌------ in-place -------┐\n",
              "", "", "", "", "");
        PRINT("#  %10s  %12s  %8s  %6s  %6s   %7s   %7s  %6s  %6s   %6s   %7s  %6s  %6s   %6s\n",
              "size", "count", "type", "redop", "root", "avgtime", "time", "algbw", "busbw",
              "#wrong", "time", "algbw", "busbw", "#wrong");
        PRINT("#  %10s  %12s  %8s  %6s  %6s    %6s    %7s  %6s  %6s  %6s   %7s  %6s  %6s  %5s\n",
              "(B)", "(elements)", "", "", "", "(us)", "(us)", bwUnitStr, bwUnitStr, "", "(us)",
              bwUnitStr, bwUnitStr, "");
    } else {
        PRINT("#\n");
        PRINT("#  %10s  %12s  %8s  %6s  %6s       ┌----- out-of-place ------┐       ┌------ "
              "in-place -------┐\n",
              "", "", "", "", "");
        PRINT("#  %10s  %12s  %8s  %6s  %6s   %7s   %6s  %6s   %6s %7s  %6s  %6s   %6s\n", "size",
              "count", "type", "redop", "root", "time", "algbw", "busbw", "#wrong", "time", "algbw",
              "busbw", "#wrong");
        PRINT("#  %10s  %12s  %8s  %6s  %6s   %7s   %6s  %6s  %6s   %7s  %6s  %6s  %5s\n", "(B)",
              "(elements)", "", "", "", "(us)", bwUnitStr, bwUnitStr, "", "(us)", bwUnitStr, bwUnitStr,
              "");
    }
    FPRINTF(fp, "#\n");
    FPRINTF(fp,
            "#  %10s  %12s  %8s  %6s  %6s       ┌----- out-of-place ------┐       ┌------ in-place "
            "-------┐\n",
            "", "", "", "", "");
    FPRINTF(fp, "#  %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s   %6s  %7s  %6s   %6s   %6s\n", "size",
            "count", "type", "redop", "root", "time", "algbw", "busbw", "#wrong", "time", "algbw",
            "busbw", "#wrong");
    FPRINTF(fp, "#  %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)",
            "(elements)", "", "", "", "(us)", bwUnitStr, bwUnitStr, "", "(us)", bwUnitStr, bwUnitStr,
            "");
    if (is_main_thread)
        fclose(fp);
    std::vector<testThread> threads(nThreads);
    memset(threads.data(), 0, sizeof(struct testThread) * nThreads);

    for (int t = nThreads - 1; t >= 0; t--) {
        threads[t].args.minbytes   = minBytes;
        threads[t].args.maxbytes   = maxBytes;
        threads[t].args.stepbytes  = stepBytes;
        threads[t].args.stepfactor = stepFactor;
        threads[t].args.localRank  = localRank;

        threads[t].args.totalProcs = totalProcs;
        threads[t].args.nProcs     = ncclProcs;
        threads[t].args.proc       = ncclProc;
        threads[t].args.nThreads   = nThreads;
        threads[t].args.thread     = t;
        threads[t].args.nGpus      = nGpus;
        threads[t].args.gpus       = &gpus[t * nGpus];
        threads[t].args.sendbuffs  = &sendbuffs[t * nGpus];
        threads[t].args.recvbuffs  = &recvbuffs[t * nGpus];
        threads[t].args.expected   = &expected[t * nGpus];
        threads[t].args.ncclId     = ncclId;
        threads[t].args.comms      = comms + t * nGpus;
        threads[t].args.streams    = &streams[t * nGpus];
        threads[t].args.collArgs   = &extendCollArgs[t * nGpus];

        threads[t].args.errors   = &errors[t];
        threads[t].args.bw       = &bw[t];
        threads[t].args.bw_count = &bw_count[t];

        threads[t].args.reportErrors = datacheck;

        threads[t].func = parallel_init ? threadInit : threadRunTests;
        if (t)
            TESTCHECK(threadLaunch(&threads[t]));
        else
            TESTCHECK(threads[t].func(&threads[t].args));
    }

    // Wait for other threads and accumulate stats and errors
    for (int t = nThreads - 1; t >= 0; t--) {
        if (t)
            pthread_join(threads[t].thread, NULL);
        TESTCHECK(threads[t].ret);
        if (t) {
            errors[0] += errors[t];
            bw[0] += bw[t];
            bw_count[0] += bw_count[t];
        }
    }

#ifdef MPI_SUPPORT
    MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    if (!parallel_init) {
        for (int i = 0; i < nGpus * nThreads; ++i)
            NCCLCHECK(ncclCommDestroy(comms[i]));
        free(comms);
    }

    // Free off CUDA allocated memory
    for (int i = 0; i < nGpus * nThreads; i++) {
        TESTCHECK(FreeBuffs(sendbuffs[i], recvbuffs[i], expected[i], datacheck));
    }

    // Free off host allocated memory
    for (int i=0; i<nGpus*nThreads; i++) {
        if (extendCollArgs[i]) {
            if (extendCollArgs[i]->sendcounts)
                CUDACHECK(cudaFreeHost(extendCollArgs[i]->sendcounts));
            if (extendCollArgs[i]->recvcounts)
                CUDACHECK(cudaFreeHost(extendCollArgs[i]->recvcounts));
            if (extendCollArgs[i]->sdispls)
                CUDACHECK(cudaFreeHost(extendCollArgs[i]->sdispls));
            if (extendCollArgs[i]->rdispls)
                CUDACHECK(cudaFreeHost(extendCollArgs[i]->rdispls));
            if (extendCollArgs[i]->sendbuffs)
                CUDACHECK(cudaFreeHost(extendCollArgs[i]->sendbuffs));
            if (extendCollArgs[i]->recvbuffs)
                CUDACHECK(cudaFreeHost(extendCollArgs[i]->recvbuffs));
            CUDACHECK(cudaFreeHost(extendCollArgs[i]));
        }
    }
    CUDACHECK(cudaFreeHost(delta));

    char *envstr        = getenv("NCCL_TESTS_MIN_BW");
    double check_avg_bw = envstr ? atof(envstr) : -1;
    bw[0] /= bw_count[0];
    FILE *fp3;
    if (is_main_thread)
        fp3 = fopen(mcclDefaultDebugFile, "a+");
    FPRINTF(fp3, "# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
    FPRINTF(fp3, "# Avg bus bandwidth    : %g %s\n", bw[0],
            check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw * (0.9) ? "FAILED" : "OK"));
    if (is_main_thread) {
        FPRINTF(fp3, "# highest time : %g\n", latency);
        FPRINTF(fp3, "# highest alg bandwith : %g\n", high_algBw);
        FPRINTF(fp3, "# highest bus bandwith : %g\n", high_busBw);
    }
    PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
    PRINT("# Avg bus bandwidth    : %g %s\n", bw[0],
          check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw * (0.9) ? "FAILED" : "OK"));
    PRINT("#\n");
#ifdef MPI_SUPPORT
    MPI_Finalize();
#endif

    if (is_main_thread)
        fclose(fp3);
    // 'cuda-memcheck --leak-check full' requires this
    cudaDeviceReset();

    if (errors[0] || bw[0] < check_avg_bw * (0.9))
        exit(EXIT_FAILURE);
    else
        exit(EXIT_SUCCESS);
}

std::string getDefaultDebugFile(){
    std::string defaultDebugFile;
    uid_t userID      = geteuid();
    struct passwd *pw = getpwuid(userID);
    time_t timep;
    time(&timep);
    char timeStr[64];
    strftime(timeStr, sizeof(timeStr), ".%Y_%m_%d_%H_%M_%S", localtime(&timep));
    defaultDebugFile += MCCL_LOGFILE_PREFIX;
    defaultDebugFile += "." + std::string(pw->pw_name);
    defaultDebugFile += "." + std::to_string(main_process);
    defaultDebugFile += std::string(timeStr);
    defaultDebugFile += MCCL_LOGFILE_END;
    return defaultDebugFile;
}
