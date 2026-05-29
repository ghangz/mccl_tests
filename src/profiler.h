/*************************************************************************
 * Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __PROFILER_H__
#define __PROFILER_H__

#include <cupti.h>
#include <cxxabi.h>
#include "common.h"
#include <vector>
#include <string>
#include <map>

#define BUF_SIZE (320 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                   \
    (((uintptr_t)(buffer) & ((align)-1))                              \
         ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
         : (buffer))

int gDataCheck;
int gIters;
int gAggIters;
static uint64_t startTimestamp;
int main_process = 0;
const char *debugLog = nullptr;
std::map<int, std::vector<float>> avgTimeVec;
thread_local int iter_kernel_cnt = 0;
thread_local double iter_avg_kernel_druation = 0;
thread_local int size_id = 0;
thread_local int size_id_profiler = 0;
thread_local int size_id_mcpti = 0;
thread_local std::vector<info_log> info_per_size;
thread_local std::vector<info_log> info_per_size_profiler;

static void ptiActivityRecordHandler(CUpti_Activity *record, uint64_t &totalTime, int &totalCnt, bool &isFirstRecord)
{
    char demangle_buffer[1024];
    size_t demangle_buffer_size;
    int demangle_ret;
    switch (record->kind)
    {
        case MCPTI_ACTIVITY_KIND_KERNEL:
        case MCPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            demangle_buffer_size = 1024;
            memset(demangle_buffer, 0, demangle_buffer_size);
            MCpti_ActivityKernel8 *kernel = (MCpti_ActivityKernel8 *)record;
            demangle_buffer_size = sizeof(demangle_buffer);
            demangle_ret = 0;
            memset(demangle_buffer, 0, demangle_buffer_size);
            if (nullptr == kernel->name) {
                if (kernel->graphNodeId) {
                    strcpy(demangle_buffer, "GraphExecKernel");
                } else {
                    strcpy(demangle_buffer, "ExecKernel");
                }
            } else {
                abi::__cxa_demangle(kernel->name, demangle_buffer, &demangle_buffer_size, &demangle_ret);
                if (demangle_ret) {
                    printf("kernel name [%s] demangling error\n", kernel->name);
                }
            }
            std::string kernel_name(demangle_buffer);

            const char *enableMcpti = getenv("MX_TRACER_ENABLED_MCPTI");
            std::string PREPARE_INPUT_KERNEL = "prepareInput";
            std::string PREPARE_EXPECTED_KERNEL = "prepareExpected";
            std::string VERIFY_PREPARED_KERNEL = "verifyPrepared";
            if ((kernel_name.find(PREPARE_INPUT_KERNEL) != std::string::npos) || (kernel_name.find(PREPARE_EXPECTED_KERNEL) != std::string::npos) || (kernel_name.find(VERIFY_PREPARED_KERNEL) != std::string::npos)) {
                break;
            }
            uint64_t dur = kernel->end - kernel->start;
            if (isFirstRecord == true) {
                isFirstRecord = false;
                return;
            }
            totalTime += dur;
            totalCnt++;

            {
                const char *kindString =
                    (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
                MCpti_ActivityKernel8 *kernel = (MCpti_ActivityKernel8 *)record;
                if ((NULL != debugLog) && (0 == strcmp(debugLog, "ON"))) {
                    printf("mccl_test: %s \"%s\" [ %llu - %llu ] pid %d, dur %ld, device %u, context %u, stream %u, correlation %u\n",
                        kindString, kernel_name.c_str(), (unsigned long long)(kernel->start - startTimestamp),
                        (unsigned long long)(kernel->end - startTimestamp), getpid(), dur, kernel->deviceId,
                        kernel->contextId, kernel->streamId, kernel->correlationId);
                    // printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
                    //       kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX, kernel->blockY,
                    //       kernel->blockZ, kernel->staticSharedMemory, kernel->dynamicSharedMemory);
                }
            }
            break;
        }
        default:
            break;
    }
}

void bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        printf("Error: out of memory\n");
        exit(EXIT_FAILURE);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
                     size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;
    uint64_t totalTime = 0;
    int totalCnt = 0;
    float avgTime = 0.0;
    if (main_process != getpid()) {
        return;
    }
    if (validSize > 0) {
        bool isFirstRecord = true;
        do
        {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                if ((gDataCheck && (totalCnt == gAggIters * gIters))) {
                    break;
                }
                ptiActivityRecordHandler(record, totalTime, totalCnt, isFirstRecord);
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                const char *errstr;
                cuptiGetResultString(status, &errstr);
                fprintf(stderr, "Error@@%s:%d.\n", __FILE__, __LINE__);
                fprintf(stderr, "Error info: %s.\n", errstr);
                exit(EXIT_FAILURE);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int)dropped);
        }
        avgTime = totalTime / (float)totalCnt;
        int pid = getpid();
        if ((NULL != debugLog) && (0 == strcmp(debugLog, "ON"))) {
            printf("pid = %d, totalTime = %lu, totalCnt = %d, avgTime = %f\n", pid, totalTime, totalCnt, avgTime);
        }
        auto it = avgTimeVec.find(pid);
        if (it == avgTimeVec.end()) {
            std::vector<float> vec;
            vec.push_back(avgTime);
            avgTimeVec.insert(make_pair(pid, vec));
        } else {
            it->second.push_back(avgTime);
        }
    }

    free(buffer);
}

void StartPtiTracing()
{
    // ONLY track kernel activity
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    cuptiGetTimestamp(&startTimestamp);
}

void StopPtiTracing()
{
    cuptiActivityFlushAll(1);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

testResult_t enableTracerMcpti(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root)
{
    for (size_t size = args->minbytes; size <= args->maxbytes; size = ((args->stepfactor > 1) ? size * args->stepfactor : size + args->stepbytes))
    {
        setupArgs(size, type, args);
        char rootName[100];
        info_log persize_profiler;
        persize_profiler.size = max(args->sendBytes, args->expectedBytes);
        persize_profiler.count = args->nbytes / wordSize(type);
        persize_profiler.type = (char *)typeName;
        persize_profiler.redop = (char *)opName;
        persize_profiler.root = rootName;

        if (is_main_thread) {
            StartPtiTracing();
        }
        TESTCHECK(BenchTime(args, type, op, root, 0, size_id_mcpti, persize_profiler)); // outofplace kernel time
        // TESTCHECK(BenchTime(args, type, op, root, 1, size_id_mcpti, persize_profiler));

        if (is_main_thread) {
            StopPtiTracing();
        }
        size_id_mcpti++;
        info_per_size_profiler.push_back(persize_profiler);
    }
    return testSuccess;
}

void printTracerMcpti()
{
    if (main_process == getpid()) {
        float avgtime = 0.0;
        auto it = avgTimeVec.find(main_process);
        if (it == avgTimeVec.end()) {
            printf("not found avgtime about main_process[%d]\n", main_process);
            for (int i = 0; i < size_id; i++) {
                info_per_size_profiler[i].kernelTime = 0;
            }
        } else {
            if ((NULL != debugLog) && (0 == strcmp(debugLog, "ON"))) {
                for (auto it = avgTimeVec.begin(); it != avgTimeVec.end(); ++it) {
                    std::vector<float>::iterator it_inner;
                    for(it_inner = it->second.begin(); it_inner != it->second.end(); ++it_inner) {
                        printf("size[%ld] = %f\n", std::distance(it->second.begin(), it_inner), *it_inner);
                    }
                }
            }
            for (int i = 0; i < size_id; i++) {
                avgtime = avgTimeVec[main_process][i];
                info_per_size_profiler[i].kernelTime = avgtime / 1000;
            }
        }
    }
}

void printKernelAvg(int i)
{
    float tempTime = info_per_size_profiler[i].kernelTime;
    char timeStr[100];
    sprintf(timeStr, "%7.2f", tempTime);
    PRINT("%11s",timeStr);
}
#endif