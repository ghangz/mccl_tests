/*************************************************************************
 * Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "nccl.h"
#include "stdlib.h"
#include "limits.h"
#include "mxc_ext.h"
#include <unordered_set>
#include <string>

#define MAX_NODES 64
#define BUSID_STRLEN 18
#define MAX_STR_LEN 255
#define MXARCH_OAM_C550 (0x99994000)
#define MXARCH_OAM_C290 (0x99994080)

typedef struct ncclTopoGpuNode
{
    char busidStr[BUSID_STRLEN];
    int64_t busId;
    int dev; // NVML dev number
    int mxArch; //
    int numaNode;
    mcDeviceArch_t arch;
    char pciePath[PATH_MAX];
    int metaxLinkNodesCnt;
    int metaxLinkNodes[MAX_NODES];
} ncclTopoGpuNode_t;

typedef enum
{
    PCIE_TOPO_OAM = 0,
    PCIE_TOPO_DRAGONFLY,
    PCIE_TOPO_COMMON,
    PCIE_TOPO_CASCADE,
    PCIE_TOPO_MAX,
} PcieTopo_t;

typedef enum
{
    METAX_TOPO_MTLK4,
    METAX_TOPO_MTLK8,
    METAX_TOPO_MTLK4_4,
    METAX_TOPO_MAX,
} MetaxTopo_t;

typedef struct topoExpectPerformance
{
    MetaxTopo_t metax;
    unsigned int process;
    unsigned int thread;
    unsigned int gpu_perthread;
    float lat;
    float bw;
} topoExpectPerformance_t;

typedef struct topoExpectPerformanceEth
{
    std::string case_name;
    unsigned int rank;
    float lat;
    float bw;
} topoExpectPerformanceEth_t;

ncclResult_t getTopoGpuNodes(ncclTopoGpuNode_t **gpu_nodes, int* cnt);
ncclResult_t getMetaxTopo(PcieTopo_t* pcie_topo, MetaxTopo_t* metax_topo);

