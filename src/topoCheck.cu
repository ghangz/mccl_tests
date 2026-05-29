/*************************************************************************
 * Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "topoCheck.h"
#include "nccl.h"
#include "stdlib.h"
#include "mxc_ext.h"
#include <unordered_set>
#include <cstring>


#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
#define VSWITCH_INFO_DIR "/opt/pci_switch_link/virtual_switch_links/0000:00:00.0"
static void memcpylower(char *dst, const char *src, const size_t size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = tolower(src[i]);
    }
}
ncclResult_t getPciPath(const char *busId, char *path)
{
    char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
    memcpylower(busPath + sizeof("/sys/class/pci_bus/") - 1, busId, BUSID_REDUCED_SIZE - 1);
    memcpylower(busPath + sizeof("/sys/class/pci_bus/0000:00/../../") - 1, busId, BUSID_SIZE - 1);
    if (realpath(busPath, path) == nullptr) {
        fprintf(stderr, "[Error] Could not find real path of %s\n", busPath);
        return ncclSystemError;
    }
    return ncclSuccess;
}
ncclResult_t ncclTopoGetStrFromSys(const char *path, const char *fileName, char *strValue)
{
    char filePath[PATH_MAX];
    sprintf(filePath, "%s/%s", path, fileName);
    int offset = 0;
    FILE *file;
    if ((file = fopen(filePath, "r")) != NULL)
    {
        while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN)
        {
            int len = fread(strValue + offset, 1, MAX_STR_LEN - offset, file);
            offset += len;
        }
        fclose(file);
    }
    if (offset == 0)
    {
        strValue[0] = '\0';
        // INFO(nccl_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
    }
    else
    {
        strValue[offset - 1] = '\0';
    }
    return ncclSuccess;
}
ncclResult_t getGpuInfoFromSys(ncclTopoGpuNode_t *gpu)
{
    if (!gpu)
    {
        return ncclInvalidArgument;
    }
    if (getPciPath(gpu->busidStr, gpu->pciePath) != ncclSuccess) {
        return ncclInternalError;
    }
    char buf[MAX_STR_LEN];
    ncclTopoGetStrFromSys(gpu->pciePath, "numa_node", buf);
    if (buf[0] == '\0')
    {
        return ncclInternalError;
    }
    gpu->numaNode = strtol(buf, NULL, 0);

    ncclTopoGetStrFromSys(gpu->pciePath, "vendor", buf);
    if (buf[0] == '\0')
    {
        return ncclInternalError;
    }
    gpu->mxArch += strtol(buf, NULL, 0) << 16;
    ncclTopoGetStrFromSys(gpu->pciePath, "device", buf);
    if (buf[0] == '\0')
    {
        return ncclInternalError;
    }
    gpu->mxArch += strtol(buf, NULL, 0);
    return ncclSuccess;
}
ncclResult_t getMetaxLinkInfo(ncclTopoGpuNode_t *gpu, int cnt)
{
    for (int i = 0; i < cnt; ++i)
    {
        ncclTopoGpuNode_t *gpu_it = gpu + i;
        for (int j = 0; j < cnt; ++j)
        {
            if (i == j)
            {
                continue;
            }
            uint32_t link_type, hops;
            if (mcExtGetLinkTypeAndHopCount(i, j, &link_type, &hops) == mcSuccess)
            {
                // TODO check status and env param
                if (link_type == MXC_VENDOR_LINK_INFO_TYPE_METALINK)
                {
                    gpu_it->metaxLinkNodes[gpu_it->metaxLinkNodesCnt++] = j;
                }
            }
        }
    }
    return ncclSuccess;
}
bool isCommon(ncclTopoGpuNode_t *gpu, int cnt)
{
    // 判断numa
    std::unordered_set<int> numaid_s;
    std::unordered_set<std::string> rootBusids;
    for (int i = 0; i < cnt; ++i)
    {
        numaid_s.emplace((gpu + i)->numaNode);
        std::string tmp((gpu + i)->pciePath + 19, 7);
        rootBusids.emplace(tmp);
    }
    return numaid_s.size() == 1 && rootBusids.size() == 2;
}
bool isCascade(ncclTopoGpuNode_t *gpu, int cnt)
{
    // 判断numa
    std::unordered_set<int> numaid_s;
    std::unordered_set<std::string> rootBusids;
    for (int i = 0; i < cnt; ++i)
    {
        numaid_s.emplace((gpu + i)->numaNode);
        std::string tmp((gpu + i)->pciePath + 19, 7);
        rootBusids.emplace(tmp);
    }
    return numaid_s.size() == 1 && rootBusids.size() == 1;
}
bool isDragonFly()
{
    mcFabricInfo systemInfo;
    mcDeviceGetFabricInfo(&systemInfo);
    // printf("systemInfo: type[%d] index[%d] totalGpu[%d]\n", systemInfo.type, systemInfo.index, systemInfo.totalGpu);
    return systemInfo.status == mcFabricConnected && systemInfo.type == mcMNFCDragonfly;
}
bool isOAMDeviceArch(ncclTopoGpuNode_t *gpu, int cnt)
{
    for (int i = 0; i < cnt; ++i)
    {
        if (gpu->mxArch != MXARCH_OAM_C550 && gpu->mxArch != MXARCH_OAM_C290)
        {
            return false;
        }
    }
    return true;
}
bool isAllMetaxlink4(ncclTopoGpuNode_t *gpu, int cnt)
{
    for (int i = 0; i < cnt; ++i)
    {
        ncclTopoGpuNode_t *gpu_it = gpu + i;
        if (gpu_it->metaxLinkNodesCnt != 3)
        {
            return false;
        }
    }
    return true;
}
bool isAllMetaxlink8(ncclTopoGpuNode_t *gpu, int cnt)
{
    for (int i = 0; i < cnt; ++i)
    {
        ncclTopoGpuNode_t *gpu_it = gpu + i;
        if (gpu_it->metaxLinkNodesCnt != 7)
        {
            return false;
        }
    }
    return true;
}
ncclResult_t checkPcieTopo(ncclTopoGpuNode_t *gpu, int cnt, PcieTopo_t *pcieTopo)
{
    if (isOAMDeviceArch(gpu, cnt))
    {
        *pcieTopo = PCIE_TOPO_OAM;
    }
    else if (isDragonFly())
    {
        *pcieTopo = PCIE_TOPO_DRAGONFLY;
    }
    else if (isCommon(gpu, cnt))
    {
        *pcieTopo = PCIE_TOPO_COMMON;
    }
    else
    {
        *pcieTopo = PCIE_TOPO_CASCADE;
    }
    return ncclSuccess;
}
ncclResult_t checkMetaxTopo(ncclTopoGpuNode_t *gpu, int cnt, MetaxTopo_t *metaxTopo)
{
    if (cnt == 4)
    {
        if (isAllMetaxlink4(gpu, cnt) == true)
        {
            *metaxTopo = METAX_TOPO_MTLK4;
            return ncclSuccess;
        }
        else
        {
            return ncclInternalError;
        }
    }
    else if (cnt == 8)
    {
        if (isAllMetaxlink4(gpu, cnt) == true)
        {
            *metaxTopo = METAX_TOPO_MTLK4_4;
            return ncclSuccess;
        }
        else if (isAllMetaxlink8(gpu, cnt))
        {
            *metaxTopo = METAX_TOPO_MTLK8;
            return ncclSuccess;
        }
        else
        {
            fprintf(stderr, "[ERROR] unexpected metaxlink cnt\n");
            return ncclInternalError;
        }
    }
    else
    {
        return ncclInternalError;
    }
    return ncclSuccess;
}
ncclResult_t getTopoGpuNodes(ncclTopoGpuNode_t **gpuNodes, int *cnt)
{
    ncclResult_t ret = ncclSuccess;
    *cnt = 0;
    mcGetDeviceCount(cnt);
    if (*cnt != 4 && *cnt != 8 && *cnt != 16)
    {
        fprintf(stderr, "[ERROR] gpu cnt %d, out-of-scope topo\n", *cnt);
        return ncclInternalError;
    }
    *gpuNodes = (ncclTopoGpuNode_t *)calloc(*cnt, sizeof(ncclTopoGpuNode_t));
    for (int i = 0; i < *cnt; ++i)
    {
        ncclTopoGpuNode_t *gpu = (*gpuNodes) + i;
        gpu->dev = i;
        mcSetDevice(i);
        mcDeviceGetPCIBusId(gpu->busidStr, sizeof(gpu->busidStr), i);
        mcDeviceProp_t devProp;
        mcGetDeviceProperties(&devProp, i);
        memcpy(&gpu->arch, &devProp.arch, sizeof(mcDeviceArch_t));
        if (getGpuInfoFromSys(gpu) != ncclSuccess) {
            return ncclInternalError;
        }
    }
    return ncclSuccess;
}
// 获取PCIE TOPO 和MetaxLink 拓扑信息
ncclResult_t getMetaxTopo(PcieTopo_t *pcie_topo, MetaxTopo_t *metax_topo)
{
    ncclResult_t ret = ncclSuccess;
    int cnt = 0;
    ncclTopoGpuNode_t *gpuNodes = nullptr;
    if (getTopoGpuNodes(&gpuNodes, &cnt) != ncclSuccess) {
        ret = ncclInternalError;
        goto exit;
    }
    getMetaxLinkInfo(gpuNodes, cnt);
    if (ncclSuccess != checkPcieTopo(gpuNodes, cnt, pcie_topo))
    {
        ret = ncclInternalError;
        goto exit;
    }
    if (ncclSuccess != checkMetaxTopo(gpuNodes, cnt, metax_topo))
    {
        ret = ncclInternalError;
        goto exit;
    }
exit:
    if (gpuNodes)
        free(gpuNodes);
    return ret;
}

