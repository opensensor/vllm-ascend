// SPDX-License-Identifier: Apache-2.0
// Read-only capability query. Does not register memory or submit DMA.
#include <acl/acl.h>
#include <ascend_hal.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    char* end = nullptr;
    long rank = std::strtol(argv[1], &end, 10);
    if (*argv[1] == '\0' || *end != '\0' || rank < 0 || rank > 3) return 1;
    if (aclInit(nullptr) || aclrtSetDevice(static_cast<int>(rank))) return 2;
    const unsigned modules[] = {DRV_ACC_MODULE_TYPE_AICPU, DRV_ACC_MODULE_TYPE_AIC,
        DRV_ACC_MODULE_TYPE_AIV, DRV_ACC_MODULE_TYPE_PCIEDMA, DRV_ACC_MODULE_TYPE_SDMA};
    const char* names[] = {"aicpu", "aicore", "aivector", "pcie_dma", "sdma"};
    for (unsigned i = 0; i < sizeof(modules) / sizeof(modules[0]); ++i) {
        unsigned capability = 0xffffffffU;
        auto status = halHostRegisterCapabilities(static_cast<unsigned>(rank), modules[i], &capability);
        std::printf("{\"rank\":%ld,\"module\":\"%s\",\"status\":%d,\"capability\":%u,"
                    "\"memory_registered\":false,\"dma_submitted\":false}\n",
                    rank, names[i], static_cast<int>(status), capability);
    }
    return aclFinalize() ? 3 : 0;
}
