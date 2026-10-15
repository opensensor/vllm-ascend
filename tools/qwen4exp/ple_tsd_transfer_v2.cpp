// SPDX-License-Identifier: Apache-2.0
// Private host-process LD_PRELOAD adapter; never install into the serving stack.
#include <ascend_hal.h>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

#include "ple_tsd_transfer_path.h"

extern "C" drvError_t drvHdcSendFile(int node, int device, const char* source,
                                    const char* destination,
                                    void (*notify)(struct drvHdcProgInfo*)) {
    const auto original = reinterpret_cast<decltype(&drvHdcSendFile)>(
        dlsym(RTLD_NEXT, "drvHdcSendFile"));
    if (!original) return DRV_ERROR_NOT_SUPPORT;
    if (node != 0 || device != 0 || !source || !destination)
        return original(node, device, source, destination, notify);

    const auto get_legacy = reinterpret_cast<decltype(&drvHdcGetTrustedBasePath)>(
        dlsym(RTLD_NEXT, "drvHdcGetTrustedBasePath"));
    const auto get_v2 = reinterpret_cast<decltype(&drvHdcGetTrustedBasePathV2)>(
        dlsym(RTLD_NEXT, "drvHdcGetTrustedBasePathV2"));
    const auto send_v2 = reinterpret_cast<decltype(&drvHdcSendFileV2)>(
        dlsym(RTLD_NEXT, "drvHdcSendFileV2"));
    constexpr unsigned int kPathCapacity = 4096;
    char legacy[kPathCapacity] = {};
    char v2[kPathCapacity] = {};
    std::string redirected;
    if (!get_legacy || !get_v2 || !send_v2 ||
        get_legacy(node, device, legacy, kPathCapacity) != DRV_ERROR_NONE ||
        get_v2(node, device, v2, kPathCapacity) != DRV_ERROR_NONE ||
        !ple_tsd_transfer_path(source, destination, legacy, v2,
                               static_cast<unsigned long>(getpid()), redirected))
        return original(node, device, source, destination, notify);

    const auto status = send_v2(node, device, source, redirected.c_str(), notify);
    std::fprintf(stderr, "PLE_TSD_V2 pid=%ld device=%d status=%d destination=%s\n",
                 static_cast<long>(getpid()), device, static_cast<int>(status),
                 redirected.c_str());
    return status;
}
