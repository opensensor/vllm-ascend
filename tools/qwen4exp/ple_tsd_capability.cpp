// SPDX-License-Identifier: Apache-2.0
// Query the installed TSD capability ABI without uploading or starting a UDF.
#include <acl/acl.h>
#include <aicpu/tsd/tsd_client.h>
#include <cstdint>
#include <cstdio>

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int status = aclInit(nullptr);
    std::printf("initialize=%d\n", status);
    if (status != 0) return 1;
    status = aclrtSetDevice(0);
    std::printf("set_device=%d\n", status);
    if (status != 0) { aclFinalize(); return 2; }
    uint32_t level = 0;
    bool builtin_udf = false;
    const uint32_t level_status = TsdCapabilityGet(
        0, TSD_CAPABILITY_LEVEL, reinterpret_cast<uint64_t>(&level));
    const uint32_t udf_status = TsdCapabilityGet(
        0, TSD_CAPABILITY_BUILTIN_UDF, reinterpret_cast<uint64_t>(&builtin_udf));
    std::printf("level_status=%u level=0x%08x builtin_udf_status=%u builtin_udf=%d\n",
                level_status, level, udf_status, builtin_udf);
    const int reset_status = aclrtResetDevice(0);
    const int finalize_status = aclFinalize();
    std::printf("reset_device=%d finalize=%d udf_launched=0 dma_submitted=0\n",
                reset_status, finalize_status);
    return level_status || udf_status || reset_status || finalize_status ? 3 : 0;
}
