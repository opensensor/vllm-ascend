/*
 * Query the device's TSD capability set.
 *
 * ProcessModeManager::IsOkToLoadFileToDevice() consults
 * IsSupportHeterogeneousInterface(), which is a bitmask query answered by the
 * device (tsdaemon).  If the device does not advertise the right capability,
 * the AICPU package load is skipped.  This prints what the device reports for
 * each capability type.
 *
 * Usage: cap_probe [logicDeviceId]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef uint32_t (*fn_cap_get)(uint32_t, int32_t, uint64_t);

int main(int argc, char **argv)
{
    void *h = dlopen("libtsdclient.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    uint32_t dev = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 0;

    fn_cap_get cap = (fn_cap_get)dlsym(h, "TsdCapabilityGet");
    if (!cap) { printf("TsdCapabilityGet absent\n"); return 1; }

    printf("=== TSD capabilities, device %u ===\n", dev);
    for (int32_t type = 0; type <= 6; type++) {
        uint64_t out = 0;
        uint32_t r = cap(dev, type, (uint64_t)(uintptr_t)&out);
        printf("  type %d : ret=%u value=0x%llx\n", type, r,
               (unsigned long long)out);
    }
    uint64_t out = 0;
    uint32_t r = cap(dev, 0xFF, (uint64_t)(uintptr_t)&out);
    printf("  type 255 (BUT): ret=%u value=0x%llx\n", r, (unsigned long long)out);
    return 0;
}
