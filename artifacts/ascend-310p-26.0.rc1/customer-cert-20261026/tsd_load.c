/*
 * Load an AICPU kernel package onto an Ascend device using the vendor's own
 * TSD client API.
 *
 * The host's CANN ships libtsdclient.so, which exports the supported way to
 * deliver an AICPU/OM package to the device:
 *
 *   uint32_t TsdOpenAicpuSd(const uint32_t logicDeviceId);
 *   uint32_t TsdOpen(const uint32_t logicDeviceId, const uint32_t rankSize);
 *   uint32_t TsdInitQs(const uint32_t logicDeviceId, ...);
 *   uint32_t TsdFileLoad(const uint32_t logicDeviceId,
 *                        const char *filePath, const uint64_t pathLen,
 *                        const char *fileName, const uint64_t fileNameLen);
 *   uint32_t TsdClose(const uint32_t logicDeviceId);
 *
 * This is the path the vendor's own ProcessModeManager uses: the package is sent
 * over HDC together with a SHA-256 "check code", and the device recomputes that
 * hash over the received file and compares.  There is no certificate and no
 * vendor key in this path -- the integrity of the package is a checksum the host
 * supplies.
 *
 * The device decompresses the package and dlopen()s the libraries it contains;
 * the op descriptor (aicpu_kernel.json) names the kernel shared object
 * explicitly, e.g. "kernelSo": "libcpu_kernels.so".
 *
 * Usage:
 *   tsd_load <package_file> [logicDeviceId] [rankSize]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef uint32_t (*fn_open_aicpu)(uint32_t);
typedef uint32_t (*fn_open)(uint32_t, uint32_t);
typedef uint32_t (*fn_file_load)(uint32_t, const char *, uint64_t,
                                  const char *, uint64_t);
typedef uint32_t (*fn_close)(uint32_t);
typedef uint32_t (*fn_capability_get)(uint32_t, void *);

static void *g_h;

static void *sym(const char *name)
{
    void *p = dlsym(g_h, name);
    printf("  %-24s %s\n", name, p ? "found" : "NOT FOUND");
    return p;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <package_file> [logicDeviceId] [rankSize]\n", argv[0]);
        return 2;
    }
    const char *pkg = argv[1];
    uint32_t dev = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 0;
    uint32_t rank = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 1;

    g_h = dlopen("libtsdclient.so", RTLD_NOW);
    if (!g_h) {
        printf("dlopen libtsdclient.so failed: %s\n", dlerror());
        return 1;
    }
    printf("=== TSD AICPU package load ===\n");
    printf("package      : %s\n", pkg);
    printf("logicDeviceId: %u\n", dev);
    printf("rankSize     : %u\n\n", rank);

    fn_open_aicpu open_aicpu = (fn_open_aicpu)sym("TsdOpenAicpuSd");
    fn_open open = (fn_open)sym("TsdOpen");
    fn_file_load file_load = (fn_file_load)sym("TsdFileLoad");
    fn_close close_fn = (fn_close)sym("TsdClose");

    if (!file_load) {
        printf("TsdFileLoad is required\n");
        return 1;
    }

    /* split the path into directory and file name, as the API wants */
    char path[4096];
    const char *base = strrchr(pkg, '/');
    char name[512];
    if (base) {
        size_t dlen = (size_t)(base - pkg);
        if (dlen >= sizeof(path)) dlen = sizeof(path) - 1;
        memcpy(path, pkg, dlen);
        path[dlen] = '\0';
        snprintf(name, sizeof(name), "%s", base + 1);
    } else {
        snprintf(path, sizeof(path), "%s", ".");
        snprintf(name, sizeof(name), "%s", pkg);
    }
    printf("\nfilePath     : %s\n", path);
    printf("fileName     : %s\n", name);

    uint32_t r;
    if (open_aicpu) {
        fprintf(stderr, "TsdOpenAicpuSd ...\n"); fflush(stderr);
        r = open_aicpu(dev);
        fprintf(stderr, "TsdOpenAicpuSd(%u) = %u\n", dev, r); fflush(stderr);
    }
    if (open) {
        fprintf(stderr, "TsdOpen ...\n"); fflush(stderr);
        r = open(dev, rank);
        fprintf(stderr, "TsdOpen(%u, %u) = %u\n", dev, rank, r); fflush(stderr);
    }

    fprintf(stderr, "TsdFileLoad(%s,%s) ...\n", path, name);
    fflush(stderr);
    r = file_load(dev, path, strlen(path), name, strlen(name));
    fprintf(stderr, "TsdFileLoad = %u\n", r);
    fflush(stderr);

    if (close_fn) {
        uint32_t c = close_fn(dev);
        printf("TsdClose = %u\n", c);
    }
    return r == 0 ? 0 : 1;
}
