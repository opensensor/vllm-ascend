/*
 * Read a file back from the Ascend device over HDC.
 *
 * The device's hdcd file-transfer service is bidirectional, and the host HAL
 * exposes drvHdcPull for the read direction.  This is used to confirm that a
 * package we pushed actually landed on the device -- and, later, to read the
 * device's own filesystem once we have a way in.
 *
 * Usage: hdc_pull3 <remote_path> <local_path> [dev_id]
 */
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

typedef int (*fn_pull)(int, int, const char *, const char *);

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::printf("usage: %s <remote_path> <local_path> [dev_id]\n", argv[0]);
        return 2;
    }
    int dev = argc > 3 ? std::atoi(argv[3]) : 0;

    void *h = dlopen("libascend_hal.so", RTLD_NOW);
    if (!h) { std::printf("dlopen failed: %s\n", dlerror()); return 1; }

    auto pull = (fn_pull)dlsym(h, "drvHdcPull");
    if (!pull) { std::printf("drvHdcPull not exported\n"); return 1; }

    int r = pull(dev, dev, argv[1], argv[2]);
    std::printf("drvHdcPull(dev=%d, \"%s\" -> \"%s\") = %d (0x%x)\n",
                dev, argv[1], argv[2], r, r);
    return r == 0 ? 0 : 1;
}
