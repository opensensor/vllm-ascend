/*
 * Read a file back from the Ascend device over HDC.
 *
 * libascend_hal.so exports drvHdcPull, the counterpart to drvHdcSendFile:
 *
 *   drvHdcPull(dev_id, peer_id, src_path, dst_path)
 *
 * This gives us visibility into the device filesystem -- which is the missing
 * piece for everything so far: the device's own log names the failing step in
 * the AICPU package install ("[TSDaemon] Checksum aicpu package failed",
 * "Move origin package to decompress path failed"), and its /etc/shadow is
 * what we would need for the SSH route.
 *
 * Usage: hdc_pull2 <dev_id> <device_src_path> <local_dst_path>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef int (*fn_pull)(int, int, const char *, const char *);

int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("usage: %s <dev_id> <device_src> <local_dst>\n", argv[0]);
        return 2;
    }
    void *h = dlopen("libascend_hal.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }

    fn_pull pull = (fn_pull)dlsym(h, "drvHdcPull");
    if (!pull) { printf("drvHdcPull not exported\n"); return 1; }

    int r = pull(atoi(argv[1]), 0, argv[2], argv[3]);
    printf("drvHdcPull(\"%s\" -> \"%s\") = %d (0x%x)\n", argv[2], argv[3], r, r);
    return r == 0 ? 0 : 1;
}
