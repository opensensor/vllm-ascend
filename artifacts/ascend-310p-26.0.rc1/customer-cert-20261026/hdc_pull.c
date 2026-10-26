/*
 * Read a file back from the Ascend device over HDC.
 *
 * The host's libascend_hal.so exports drvHdcPull, so no client needs to be
 * written.  This is used to confirm that a file pushed with drvHdcSendFileV2
 * actually landed on the device filesystem.
 *
 * Usage: hdc_pull <remote_path> <local_path> [dev_id]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*fn_pull)(int, int, const char *, const char *, void *);
typedef int (*fn_get_peer)(int, int *);

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s <remote_path> <local_path> [dev_id]\n", argv[0]);
        return 2;
    }
    const char *remote = argv[1];
    const char *local = argv[2];
    int dev_id = argc > 3 ? atoi(argv[3]) : 0;

    void *h = dlopen("libascend_hal.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }

    fn_get_peer get_peer = (fn_get_peer)dlsym(h, "drvHdcGetPeerDevId");
    fn_pull pull = (fn_pull)dlsym(h, "drvHdcPull");
    if (!pull) { printf("drvHdcPull NOT exported\n"); return 1; }

    int peer = -1;
    int r = get_peer(0, &peer);
    printf("peer_dev_id=%d\n", peer);
    if (r != 0) return 1;

    printf("pulling %s -> %s (dev %d)\n", remote, local, dev_id);
    fflush(stdout);
    r = pull(0, peer, remote, local, NULL);
    printf("drvHdcPull ret=%d\n", r);
    return r == 0 ? 0 : 1;
}
