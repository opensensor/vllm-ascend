/*
 * Push a file to the Ascend device over HDC, using the vendor's own API.
 *
 * The host's libascend_hal.so already exports the complete HDC client API
 * (43 drvHdc* symbols), so no client has to be written from scratch:
 *
 *   drvHdcClientCreate(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag)
 *   drvHdcGetPeerDevId(int peer_node, int *peer_devid)
 *   drvHdcSessionConnect(int peer_node, int peer_devid, HDC_CLIENT client, HDC_SESSION *session)
 *   drvHdcSendFile(int peer_node, int peer_devid, const char *file,
 *                  const char *dst_path, void (*progress_notifier)(struct drvHdcProgInfo *))
 *
 * The device side is /var/hdcd (uid 1000), and the device's file-transfer
 * whitelist (/etc/hdcFileTransWhiteList.cfg) is permissive for the AICPU
 * kernel tarballs:
 *
 *   _Ascend310P-aicpu_syskernels.tar.gz        *  0  104857600  64  0
 *   _Ascend310P-aicpu_extend_syskernels.tar.gz  *  0  104857600  64  0
 *
 * i.e. save_sub_path = "*" (any path), name_full_check_flag = 0 (no filename
 * check), 100 MB, 64 files, root_privilege_flag = 0.
 *
 * Usage: hdc_push <local_file> <dst_path> [dev_id]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *HDC_CLIENT;
typedef void *HDC_SESSION;

typedef int (*fn_client_create)(HDC_CLIENT *, int, int, int);
typedef int (*fn_get_peer)(int, int *);
typedef int (*fn_session_connect)(int, int, HDC_CLIENT, HDC_SESSION *);
typedef int (*fn_send_file)(int, int, const char *, const char *, void *);
typedef int (*fn_send_file_v2)(int, int, const char *, const char *, void *);
typedef int (*fn_get_base_path)(int, int, char *, unsigned int);
typedef int (*fn_load_wl)(const char *);
typedef int (*fn_client_destroy)(HDC_CLIENT);
typedef int (*fn_session_close)(HDC_CLIENT, HDC_SESSION);

#define HDC_SERVICE_TYPE_FILE_TRANS 3

static void *g_h;

static void *sym(const char *name)
{
    void *p = dlsym(g_h, name);
    if (!p)
        printf("  [symbol %s NOT exported]\n", name);
    return p;
}

static void progress(void *info)
{
    /* struct drvHdcProgInfo is opaque here; the callback is optional. */
    (void)info;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s <local_file> <dst_path> [dev_id]\n", argv[0]);
        return 2;
    }
    const char *file = argv[1];
    const char *dst = argv[2];
    int dev_id = argc > 3 ? atoi(argv[3]) : 0;

    g_h = dlopen("libascend_hal.so", RTLD_NOW);
    if (!g_h) {
        printf("dlopen libascend_hal.so failed: %s\n", dlerror());
        return 1;
    }

    fn_client_create client_create = (fn_client_create)sym("drvHdcClientCreate");
    fn_get_peer get_peer = (fn_get_peer)sym("drvHdcGetPeerDevId");
    fn_session_connect session_connect = (fn_session_connect)sym("drvHdcSessionConnect");
    fn_send_file send_file = (fn_send_file)sym("drvHdcSendFile");
    fn_send_file_v2 send_file_v2 = (fn_send_file_v2)sym("drvHdcSendFileV2");
    fn_send_file_v2 send_file_ex = (fn_send_file_v2)sym("drvHdcSendFileEx");
    fn_get_base_path get_base = (fn_get_base_path)sym("drvHdcGetTrustedBasePath");
    fn_load_wl load_wl = (fn_load_wl)sym("drvHdcLoadFileTransWhiteList");
    fn_client_destroy client_destroy = (fn_client_destroy)sym("drvHdcClientDestroy");
    fn_session_close session_close = (fn_session_close)sym("drvHdcSessionClose");
    if (!client_create || !get_peer || !session_connect || !send_file) {
        printf("the host libascend_hal.so does not export the HDC client API\n");
        return 1;
    }

    printf("=== HDC push: %s -> %s (dev_id %d) ===\n", file, dst, dev_id);

    int peer_devid = -1;
    int r = get_peer(0, &peer_devid);
    printf("drvHdcGetPeerDevId(node 0) ret=%d peer_devid=%d\n", r, peer_devid);
    if (r != 0)
        return 1;

    /* The host-side whitelist is what rejects a push ("File not in the white
     * list, quit.").  It is absent by default; load it before sending. */
    if (load_wl) {
        int lr = load_wl("/etc/hdcFileTransWhiteList.cfg");
        printf("drvHdcLoadFileTransWhiteList(/etc/hdcFileTransWhiteList.cfg) ret=%d\n", lr);
    }

    if (get_base) {
        char base[512];
        memset(base, 0, sizeof(base));
        int br = get_base(0, peer_devid, base, sizeof(base));
        printf("drvHdcGetTrustedBasePath ret=%d base=\"%s\"\n", br, base);
    }

    HDC_CLIENT client = NULL;
    r = client_create(&client, 1, HDC_SERVICE_TYPE_FILE_TRANS, 0);
    printf("drvHdcClientCreate ret=%d client=%p\n", r, client);
    if (r != 0 || !client)
        return 1;

    HDC_SESSION session = NULL;
    r = session_connect(0, peer_devid, client, &session);
    printf("drvHdcSessionConnect ret=%d session=%p\n", r, session);
    if (r != 0 || !session) {
        if (client_destroy)
            client_destroy(client);
        return 1;
    }

    /* Try the plain API, then the V2 variant (documented as not doing
     * whitelist validation for file names), then Ex. */
    printf("calling drvHdcSendFile ...\n");
    fflush(stdout);
    r = send_file(0, peer_devid, file, dst, progress);
    printf("drvHdcSendFile ret=%d\n", r);

    if (r != 0 && send_file_v2) {
        printf("calling drvHdcSendFileV2 ...\n");
        fflush(stdout);
        r = send_file_v2(0, peer_devid, file, dst, progress);
        printf("drvHdcSendFileV2 ret=%d\n", r);
    }
    if (r != 0 && send_file_ex) {
        printf("calling drvHdcSendFileEx ...\n");
        fflush(stdout);
        r = send_file_ex(0, peer_devid, file, dst, progress);
        printf("drvHdcSendFileEx ret=%d\n", r);
    }

    if (session_close)
        session_close(client, session);
    if (client_destroy)
        client_destroy(client);
    return r == 0 ? 0 : 1;
}
