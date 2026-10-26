/*
 * A minimal host-side HDC client for the Ascend 310P.
 *
 * There is no HDC client library on the host, but the host driver -- which we
 * own and rebuild -- exposes a complete RPC to the device OS over PCIe:
 *
 *   /dev/hisi_hdc  ioctl(cmd, &union hdcdrv_cmd)
 *
 * The ioctl number is _KA_IO('H', HDCDRV_CMD_xxx), i.e. the standard _IOC
 * encoding with type 'H' (HDCDRV_CMD_MAGIC) and the command number as _NR.
 * The driver dispatches on _KA_IOC_NR(cmd) and copies a union hdcdrv_cmd in
 * (and back out when copy_flag is set).
 *
 * The union is large; only the members this client uses are declared, with
 * explicit padding so the layout matches.  All the offsets below are taken
 * from hdc_host/hdcdrv_cmd.h.
 *
 * The connection sequence, per hdcdrv_core.c:
 *
 *   HDCDRV_CMD_GET_PEER_DEV_ID  -> peer_dev_id
 *   HDCDRV_CMD_SERVER_CREATE    -> creates a server for service_type
 *   HDCDRV_CMD_CONNECT          -> returns a session id
 *   HDCDRV_CMD_SEND / _RECV     -> message traffic
 *   HDCDRV_CMD_CLOSE            -> tear down
 *
 * Usage:
 *   hdc_client peer                       print the peer's device id
 *   hdc_client connect <service_type>     open a session and report the id
 *   hdc_client ping <service_type>        connect, send a small message, report
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define HDC_DEV "/dev/hisi_hdc"
#define HDCDRV_CMD_MAGIC 'H'

/* Command numbers, from hdc_host/hdcdrv_cmd_enum.h */
#define HDCDRV_CMD_SERVER_WAKEUP_WAIT   0x04
#define HDCDRV_CMD_CLIENT_WAKEUP_WAIT   0x05
#define HDCDRV_CMD_CLIENT_DESTROY       0x06
#define HDCDRV_CMD_GET_PEER_DEV_ID      0x07
#define HDCDRV_CMD_CONFIG               0x08
#define HDCDRV_CMD_SET_SERVICE_LEVEL    0x09
#define HDCDRV_CMD_SERVER_CREATE        0x10
#define HDCDRV_CMD_SERVER_DESTROY       0x11
#define HDCDRV_CMD_ACCEPT               0x12
#define HDCDRV_CMD_CONNECT              0x13
#define HDCDRV_CMD_CLOSE                0x14
#define HDCDRV_CMD_SEND                 0x15
#define HDCDRV_CMD_RECV_PEEK            0x16
#define HDCDRV_CMD_RECV                 0x17

/* Service types */
#define HDCDRV_SERVICE_TYPE_DMP        0
#define HDCDRV_SERVICE_TYPE_PROFILING  1
#define HDCDRV_SERVICE_TYPE_FILE_TRANS 3
#define HDCDRV_SERVICE_TYPE_UPGRADE    17

/* The union is at least as large as its biggest member.  Declaring the
 * members we use with explicit padding keeps the layout honest. */
struct hdcdrv_cmd_common {
    int ret;
    int dev_id;
    unsigned long long pid;
};

struct hdcdrv_cmd_get_peer_dev_id {
    int ret;
    int dev_id;
    unsigned long long pid;
    int peer_dev_id;
};

struct hdcdrv_cmd_server_create {
    int ret;
    int dev_id;
    unsigned long long pid;
    int service_type;
};

struct hdcdrv_cmd_connect {
    int peer_pid;
    int dev_id;
    unsigned long long pid;
    int service_type;
    int session;
    unsigned int timeout;
    unsigned int session_cur_alloc_idx;
};

struct hdcdrv_cmd_close {
    int ret;
    int dev_id;
    unsigned long long pid;
    int session;
    unsigned int unique_val;
    unsigned long long task_start_time;
    unsigned int session_cur_alloc_idx;
};

struct hdcdrv_cmd_send {
    int ret;
    int dev_id;
    unsigned long long pid;
    int session;
    void *src_buf;
    int len;
    void *pool_buf;
    unsigned long long pool_addr;
    int wait_flag;
    unsigned int timeout;
};

struct hdcdrv_cmd_recv_peek {
    int ret;
    int dev_id;
    unsigned long long pid;
    int session;
    int len;
    int wait_flag;
    unsigned int timeout;
    int group_flag;
    int count;
    unsigned int session_cur_alloc_idx;
};

struct hdcdrv_cmd_recv {
    int ret;
    int dev_id;
    unsigned long long pid;
    int session;
    void *dst_buf;
    int len;
    void *pool_buf;
    int out_len;
    int buf_count;
    int group_flag;
    void *buf_list[32];
    unsigned int buf_len[32];
};

/* Large enough for any member we use. */
static union {
    struct hdcdrv_cmd_common common;
    struct hdcdrv_cmd_get_peer_dev_id peer;
    struct hdcdrv_cmd_server_create server_create;
    struct hdcdrv_cmd_connect connect;
    struct hdcdrv_cmd_close close_cmd;
    struct hdcdrv_cmd_send send;
    struct hdcdrv_cmd_recv_peek recv_peek;
    struct hdcdrv_cmd_recv recv;
    char pad[4096];
} g_cmd;

static unsigned int hdc_ioctl_nr(int nr)
{
    return _IOC(_IOC_NONE, HDCDRV_CMD_MAGIC, nr, sizeof(g_cmd));
}

static int hdc_open(void)
{
    int fd = open(HDC_DEV, O_RDWR);
    if (fd < 0)
        printf("open %s failed: %s\n", HDC_DEV, strerror(errno));
    return fd;
}

static int hdc_peer_dev_id(int fd, int dev_id)
{
    memset(&g_cmd, 0, sizeof(g_cmd));
    g_cmd.peer.dev_id = dev_id;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_GET_PEER_DEV_ID), &g_cmd) < 0) {
        printf("GET_PEER_DEV_ID ioctl failed: %s\n", strerror(errno));
        return -1;
    }
    if (g_cmd.common.ret != 0) {
        printf("GET_PEER_DEV_ID ret=%d\n", g_cmd.common.ret);
        return -1;
    }
    return g_cmd.peer.peer_dev_id;
}

static int hdc_server_create(int fd, int dev_id, int service_type)
{
    memset(&g_cmd, 0, sizeof(g_cmd));
    g_cmd.server_create.dev_id = dev_id;
    g_cmd.server_create.service_type = service_type;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_SERVER_CREATE), &g_cmd) < 0) {
        printf("SERVER_CREATE ioctl failed: %s\n", strerror(errno));
        return -1;
    }
    if (g_cmd.common.ret != 0) {
        printf("SERVER_CREATE ret=%d\n", g_cmd.common.ret);
        return -1;
    }
    return 0;
}

static int hdc_connect(int fd, int dev_id, int service_type, int *out_session)
{
    memset(&g_cmd, 0, sizeof(g_cmd));
    g_cmd.connect.dev_id = dev_id;
    g_cmd.connect.service_type = service_type;
    g_cmd.connect.timeout = 5000;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_CONNECT), &g_cmd) < 0) {
        printf("CONNECT ioctl failed: %s\n", strerror(errno));
        return -1;
    }
    if (g_cmd.connect.peer_pid != 0) {
        printf("CONNECT peer_pid=%d (a peer is already bound)\n", g_cmd.connect.peer_pid);
    }
    if (g_cmd.common.ret != 0) {
        printf("CONNECT ret=%d\n", g_cmd.common.ret);
        return -1;
    }
    *out_session = g_cmd.connect.session;
    return 0;
}

static int hdc_close(int fd, int dev_id, int session)
{
    memset(&g_cmd, 0, sizeof(g_cmd));
    g_cmd.close_cmd.dev_id = dev_id;
    g_cmd.close_cmd.session = session;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_CLOSE), &g_cmd) < 0) {
        printf("CLOSE ioctl failed: %s\n", strerror(errno));
        return -1;
    }
    printf("CLOSE ret=%d\n", g_cmd.common.ret);
    return 0;
}

int main(int argc, char **argv)
{
    const char *cmd = argc > 1 ? argv[1] : "peer";
    int dev_id = argc > 2 ? atoi(argv[2]) : 0;
    int service = argc > 2 ? atoi(argv[2]) : HDCDRV_SERVICE_TYPE_FILE_TRANS;

    int fd = hdc_open();
    if (fd < 0)
        return 1;

    if (strcmp(cmd, "peer") == 0) {
        int peer = hdc_peer_dev_id(fd, dev_id);
        if (peer >= 0)
            printf("peer_dev_id = %d\n", peer);
        close(fd);
        return peer >= 0 ? 0 : 1;
    }

    if (strcmp(cmd, "connect") == 0) {
        int peer = hdc_peer_dev_id(fd, dev_id);
        if (peer < 0) { close(fd); return 1; }
        printf("peer_dev_id = %d\n", peer);

        if (hdc_server_create(fd, dev_id, service) != 0) { close(fd); return 1; }
        printf("server created for service %d\n", service);

        int session = -1;
        if (hdc_connect(fd, dev_id, service, &session) != 0) { close(fd); return 1; }
        printf("CONNECTED: session=%d service=%d\n", session, service);

        hdc_close(fd, dev_id, session);
        close(fd);
        return 0;
    }

    printf("usage: %s peer [dev_id]\n"
           "       %s connect <service_type> [dev_id]\n", argv[0], argv[0]);
    close(fd);
    return 2;
}
