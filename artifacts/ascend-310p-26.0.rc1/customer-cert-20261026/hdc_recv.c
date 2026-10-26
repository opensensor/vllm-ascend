/*
 * Read whatever the device sent back over HDC.
 *
 * The device's hdcd file-transfer agent replies with a FILE_OPT message
 * ("Calling send_reply error.", "Calling send_reply FILE_OPT_NOSPC error.",
 * "Calling send_reply FILE_OPT_PERDE error."), and the AICPU loader blocks in
 * hdcdrv_recv_peek waiting for it.  This opens the same session and reads
 * pending messages, so we can see whether the device replies at all and what
 * it says.
 *
 * Usage: hdc_recv <dev_id> [timeout_ms]
 */
#include <dlfcn.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define HDC_DEV "/dev/hisi_hdc"
#define HDCDRV_CMD_MAGIC 'H'

#define HDCDRV_CMD_GET_PEER_DEV_ID 0x07
#define HDCDRV_CMD_SERVER_CREATE   0x10
#define HDCDRV_CMD_CONNECT         0x13
#define HDCDRV_CMD_RECV_PEEK       0x16
#define HDCDRV_CMD_RECV            0x17
#define HDCDRV_CMD_CLOSE           0x14

#define HDCDRV_SERVICE_TYPE_FILE_TRANS 3
#define HDCDRV_SERVICE_TYPE_TSD 9

struct hdcdrv_cmd_common {
    int ret;
    int dev_id;
    unsigned long long pid;
};

struct hdcdrv_cmd_peer {
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

struct hdcdrv_cmd_close {
    int ret;
    int dev_id;
    unsigned long long pid;
    int session;
    unsigned int unique_val;
    unsigned long long task_start_time;
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

/* the ioctl size must match the union hdcdrv_cmd the driver copies */
#define HDC_CMD_SIZE 4096

static unsigned int hdc_ioctl_nr(int nr)
{
    return _IOC(_IOC_NONE, HDCDRV_CMD_MAGIC, nr, HDC_CMD_SIZE);
}

int main(int argc, char **argv)
{
    int dev_id = argc > 1 ? atoi(argv[1]) : 0;
    int timeout_ms = argc > 2 ? atoi(argv[2]) : 5000;
    int service_type = argc > 3 ? atoi(argv[3]) : HDCDRV_SERVICE_TYPE_FILE_TRANS;

    int fd = open(HDC_DEV, O_RDWR);
    if (fd < 0) { printf("open %s: %s\n", HDC_DEV, strerror(errno)); return 1; }

    /* the ioctl copies a union hdcdrv_cmd in/out; its size is the driver's */
    static unsigned char cmd[4096];
    memset(cmd, 0, sizeof(cmd));

    struct hdcdrv_cmd_peer *peer = (struct hdcdrv_cmd_peer *)cmd;
    peer->dev_id = dev_id;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_GET_PEER_DEV_ID), cmd) < 0) {
        printf("GET_PEER_DEV_ID: %s\n", strerror(errno));
        return 1;
    }
    printf("peer_dev_id = %d (ret %d) service_type=%d\n", peer->peer_dev_id, peer->ret, service_type);
    int peer_id = peer->peer_dev_id;

    memset(cmd, 0, sizeof(cmd));
    struct hdcdrv_cmd_server_create *sc = (struct hdcdrv_cmd_server_create *)cmd;
    sc->dev_id = dev_id;
    sc->service_type = service_type;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_SERVER_CREATE), cmd) < 0) {
        printf("SERVER_CREATE: %s\n", strerror(errno));
        return 1;
    }
    printf("SERVER_CREATE ret=%d\n", sc->ret);

    memset(cmd, 0, sizeof(cmd));
    struct hdcdrv_cmd_connect *cn = (struct hdcdrv_cmd_connect *)cmd;
    cn->dev_id = dev_id;
    cn->service_type = service_type;
    cn->timeout = timeout_ms;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_CONNECT), cmd) < 0) {
        printf("CONNECT: %s\n", strerror(errno));
        return 1;
    }
    printf("CONNECT ret=%d session=%d peer_pid=%d\n", cn->peer_pid, cn->session, cn->peer_pid);
    int session = cn->session;

    memset(cmd, 0, sizeof(cmd));
    struct hdcdrv_cmd_recv_peek *pk = (struct hdcdrv_cmd_recv_peek *)cmd;
    pk->dev_id = dev_id;
    pk->session = session;
    pk->len = 4096;
    pk->wait_flag = 0;
    pk->timeout = timeout_ms;
    pk->group_flag = 0;
    pk->count = 1;
    if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_RECV_PEEK), cmd) < 0) {
        printf("RECV_PEEK: %s\n", strerror(errno));
        return 1;
    }
    printf("RECV_PEEK ret=%d len=%d count=%d\n", pk->ret, pk->len, pk->count);

    if (pk->ret == 0 && pk->len > 0) {
        static unsigned char buf[65536];
        memset(cmd, 0, sizeof(cmd));
        struct hdcdrv_cmd_recv *rc = (struct hdcdrv_cmd_recv *)cmd;
        rc->dev_id = dev_id;
        rc->session = session;
        rc->dst_buf = buf;
        rc->len = pk->len;
        rc->buf_count = 1;
        rc->group_flag = 0;
        rc->buf_list[0] = buf;
        rc->buf_len[0] = pk->len;
        if (ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_RECV), cmd) < 0)
            printf("RECV: %s\n", strerror(errno));
        else {
            printf("RECV ret=%d out_len=%d\n", rc->ret, rc->out_len);
            printf("  hex : ");
            for (int i = 0; i < rc->out_len && i < 128; i++) printf("%02x", buf[i]);
            printf("\n  text: ");
            for (int i = 0; i < rc->out_len && i < 256; i++) {
                unsigned char c = buf[i];
                putchar((c >= 0x20 && c < 0x7f) ? c : '.');
            }
            printf("\n");
        }
    }

    memset(cmd, 0, sizeof(cmd));
    struct hdcdrv_cmd_close *cl = (struct hdcdrv_cmd_close *)cmd;
    cl->dev_id = dev_id;
    cl->session = session;
    ioctl(fd, hdc_ioctl_nr(HDCDRV_CMD_CLOSE), cmd);
    close(fd);
    return 0;
}
