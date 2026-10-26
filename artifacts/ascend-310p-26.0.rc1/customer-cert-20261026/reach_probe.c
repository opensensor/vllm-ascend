/*
 * Live reachability test against ONE card.
 *
 * Each candidate entry point is exercised in a forked child, so a crash in
 * one (which several of these do, because they are stubs reached through a
 * PLT trampoline with a different signature than the header suggests) cannot
 * stop the others from being measured.
 *
 * The cryptographic half of the question is already settled:
 *   - the MCU electronic label and customer-certificate regions are protected
 *     by CRC-16/XMODEM only  (mcu_elabel.py: 21/21 items verify)
 *   - the MCU firmware is protected by an RSA signature anchored on
 *     sha256(vendor_pubkey) = ab510f4a...c5074 at 0x08032be4
 * so the question here is purely: can the host reach those regions at all?
 *
 * Usage: reach_probe <dev_id>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

typedef enum { DSMI_CONTENT_RTC_CONFIG = 0 } flash_content_type;

typedef struct {
    flash_content_type type;
    unsigned char *buf;
    unsigned int size;
    unsigned int offset;
    unsigned int reserved[4];
} dsmi_flash_content;

/* Run fn() in a child; report its exit status and captured stdout. */
static int run_child(void (*fn)(void), const char *label)
{
    int fds[2];
    if (pipe(fds) != 0) { printf("%-34s pipe failed\n", label); return -1; }
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return -1; }
    if (pid == 0) {
        close(fds[0]);
        /* redirect to a per-probe temp file so concurrent output cannot
         * interleave and so a crash still leaves what was printed */
        char path[128];
        snprintf(path, sizeof(path), "/tmp/rp_%d.txt", (int)getpid());
        FILE *f = freopen(path, "w", stdout);
        if (f) { setvbuf(f, NULL, _IONBF, 0); }
        fn();
        fflush(stdout);
        _exit(0);
    }
    close(fds[1]);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    char path[128];
    snprintf(path, sizeof(path), "/tmp/rp_%d.txt", (int)pid);
    FILE *f = fopen(path, "r");
    static char buf[4096];
    ssize_t n = 0;
    if (f) { n = fread(buf, 1, sizeof(buf) - 1, f); fclose(f); }
    unlink(path);
    if (n > 0) buf[n] = '\0';
    if (WIFSIGNALED(status)) {
        printf("%-34s CRASHED (signal %d)\n", label, WTERMSIG(status));
        if (n > 0) printf("    %s", buf);
        return -1;
    }
    printf("%-34s %s", label, buf);
    return 0;
}

static int g_dev;
static void *g_h;

static void probe_get_flash(void)
{
    typedef int (*fn_t)(int, dsmi_flash_content);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_get_flash_content");
    if (!fn) { printf("ABSENT\n"); return; }
    static unsigned char buf[1024];
    memset(buf, 0, sizeof(buf));
    dsmi_flash_content c;
    memset(&c, 0, sizeof(c));
    c.type = DSMI_CONTENT_RTC_CONFIG;
    c.buf = buf;
    c.size = sizeof(buf);
    c.offset = 0;
    int r = fn(g_dev, c);
    printf("ret=0x%x", r);
    if (r == 0) {
        printf("  data=");
        for (unsigned i = 0; i < 32; i++) printf("%02x", buf[i]);
    }
    printf("\n");
}

static void probe_set_flash(void)
{
    typedef int (*fn_t)(int, dsmi_flash_content);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_set_flash_content");
    if (!fn) { printf("ABSENT (no write path)\n"); return; }
    static unsigned char buf[1024];
    memset(buf, 0, sizeof(buf));
    dsmi_flash_content c;
    memset(&c, 0, sizeof(c));
    c.type = DSMI_CONTENT_RTC_CONFIG;
    c.buf = buf;
    c.size = sizeof(buf);
    c.offset = 0;
    printf("ret=0x%x (write path present, not exercised)\n", fn(g_dev, c));
}

static void probe_elable(void)
{
    typedef int (*fn_t)(int, int, char *, int *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_dft_get_elable");
    if (!fn) { printf("ABSENT\n"); return; }
    static char buf[16384];
    memset(buf, 0, sizeof(buf));
    int len = (int)sizeof(buf);
    int r = fn(g_dev, 0, buf, &len);
    printf("ret=0x%x len=%d\n", r, len);
}

static void probe_elabel_cmd(void)
{
    typedef int (*fn_t)(int, unsigned char, char *, int *, void *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_cmd_dft_get_elabel");
    if (!fn) { printf("ABSENT\n"); return; }
    static char buf[16384];
    memset(buf, 0, sizeof(buf));
    int len = (int)sizeof(buf);
    int r = fn(g_dev, 0, buf, &len, NULL);
    printf("ret=0x%x len=%d\n", r, len);
}

static void probe_secverify_cert(void)
{
    typedef int (*fn_t)(int, int, void *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_get_custom_op_secverify_cert_show_info");
    if (!fn) { printf("ABSENT\n"); return; }
    static char buf[4096];
    memset(buf, 0, sizeof(buf));
    int r = fn(g_dev, 0, buf);
    printf("ret=0x%x\n", r);
}

static void probe_sign_cert(void)
{
    typedef int (*fn_t)(int, char *, uint32_t *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_cmd_get_sign_cert");
    if (!fn) { printf("ABSENT\n"); return; }
    static char buf[4096];
    memset(buf, 0, sizeof(buf));
    uint32_t len = sizeof(buf);
    int r = fn(g_dev, buf, &len);
    printf("ret=0x%x len=%u", r, len);
    if (r == 0 && len > 0) {
        printf("  data=");
        for (unsigned i = 0; i < 48 && i < len; i++) printf("%02x", (unsigned char)buf[i]);
    }
    printf("\n");
}

static void probe_rootkey(void)
{
    typedef int (*fn_t)(int, int, char *, uint32_t *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_get_rootkey_status");
    if (!fn) { printf("ABSENT\n"); return; }
    static char buf[4096];
    memset(buf, 0, sizeof(buf));
    uint32_t len = sizeof(buf);
    int r = fn(g_dev, 0, buf, &len);
    printf("ret=%d len=%u\n", r, len);
}

static void probe_board_id(void)
{
    typedef int (*fn_t)(int, uint32_t *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_get_board_id");
    if (!fn) { printf("ABSENT\n"); return; }
    uint32_t id = 0;
    int r = fn(g_dev, &id);
    printf("ret=%d board_id=0x%x\n", r, id);
}

static void probe_systime(void)
{
    typedef int (*fn_t)(int, uint32_t *);
    fn_t fn = (fn_t)dlsym(g_h, "dsmi_get_system_time");
    if (!fn) { printf("ABSENT\n"); return; }
    uint32_t t = 0;
    int r = fn(g_dev, &t);
    printf("ret=%d time=0x%x\n", r, t);
}

int main(int argc, char **argv)
{
    g_h = dlopen("libdrvdsmi_host.so", RTLD_NOW);
    if (!g_h) { printf("dlopen failed\n"); return 1; }
    g_dev = argc > 1 ? atoi(argv[1]) : 0;

    printf("=== reachability probe, dev %d ===\n", g_dev);
    printf("(each probe runs in a forked child; CRASHED = wrong signature or stub)\n\n");

    run_child(probe_board_id,       "dsmi_get_board_id");
    run_child(probe_systime,        "dsmi_get_system_time");
    run_child(probe_get_flash,      "dsmi_get_flash_content");
    run_child(probe_set_flash,      "dsmi_set_flash_content");
    run_child(probe_elable,         "dsmi_dft_get_elable");
    run_child(probe_elabel_cmd,     "dsmi_cmd_dft_get_elabel");
    run_child(probe_secverify_cert, "custom_secverify_cert_show_info");
    run_child(probe_sign_cert,      "dsmi_cmd_get_sign_cert");
    run_child(probe_rootkey,        "dsmi_get_rootkey_status");

    return 0;
}
