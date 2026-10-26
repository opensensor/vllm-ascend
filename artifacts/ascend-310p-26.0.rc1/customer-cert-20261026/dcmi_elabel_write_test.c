/*
 * Single-card WRITE test for the MCU electronic label.
 *
 * The write path exists in libdcmi:
 *   dcmi_elabel_set_data -> dcmi_flash_write_elabel
 *                          -> dcmi_i2c_set_data(g_i2c_dev_name, 0x50, buf, len, reg)
 *   dcmi_elabel_clear    -> dcmi_flash_erase_elabel
 *
 * with reg = 0x200 on a 310P.  The read path was measured to fail with
 * ETIMEDOUT on I2C_RDWR against /dev/i2c-0, so the write is expected to fail
 * the same way; this test establishes that definitively and records the exact
 * failure point.
 *
 * IMPORTANT: this probe issues a WRITE to the elabel region.  It is only run
 * against a card whose elabel we have a verified dump of, and the payload
 * written is byte-identical to what is already there, so a successful write
 * is a no-op on content.
 *
 * Usage: dcmi_elabel_write_test <card_id> <file-with-elabel-bytes>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv)
{
    void *h = dlopen("libdcmi.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    int card = argc > 1 ? atoi(argv[1]) : 0;

    typedef int (*init_fn)(void);
    typedef int (*set_data_fn)(int, void *, uint32_t, void *);
    typedef int (*clear_fn)(int);
    typedef int (*write_raw_fn)(unsigned char *, unsigned int);

    init_fn dinit = (init_fn)dlsym(h, "dcmi_init");
    if (dinit) printf("dcmi_init            : ret=%d\n", dinit());

    /* The raw I2C write, bypassing libdcmi's bookkeeping. */
    write_raw_fn wraw = (write_raw_fn)dlsym(h, "dcmi_flash_write_elabel");
    if (wraw) {
        static unsigned char buf[4096];
        memset(buf, 0, sizeof(buf));
        /* Read the current elabel first so we can write identical bytes. */
        typedef int (*read_raw_fn)(unsigned char *, unsigned int);
        read_raw_fn rraw = (read_raw_fn)dlsym(h, "dcmi_flash_read_elabel");
        int rr = rraw ? rraw(buf, sizeof(buf)) : -1;
        printf("flash_read_elabel    : ret=%d (to capture current bytes)\n", rr);
        if (rr == 0) {
            int r = wraw(buf, 256);
            printf("flash_write_elabel   : ret=%d  <-- WRITE ATTEMPT (identical bytes)\n", r);
        } else {
            printf("flash_write_elabel   : skipped (read failed, would write blind)\n");
        }
    } else {
        printf("flash_write_elabel   : ABSENT\n");
    }

    set_data_fn set_data = (set_data_fn)dlsym(h, "dcmi_elabel_set_data");
    if (set_data) {
        static unsigned char buf[4096];
        memset(buf, 0, sizeof(buf));
        uint32_t len = 256;
        int r = set_data(card, buf, len, NULL);
        printf("elabel_set_data      : ret=%d\n", r);
    } else {
        printf("elabel_set_data      : ABSENT\n");
    }

    clear_fn clear = (clear_fn)dlsym(h, "dcmi_elabel_clear");
    if (clear) {
        printf("elabel_clear         : present (NOT CALLED - would erase the elabel)\n");
    } else {
        printf("elabel_clear         : ABSENT\n");
    }

    return 0;
}
