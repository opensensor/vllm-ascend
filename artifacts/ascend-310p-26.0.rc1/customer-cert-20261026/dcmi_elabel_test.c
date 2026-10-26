/*
 * Single-card test: can the host read (and then write) the card's electronic
 * label -- the MCU record that the firmware protects with a CRC-16 instead of
 * a signature?
 *
 * The chain, recovered from libdcmi.so:
 *
 *   npu-smi  ->  dcmi_get_card_elabel_v2
 *             ->  dcmi_elabel_init
 *             ->  dcmi_flash_read_elabel(buf, len)
 *             ->  dcmi_i2c_get_data(g_i2c_dev_name, 0x50, buf, len, reg)
 *
 * where `reg` is 0x200 on a 310P and 0x280 on a 910B:
 *
 *   call dcmi_board_chip_type_is_ascend_310b   ; 1 iff chip type == 4
 *   sbb  %rdx, %rdx                           ; -1 on 910B, 0 otherwise
 *   and  $0xfffffffffffffe80, %rdx
 *   add  $0x200, %rdx
 *
 * The write path is the mirror image and also exists:
 *   dcmi_elabel_set_data  ->  dcmi_flash_write_elabel  ->  dcmi_i2c_set_data(..., 0x50, ...)
 *   dcmi_elabel_clear    ->  dcmi_flash_erase_elabel
 *
 * This probe exercises the READ path only. It is safe: no write is issued.
 *
 * Usage: dcmi_elabel_test <card_id>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv)
{
    void *h = dlopen("libdcmi.so", RTLD_NOW);
    if (!h) { printf("dlopen libdcmi.so failed: %s\n", dlerror()); return 1; }
    int card = argc > 1 ? atoi(argv[1]) : 0;

    typedef int (*set_name_fn)(const char *);
    typedef int (*init_fn)(void);
    typedef int (*read_fn)(unsigned char *, unsigned int);
    typedef int (*card_fn)(int, void *, uint32_t *);
    typedef int (*chip_fn)(void);

    set_name_fn set_name = (set_name_fn)dlsym(h, "dcmi_set_i2c_dev_name");
    init_fn elabel_init = (init_fn)dlsym(h, "dcmi_elabel_init");
    read_fn read_elabel = (read_fn)dlsym(h, "dcmi_flash_read_elabel");
    card_fn card_elabel = (card_fn)dlsym(h, "dcmi_get_card_elabel_v2");
    chip_fn chip_type = (chip_fn)dlsym(h, "dcmi_get_board_chip_type");

    printf("=== elabel reachability, card %d ===\n", card);

    /* libdcmi needs its own environment initialised before any device access. */
    typedef int (*dcmi_init_fn)(void);
    dcmi_init_fn dinit = (dcmi_init_fn)dlsym(h, "dcmi_init");
    if (dinit) {
        int r = dinit();
        printf("dcmi_init            : ret=%d\n", r);
    } else {
        printf("dcmi_init            : ABSENT\n");
    }

    if (chip_type) printf("chip type            : %d (1=310P, 4=910B)\n", chip_type());

    if (set_name) {
        int r = set_name("/dev/i2c-0");
        printf("set_i2c_dev_name     : ret=%d (/dev/i2c-0)\n", r);
    } else {
        printf("set_i2c_dev_name     : ABSENT\n");
    }

    if (read_elabel) {
        static unsigned char buf[4096];
        memset(buf, 0, sizeof(buf));
        int r = read_elabel(buf, sizeof(buf));
        printf("flash_read_elabel    : ret=%d\n", r);
        if (r == 0) {
            printf("   hex : ");
            for (unsigned i = 0; i < 64; i++) printf("%02x", buf[i]);
            printf("\n   text: ");
            for (unsigned i = 0; i < 64; i++) {
                unsigned char c = buf[i];
                putchar((c >= 0x20 && c < 0x7f) ? c : '.');
            }
            printf("\n");
        }
    } else {
        printf("flash_read_elabel    : ABSENT\n");
    }

    if (elabel_init) {
        int r = elabel_init();
        printf("dcmi_elabel_init     : ret=%d\n", r);
    }

    if (card_elabel) {
        static unsigned char buf[65536];
        memset(buf, 0, sizeof(buf));
        uint32_t len = sizeof(buf);
        int r = card_elabel(card, buf, &len);
        printf("get_card_elabel_v2   : ret=%d len=%u\n", r, len);
        if (r == 0 && len > 0) {
            printf("   hex : ");
            for (unsigned i = 0; i < 96 && i < len; i++) printf("%02x", buf[i]);
            printf("\n   text: ");
            for (unsigned i = 0; i < 160 && i < len; i++) {
                unsigned char c = buf[i];
                putchar((c >= 0x20 && c < 0x7f) ? c : '.');
            }
            printf("\n");
        }
    } else {
        printf("get_card_elabel_v2   : ABSENT\n");
    }

    return 0;
}
