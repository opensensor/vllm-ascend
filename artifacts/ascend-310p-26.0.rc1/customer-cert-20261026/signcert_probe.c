/*
 * Read the card's signing certificate through the host driver.
 *
 * dsmi_cmd_get_sign_cert is a real host->device command.  Its signature,
 * recovered from the disassembly at 0x702e5:
 *
 *   int dsmi_cmd_get_sign_cert(int device_id, int cert_type, int flags,
 *                               void *out_buf, uint32_t *out_len)
 *
 * with device_id <= 0xffff and both out_buf and out_len required non-NULL.
 *
 * The device-side service is drv_smbus_intf.ko, whose smbus_secure_boot_*
 * family reads certs out of the card's flash via devdrv_driver_get_cert_info
 * (drv_ascend_devmng.ko).  The device strings name the cert set:
 *   "Huawei Certification Authority1"
 *   "Huawei CSI RSA-PSS Code-Signing Authority"
 *   "Can not get cert from flash"
 *
 * Usage: signcert_probe <dev_id> [cert_type]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv)
{
    void *h = dlopen("libdrvdsmi_host.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    int dev = argc > 1 ? atoi(argv[1]) : 0;

    typedef int (*init_fn)(void);
    init_fn in = (init_fn)dlsym(h, "dsmi_init");
    if (in) printf("dsmi_init ret=%d\n", in());

    typedef int (*fn_t)(int, int, int, void *, uint32_t *);
    fn_t fn = (fn_t)dlsym(h, "dsmi_cmd_get_sign_cert");
    if (!fn) { printf("dsmi_cmd_get_sign_cert ABSENT\n"); return 1; }

    for (int type = 0; type < 6; type++) {
        static unsigned char buf[262144];
        memset(buf, 0, sizeof(buf));
        uint32_t len = sizeof(buf);
        int r = fn(dev, type, 0, buf, &len);
        printf("get_sign_cert(dev=%d type=%d) ret=%d len=%u", dev, type, r, len);
        if (r == 0 && len > 0) {
            printf("\n  hex : ");
            for (unsigned i = 0; i < 48 && i < len; i++) printf("%02x", buf[i]);
            printf("\n  text: ");
            for (unsigned i = 0; i < 160 && i < len; i++) {
                unsigned char c = buf[i];
                putchar((c >= 0x20 && c < 0x7f) ? c : '.');
            }
            printf("\n");
        } else {
            printf("\n");
        }
    }
    return 0;
}
