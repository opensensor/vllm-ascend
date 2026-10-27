/*
 * Read and set the device's custom-signature policy.
 *
 * The device's tsdaemon reads its signature policy from
 *
 *     /device-sw-plugin/verify/flag
 *
 * which holds one line, "verify_flag=<N>", where N is:
 *
 *     SIGN_FLAG_DISABLED       = 0
 *     SIGN_FLAG_ONLY_VENDOR    = 1
 *     SIGN_FLAG_ONLY_USER      = 2
 *     SIGN_FLAG_VENDOR_OR_USER  = 3
 *     SIGN_FLAG_ONLY_CMTY      = 4
 *     SIGN_FLAG_VENDOR_OR_CMTY = 5   <- what the driver writes by default
 *     SIGN_FLAG_USER_OR_CMTY   = 6
 *     SIGN_FLAG_ALL            = 7
 *
 * The host driver reaches the device's copy through the DMP command set
 * ("main_cmd=0x35,sub_cmd=0x2"), and libdrvdsmi_host.so exports the wrappers
 * dsmi_cmd_get_custom_sign_flag / dsmi_cmd_set_custom_sign_flag.
 *
 * This calls them.  With --set it writes the flag; without, it only reads.
 *
 * Usage:
 *   signflag <device_id>                 read the current policy
 *   signflag <device_id> <flag>          set it (0 = DISABLED)
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* The command payload is a struct dms_hal_device_info_stru:
 *   u32 dev_id; u32 buff_size; u8 payload[];
 * with the flag in payload as an unsigned int. */
struct hal_dev_info {
    uint32_t dev_id;
    uint32_t buff_size;
    uint32_t payload[16];
};

typedef int (*fn_get)(uint32_t, uint32_t, uint32_t);
typedef int (*fn_set)(uint32_t, uint32_t, uint32_t);

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <device_id> [flag]\n", argv[0]);
        return 2;
    }
    uint32_t dev = (uint32_t)strtoul(argv[1], NULL, 10);

    void *h = dlopen("libdrvdsmi_host.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }

    fn_get get_fn = (fn_get)dlsym(h, "dsmi_cmd_get_custom_sign_flag");
    fn_set set_fn = (fn_set)dlsym(h, "dsmi_cmd_set_custom_sign_flag");
    printf("get=%p set=%p\n", (void *)get_fn, (void *)set_fn);
    if (!get_fn) return 1;

    struct hal_dev_info in;
    memset(&in, 0, sizeof(in));
    in.dev_id = dev;
    in.buff_size = sizeof(uint32_t);

    int r = get_fn(dev, 0 /* feature */, (uint32_t)(uintptr_t)&in);
    printf("dsmi_cmd_get_custom_sign_flag -> %d (0x%x)\n", r, r);
    if (r == 0) {
        printf("  current flag = %u\n", in.payload[0]);
    }

    if (argc >= 3) {
        uint32_t flag = (uint32_t)strtoul(argv[2], NULL, 10);
        struct hal_dev_info si;
        memset(&si, 0, sizeof(si));
        si.dev_id = dev;
        si.buff_size = sizeof(uint32_t);
        si.payload[0] = flag;
        if (set_fn) {
            int rs = set_fn(dev, 0, (uint32_t)(uintptr_t)&si);
            printf("dsmi_cmd_set_custom_sign_flag(%u) -> %d (0x%x)\n", flag, rs, rs);
        } else {
            printf("dsmi_cmd_set_custom_sign_flag not exported\n");
        }
    }
    return 0;
}
