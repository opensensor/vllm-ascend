/*
 * Probe the card's GPIO pins through the public driver API.
 *
 * dsmi_get_gpio_status(dev_id, gpio_num, &status) is registered in the driver
 * as DMS_GET_GET_GPIO_STATUS_CMD with DMS_SUPPORT_ROOT_ONLY:
 *
 *   dms_get_gpio_status():
 *       gpio_num = arg->gpio_num;
 *       if (!gpio_is_valid(gpio_num))  -> "Invalid GPIO. (num=%u)"
 *       ret = gpio_request(gpio_num, DEVDRV_GPIO_NAME);
 *       gpio_val = gpio_get_value(gpio_num);
 *       gpio_free(gpio_num);
 *
 * The HAL's dmanage_i2c_gpio_read() uses the same mechanism, so if the GPIO
 * block is exposed on the 310P this is the primitive a software I2C would
 * need.  Read-only by design: there is no exported set_gpio.
 *
 * Usage: gpio_probe <dev_id> [gpio_start] [gpio_count]
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
    int start = argc > 2 ? atoi(argv[2]) : 0;
    int count = argc > 3 ? atoi(argv[3]) : 32;

    typedef int (*init_fn)(void);
    init_fn in = (init_fn)dlsym(h, "dsmi_init");
    if (in) printf("dsmi_init ret=%d\n", in());

    typedef int (*gpio_fn)(int, unsigned int, unsigned int *);
    gpio_fn fn = (gpio_fn)dlsym(h, "dsmi_get_gpio_status");
    if (!fn) { printf("dsmi_get_gpio_status ABSENT\n"); return 1; }

    int ok = 0;
    for (unsigned g = start; g < start + count; g++) {
        unsigned int status = 0xffffffff;
        int r = fn(dev, g, &status);
        if (r == 0) {
            printf("  gpio %3u : ret=0 status=%u\n", g, status);
            ok++;
        }
        if (g < start + 8) {
            printf("  gpio %3u : ret=%d status=0x%x\n", g, r, status);
        }
    }
    printf("\n%d/%d GPIOs readable\n", ok, count);
    return 0;
}
