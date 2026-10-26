/*
 * Read the card's vNIC IP addresses through the driver.
 *
 * The host<->device link is a PCIe vNIC (endvnic) that carries only IPv6
 * link-local.  The driver exposes:
 *   dsmi_get_device_ip_address(dev, port_type, port_id, &ip, &mask)
 *   dsmi_get_device_ip_address_from_udis(dev, port_type, port_id, &ip, &mask)
 *
 * Usage: ip_probe <dev_id>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>

typedef unsigned char *ip_addr_t;   /* pointer to a 16-byte address */

int main(int argc, char **argv)
{
    void *h = dlopen("libdrvdsmi_host.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    int dev = argc > 1 ? atoi(argv[1]) : 0;

    typedef int (*init_fn)(void);
    init_fn in = (init_fn)dlsym(h, "dsmi_init");
    if (in) printf("dsmi_init ret=%d\n", in());

    typedef int (*getip_fn)(int, int, int, unsigned char *, unsigned char *);
    getip_fn get_ip = (getip_fn)dlsym(h, "dsmi_get_device_ip_address");
    getip_fn get_udis = (getip_fn)dlsym(h, "dsmi_get_device_ip_address_from_udis");

    fprintf(stderr, "start\n"); fflush(stderr);
    for (int port = 0; port < 2; port++) {
        for (int pid = 0; pid < 2; pid++) {
            unsigned char ip[16], mask[16];
            memset(ip, 0, sizeof(ip));
            memset(mask, 0, sizeof(mask));
            fprintf(stderr, "  calling get_ip port=%d pid=%d\n", port, pid); fflush(stderr);
            int r = get_ip ? get_ip(dev, port, pid, ip, mask) : -1;
            fprintf(stderr, "  returned %d\n", r); fflush(stderr);
            char ips[INET6_ADDRSTRLEN] = "?";
            inet_ntop(AF_INET6, ip, ips, sizeof(ips));
            char ms[INET6_ADDRSTRLEN] = "?";
            inet_ntop(AF_INET6, mask, ms, sizeof(ms));
            printf("get_device_ip_address(port=%d pid=%d) ret=%d ip=%s mask=%s\n",
                   port, pid, r, ips, ms);

            /* from_udis crashes on this build -- skip it. */
            (void)get_udis;
        }
    }
    return 0;
}
