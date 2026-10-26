/*
 * Supply the SoC version the Ascend host HAL refuses to report.
 *
 * The problem
 * ----------
 * The AICPU package loader (CANN's libtsdclient.so) needs a short SoC version
 * to build per-SOC package names.  It asks for it in two ways:
 *
 *   halGetSocVersion(devId, buf, len)                     -> 0xfffe
 *   fe::PlatformInfoManager::GetPlatformInfos(...)        -> fails
 *
 * On this card the first one is a *stub* in the host's libascend_hal.so:
 *
 *   halGetSocVersion:
 *       mov  %edi,-0x4(%rbp)
 *       mov  %rsi,-0x10(%rbp)
 *       mov  %edx,-0x8(%rbp)
 *       mov  $0xfffe,%eax
 *       pop  %rbp
 *       ret
 *
 * so the loader falls back to generic names and never takes the per-SOC path:
 *
 *   [LoadSinglePackageToDevice] cannot find package:Ascend-aicpu_legacy.tar.gz, optional is true skip
 *   [LoadSinglePackageToDevice] cannot find package:Ascend910-aicpu_legacy.tar.gz, optional is true skip
 *
 * even though the file on disk is Ascend310P-aicpu_legacy.tar.gz.
 *
 * The interesting part: the driver *does* implement the query.  The HAL has
 * dead code for it at 0x29a57, which issues dev-manager command 0x4d97:
 *
 *   devdrv_get_dev_manager_fd(&fd, &devId, 8);
 *   devdrv_ioctl(fd, 0x4d97, buf, len);
 *
 * and drv_devmng_host.ko compares against 0x4d97 twice, so the command is
 * handled -- the HAL just never reaches it.
 *
 * The fix
 * -------
 * The card's platform is not a secret: npu-smi reports "310P3", and CANN ships
 * x86_64-linux/data/platform_config/Ascend310P3.ini with
 *
 *   SoC_version=Ascend310P3
 *   Short_SoC_version=Ascend310P
 *
 * This preload defines halGetSocVersion and answers with exactly that, read
 * from the platform ini if it can be found, so the loader takes the per-SOC
 * path, sends the omfile_name message, and the device can act on the package.
 *
 * It is loaded with LD_PRELOAD, so it overrides the stub in the vendor HAL.
 *
 * Build:
 *   gcc -shared -fPIC -o socver_shim.so socver_shim.c -ldl
 *
 * Use:
 *   LD_PRELOAD=./socver_shim.so <command>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define HAL_OK              0
#define HAL_ERR_NOT_SUPPORT 0xfffe

/* The real one, for the case where the vendor stub is not in the way. */
typedef int (*real_socver_fn)(uint32_t dev_id, char *soc_version, uint32_t len);

static int read_platform_ini(char *out, size_t out_len)
{
    static const char *roots[] = {
        "/usr/local/Ascend/latest",
        "/usr/local/Ascend/cann-9.1.0",
        "/usr/local/Ascend/cann-9.0.0",
        NULL
    };
    static const char *names[] = {
        "Ascend310P3.ini", "Ascend310P1.ini", "Ascend310P5.ini",
        "Ascend310P7.ini", "Ascend310P.ini", NULL
    };

    for (int r = 0; roots[r]; r++) {
        for (int n = 0; names[n]; n++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/x86_64-linux/data/platform_config/%s",
                     roots[r], names[n]);
            FILE *f = fopen(path, "r");
            if (!f)
                continue;
            char line[512];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "SoC_version=", 12) == 0) {
                    char *v = line + 12;
                    char *nl = strpbrk(v, "\r\n");
                    if (nl)
                        *nl = '\0';
                    snprintf(out, out_len, "%s", v);
                    fclose(f);
                    return 0;
                }
            }
            fclose(f);
        }
    }
    return -1;
}

/*
 * drvError_t halGetSocVersion(uint32_t devId, char *socVersion, uint32_t len)
 *
 * Returns the SoC version string, e.g. "Ascend310P3".
 */
int halGetSocVersion(uint32_t dev_id, char *soc_version, uint32_t len)
{
    if (soc_version == NULL || len == 0)
        return HAL_ERR_NOT_SUPPORT;

    /* If the real implementation is reachable (it is not on this build, but
     * be correct if a future one is), prefer it. */
    real_socver_fn real = (real_socver_fn)dlsym(RTLD_NEXT, "halGetSocVersion");
    if (real) {
        char probe[256];
        memset(probe, 0, sizeof(probe));
        int r = real(dev_id, probe, sizeof(probe));
        if (r == HAL_OK && probe[0] != '\0') {
            snprintf(soc_version, len, "%s", probe);
            return HAL_OK;
        }
    }

    char ver[128];
    if (read_platform_ini(ver, sizeof(ver)) != 0)
        return HAL_ERR_NOT_SUPPORT;

    snprintf(soc_version, len, "%s", ver);
    fprintf(stderr, "[socver_shim] halGetSocVersion(dev=%u) -> \"%s\"\n",
            dev_id, soc_version);
    return HAL_OK;
}

/*
 * The loader's second fallback is
 * fe::PlatformInfoManager::GetPlatformInfos(socVersion, plat, opt)
 * which reads data/platform_config/<socVersion>.ini and fills in
 * Short_SoC_version.  That path works already -- it is the *first* lookup
 * (halGetSocVersion) that fails -- but the loader's error handling gives up
 * after the first failure, so supplying the version here is enough.
 */
