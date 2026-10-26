/*
 * Compute the AICPU package "check code" with the vendor's own function.
 *
 * The device gates the AICPU package install on a checksum comparison:
 *
 *   [TsdClient] hostCheckCode[%u] no equal to deviceCheckCode[%u], begin send file[%s] ...
 *   [TSDaemon]  Check sum, curCode=%lu, checkCode=%lu, deviceId=%u, vfId=%u
 *   [TSDaemon]  Check sum wait %u second timeout.
 *
 * The host computes it with tsd::CalFileSha256HashValue(const std::string &path),
 * which libtsdclient.so exports.  Calling it directly guarantees we produce
 * exactly the value the device expects -- no need to reverse the algorithm.
 *
 * Usage: checkcode_probe <file> [<file> ...]
 */
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <string>

/* tsd::CalFileSha256HashValue(const std::string &) -> std::string (32 raw bytes) */
typedef std::string (*fn_cal_sha)(const std::string &);

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::printf("usage: %s <file> [<file> ...]\n", argv[0]);
        return 2;
    }
    void *h = dlopen("libtsdclient.so", RTLD_NOW);
    if (!h) {
        std::printf("dlopen libtsdclient.so failed: %s\n", dlerror());
        return 1;
    }
    auto cal = (fn_cal_sha)dlsym(h, "_ZN3tsd22CalFileSha256HashValueERKSs");
    if (!cal) {
        std::printf("tsd::CalFileSha256HashValue not exported\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        std::string path(argv[i]);
        std::string v = cal(path);
        std::printf("%s\n", argv[i]);
        std::printf("  len=%zu  hex=", v.size());
        for (unsigned char c : v) std::printf("%02x", c);
        std::printf("\n");
    }
    return 0;
}
