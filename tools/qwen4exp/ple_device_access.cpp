// SPDX-License-Identifier: Apache-2.0
// Standalone diagnostic for a supported device subprocess launcher. Build with
// the matching ARM userspace toolchain; this does not require kernel headers.
#include <cstdio>
#include "ple_device_access.h"

int main() {
    char receipt[16384];
    PleProbeText out(receipt, sizeof(receipt));
    ple_device_access_report(out);
    if (std::fwrite(receipt, 1, out.size(), stdout) != out.size()) return 1;
    return out.truncated() ? 2 : 0;
}
