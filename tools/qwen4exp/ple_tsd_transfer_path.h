// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

// Deliberately limited to the private, single-graph DataFlow access probe.
// The device inotify loader recognizes the basename in both trusted roots.
inline bool ple_tsd_transfer_path(const std::string& source,
                                  const std::string& destination,
                                  const std::string& legacy_root,
                                  const std::string& v2_root, unsigned long pid,
                                  std::string& redirected) {
    constexpr const char* kProbePackage = "0_Ascend-flow_model.tar.gz";
    redirected.clear();
    if (pid == 0 || legacy_root.empty() || legacy_root.front() != '/' ||
        legacy_root.back() == '/' || v2_root != legacy_root + "/aicpu") return false;
    const auto slash = source.find_last_of('/');
    if (slash == std::string::npos || source.substr(slash + 1) != kProbePackage) return false;
    const std::string basename = std::to_string(pid) + "_" + kProbePackage;
    if (destination != legacy_root + "/" + basename) return false;
    redirected = v2_root + "/" + basename;
    return true;
}
