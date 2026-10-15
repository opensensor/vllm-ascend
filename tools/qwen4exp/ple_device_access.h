// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/xattr.h>
#include <unistd.h>

#include "ple_probe_text.h"

// Shared by the custom AI CPU kernel and a standalone ARM subprocess. No HAL,
// ioctl, mmap, privilege changes, or DMA submissions belong in this diagnostic.
inline void ple_access_file(PleProbeText& out, const char* path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    out.append("FILE %s open_errno=%d\n", path, fd < 0 ? errno : 0);
    if (fd < 0) return;
    constexpr size_t kFileLimit = 4096;
    char data[kFileLimit + 1];
    const ssize_t count = read(fd, data, kFileLimit);
    const int error = count < 0 ? errno : 0;
    close(fd);
    if (count >= 0) {
        data[count] = '\0';
        out.append("%s\n", data);
    }
    out.append("END read_errno=%d file_limit_reached=%d\n", error,
               count == static_cast<ssize_t>(kFileLimit));
}

inline void ple_access_node(PleProbeText& out, const char* path) {
    struct stat info{};
    const int status = stat(path, &info);
    out.append("NODE %s stat_errno=%d", path, status < 0 ? errno : 0);
    if (status == 0) {
        out.append(" uid=%lu gid=%lu mode=%04o character_device=%d",
                   static_cast<unsigned long>(info.st_uid),
                   static_cast<unsigned long>(info.st_gid),
                   static_cast<unsigned>(info.st_mode & 07777), S_ISCHR(info.st_mode));
        char label[256];
        const ssize_t size = getxattr(path, "security.selinux", label, sizeof(label) - 1);
        const int error = size < 0 ? errno : 0;
        if (size >= 0) label[size] = '\0';
        out.append(" label_errno=%d label=%s", error, size >= 0 ? label : "unavailable");
    }
    // Record actual open results; stat bits alone cannot diagnose MAC or driver
    // checks. Never read a device node, and never try an O_RDWR fallback.
    const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    const int error = fd < 0 ? errno : 0;
    if (fd >= 0) close(fd);
    out.append(" open_readonly_errno=%d\n", error);
}

inline void ple_device_access_report(PleProbeText& out) {
    out.append("PLE_DEVICE_ACCESS version=1 pid=%ld uid=%lu euid=%lu gid=%lu egid=%lu\n",
               static_cast<long>(getpid()), static_cast<unsigned long>(getuid()),
               static_cast<unsigned long>(geteuid()), static_cast<unsigned long>(getgid()),
               static_cast<unsigned long>(getegid()));
    struct utsname platform{};
    if (uname(&platform) == 0) {
        out.append("PLATFORM machine=%s release=%s\n", platform.machine, platform.release);
    }
    const char* files[] = {"/proc/self/status", "/proc/self/attr/current",
                           "/proc/self/uid_map", "/proc/self/gid_map",
                           "/sys/fs/selinux/enforce", "/etc/hdcFileTransWhiteList.cfg"};
    for (const char* path : files) ple_access_file(out, path);
    const char* nodes[] = {"/dev/hisi_hdc", "/dev/svm0", "/dev/davinci_manager",
                           "/dev/sdma", "/dev/ascend_device_dma"};
    for (const char* path : nodes) ple_access_node(out, path);
    const char* runtime_paths[] = {"/home/HwHiAiUser", "/home/HwHiAiUser/hs",
                                  "/home/HwHiAiUser/hs/0",
                                  "/home/HwHiAiUser/hs/0/omfile",
                                  "/home/HwHiAiUser/hs/0/runtime",
                                  "/home/HwHiAiUser/hs/0/runtime/bin/udf_executor",
                                  "/var", "/var/tsdaemon_check_illegal_link.sh",
                                  "/bin/sh", "/bin/bash"};
    for (const char* path : runtime_paths) ple_access_node(out, path);
    out.append("DMA_SUBMITTED=0\n");
}
