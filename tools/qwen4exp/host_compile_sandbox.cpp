// SPDX-License-Identifier: Apache-2.0
// Host-only execution containment. No CANN headers or libraries are linked.
#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <string>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <vector>

#ifndef LANDLOCK_ACCESS_FS_REFER
  #define LANDLOCK_ACCESS_FS_REFER (1ULL << 13)
#endif
#ifndef LANDLOCK_ACCESS_FS_TRUNCATE
  #define LANDLOCK_ACCESS_FS_TRUNCATE (1ULL << 14)
#endif
#ifndef LANDLOCK_ACCESS_FS_IOCTL_DEV
  #define LANDLOCK_ACCESS_FS_IOCTL_DEV (1ULL << 15)
#endif

namespace {
constexpr int FAILURE = 125;
constexpr int MINIMUM_ABI = 3;  // Includes path-based truncation confinement.
constexpr uint64_t READ = LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;
constexpr uint64_t WRITE = LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_REMOVE_DIR |
                           LANDLOCK_ACCESS_FS_REMOVE_FILE | LANDLOCK_ACCESS_FS_MAKE_DIR | LANDLOCK_ACCESS_FS_MAKE_REG |
                           LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_SOCK;

int fail(const char* reason) {
  std::fprintf(stderr, "host-compile-sandbox: %s: %s\n", reason, std::strerror(errno));
  return FAILURE;
}

bool safe_device(const std::string& path) {
  return path == "/dev/null" || path == "/dev/zero" || path == "/dev/random" || path == "/dev/urandom";
}

bool allowed_root(const std::string& path) {
  // A root/ancestor grant would silently cover /dev and invalidate containment.
  const char* broad[] = {"/", "/dev", "/srv", "/home", "/root", "/run", "/var"};
  for (const char* entry : broad)
    if (path == entry) return false;
  if (path.compare(0, 5, "/dev/") == 0) return safe_device(path);
  return true;
}

int add_rule(int ruleset, const char* argument, bool writable, uint64_t handled) {
  if (!argument || argument[0] != '/') {
    errno = EINVAL;
    return fail("allow roots must be absolute existing paths");
  }
  char resolved[PATH_MAX];
  if (!realpath(argument, resolved)) return fail("cannot resolve allow root");
  if (!allowed_root(resolved)) {
    errno = EACCES;
    return fail("broad or unsafe device allow root rejected");
  }
  int fd = open(resolved, O_PATH | O_CLOEXEC);
  if (fd < 0) return fail("cannot open rule metadata");
  struct stat state{};
  if (fstat(fd, &state) < 0) {
    close(fd);
    return fail("cannot inspect rule metadata");
  }
  if ((!S_ISDIR(state.st_mode) && !S_ISREG(state.st_mode) && !safe_device(resolved)) || S_ISBLK(state.st_mode)) {
    close(fd);
    errno = EACCES;
    return fail("only ordinary paths and explicit safe devices may be allowed");
  }
  if (safe_device(resolved)) {
    const unsigned int expected_minor = std::strcmp(resolved, "/dev/null") == 0     ? 3
                                        : std::strcmp(resolved, "/dev/zero") == 0   ? 5
                                        : std::strcmp(resolved, "/dev/random") == 0 ? 8
                                                                                    : 9;
    if (!S_ISCHR(state.st_mode) || major(state.st_rdev) != 1 || minor(state.st_rdev) != expected_minor) {
      close(fd);
      errno = EACCES;
      return fail("safe device path has unexpected device identity");
    }
  }
  uint64_t access = READ | (writable ? WRITE | LANDLOCK_ACCESS_FS_REFER | LANDLOCK_ACCESS_FS_TRUNCATE : 0);
  if (!S_ISDIR(state.st_mode)) {
    access &= LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE |
              LANDLOCK_ACCESS_FS_TRUNCATE;
    if (S_ISCHR(state.st_mode)) access &= LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE;
  }
  landlock_path_beneath_attr rule{};
  rule.parent_fd = fd;
  rule.allowed_access = access & handled;
  const long result = syscall(SYS_landlock_add_rule, ruleset, LANDLOCK_RULE_PATH_BENEATH, &rule, 0);
  close(fd);
  return result < 0 ? fail("cannot install filesystem rule") : 0;
}

int secure_standard_fds() {
  for (int fd = 0; fd < 3; ++fd) {
    struct stat state{};
    if (fstat(fd, &state) < 0) {
      if (errno == EBADF) continue;
      return fail("cannot inspect standard descriptor");
    }
    if (S_ISBLK(state.st_mode) || S_ISSOCK(state.st_mode)) {
      errno = EACCES;
      return fail("block device or socket inherited as standard descriptor");
    }
    if (S_ISCHR(state.st_mode)) {
      const auto device_major = major(state.st_rdev);
      const auto device_minor = minor(state.st_rdev);
      const bool memory_safe =
          device_major == 1 && (device_minor == 3 || device_minor == 5 || device_minor == 8 || device_minor == 9);
      const bool terminal = device_major == 4 || device_major == 5 || (device_major >= 136 && device_major <= 143);
      if (!memory_safe && !terminal) {
        errno = EACCES;
        return fail("unsafe character device inherited as standard descriptor");
      }
    }
  }
  return 0;
}

int deny_driver_ipc() {
#if defined(__x86_64__)
  // Filesystem isolation alone does not block a driver daemon or SCM_RIGHTS.
  // Host compilation needs neither network sockets nor device ioctls.
  std::vector<sock_filter> filter = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000U, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
  };
  const int denied[] = {SYS_socket, SYS_socketpair, SYS_connect, SYS_sendmsg, SYS_recvmsg, SYS_ioctl};
  for (int call : denied) {
    filter.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, static_cast<uint32_t>(call), 0, 1));
    filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EACCES));
  }
  filter.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
  sock_fprog program{static_cast<unsigned short>(filter.size()), filter.data()};
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) < 0) return fail("cannot install driver IPC filter");
  return 0;
#else
  errno = ENOTSUP;
  return fail("host architecture has no verified syscall filter");
#endif
}
}  // namespace

int main(int argc, char** argv) {
  const long abi = syscall(SYS_landlock_create_ruleset, nullptr, 0, LANDLOCK_CREATE_RULESET_VERSION);
  if (abi < MINIMUM_ABI) {
    if (abi >= 0) errno = ENOTSUP;
    return fail("required Landlock ABI >=3 unavailable; child not executed");
  }
  if (argc == 2 && std::strcmp(argv[1], "--probe") == 0) {
    std::printf("{\"landlock_abi\":%ld,\"probe\":\"query_only\"}\n", abi);
    return 0;
  }
  uint64_t handled = (1ULL << 13) - 1;
  if (abi >= 2) handled |= LANDLOCK_ACCESS_FS_REFER;
  if (abi >= 3) handled |= LANDLOCK_ACCESS_FS_TRUNCATE;
  if (abi >= 5) handled |= LANDLOCK_ACCESS_FS_IOCTL_DEV;
  landlock_ruleset_attr attributes{};
  attributes.handled_access_fs = handled;
  // Only filesystem fields: works with older installed/kernel ABI struct sizes.
  const int ruleset = syscall(SYS_landlock_create_ruleset, &attributes, sizeof(uint64_t), 0);
  if (ruleset < 0) return fail("cannot create filesystem ruleset");
  int index = 1;
  bool any_rule = false;
  for (; index < argc && std::strcmp(argv[index], "--") != 0; index += 2) {
    if (index + 1 >= argc ||
        (std::strcmp(argv[index], "--allow-read") != 0 && std::strcmp(argv[index], "--allow-write") != 0)) {
      errno = EINVAL;
      return fail("usage: --allow-read PATH / --allow-write PATH ... -- COMMAND ARGS");
    }
    const int result = add_rule(ruleset, argv[index + 1], std::strcmp(argv[index], "--allow-write") == 0, handled);
    if (result) return result;
    any_rule = true;
  }
  if (!any_rule || index + 1 >= argc || std::strcmp(argv[index], "--") != 0) {
    errno = EINVAL;
    return fail("explicit allow roots and child command required");
  }
  if (secure_standard_fds()) return FAILURE;
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) return fail("cannot establish no_new_privs");
  if (syscall(SYS_landlock_restrict_self, ruleset, 0) < 0)
    return fail("filesystem restriction failed; child not executed");
  close(ruleset);
  if (syscall(SYS_close_range, 3U, UINT_MAX, 0) < 0) return fail("cannot close inherited descriptors");
  if (deny_driver_ipc()) return FAILURE;
  // The helper must itself be built statically and started with a clean loader
  // environment. After containment, discard preload/audit instructions before
  // dynamic child startup and disable automatic Torch backend discovery.
  unsetenv("LD_PRELOAD");
  unsetenv("LD_AUDIT");
  if (setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1) < 0) return fail("cannot set host-only child environment");
  std::fprintf(stderr,
               "{\"containment\":\"landlock_seccomp\",\"landlock_abi\":%ld,\"restricted\":true,"
               "\"nonstandard_fds_closed\":true,\"driver_ioctls_and_sockets_denied\":true}\n",
               abi);
  execvp(argv[index + 1], &argv[index + 1]);
  return fail("sandboxed child exec failed");
}
