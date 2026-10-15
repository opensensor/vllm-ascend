// SPDX-License-Identifier: Apache-2.0
#ifndef QWEN_PLE_HOST_MAPPING_H
#define QWEN_PLE_HOST_MAPPING_H

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cerrno>
#include <memory>

namespace ple_host {
// One populated mapping per serving manager. Lookup owners share its lifetime;
// closing the first lookup must not invalidate later lookups or their gathers.
class SharedTable {
 public:
  void* data = MAP_FAILED;
  size_t bytes = 0;

  SharedTable() = default;
  SharedTable(const SharedTable&) = delete;
  SharedTable& operator=(const SharedTable&) = delete;
  ~SharedTable() {
    if (data != MAP_FAILED) munmap(data, bytes);
  }

  static int Open(int fd, size_t row_bytes, std::shared_ptr<SharedTable>& result) {
    struct stat info{};
    if (fstat(fd, &info)) return -errno;
    constexpr int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    const int actual_seals = fcntl(fd, F_GET_SEALS);
    if (!row_bytes || actual_seals < 0 || (actual_seals & seals) != seals || info.st_size <= 0 ||
        info.st_size % row_bytes)
      return -EINVAL;
    auto table = std::make_shared<SharedTable>();
    table->bytes = static_cast<size_t>(info.st_size);
    table->data = mmap(nullptr, table->bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (table->data == MAP_FAILED) return -errno;
    // Populate only once, at admission. Subsequent batch shapes reuse both
    // the physical pages and this process's page-table entries.
    if (madvise(table->data, table->bytes, MADV_POPULATE_READ)) return -errno;
    result = std::move(table);
    return 0;
  }
};
}  // namespace ple_host
#endif
