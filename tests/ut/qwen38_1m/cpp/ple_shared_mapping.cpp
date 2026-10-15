// SPDX-License-Identifier: Apache-2.0
#include <sys/mman.h>
#include <unistd.h>
#include <cassert>
#include <vector>
#include "ple_host_mapping.h"

int main() {
  int fd = memfd_create("ple-mapping-test", MFD_ALLOW_SEALING);
  assert(fd >= 0);
  std::vector<char> bytes(320 * 128, 'q');
  assert(write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
  std::shared_ptr<ple_host::SharedTable> parent;
  assert(ple_host::SharedTable::Open(fd, 320, parent) == -EINVAL);
  assert(!parent);
  assert(fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0);
  assert(ple_host::SharedTable::Open(fd, 321, parent) == -EINVAL);
  assert(ple_host::SharedTable::Open(fd, 320, parent) == 0);
  void* address = parent->data;
  assert(parent->bytes == bytes.size());
  auto child = parent;
  auto other = parent;
  assert(parent.use_count() == 3);
  close(fd);
  parent.reset();
  other.reset();
  // The populated mapping remains readable even when the fd and first owner
  // are gone. Destroying the last child unmaps exactly this shared address.
  assert(child->data == address && static_cast<char*>(address)[bytes.size() - 1] == 'q');
  child.reset();
  unsigned char resident = 0;
  errno = 0;
  assert(mincore(address, 4096, &resident) == -1 && errno == ENOMEM);
  assert(ple_host::SharedTable::Open(-1, 320, parent) == -EBADF);
  assert(!parent);
}
