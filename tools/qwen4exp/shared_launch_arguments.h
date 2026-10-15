// SPDX-License-Identifier: Apache-2.0
#ifndef QWEN_SHARED_LAUNCH_ARGUMENTS_H
#define QWEN_SHARED_LAUNCH_ARGUMENTS_H

#include <atomic>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen_native {
// All std::function copies share this owner, not independent tensor vectors.
// Exactly one host submission is allowed. A device graph replay runs recorded
// native work and must not re-enter this one-shot host callback.
template <typename Argument>
class SharedLaunchArguments {
 public:
  explicit SharedLaunchArguments(std::vector<Argument> arguments) : arguments_(std::move(arguments)) {}

  template <typename Submit>
  int Run(Submit&& submit) {
    if (attempted_.exchange(true, std::memory_order_acq_rel)) {
      throw std::logic_error("native submission callback invoked more than once");
    }
    // A failed or throwing submission preserves every argument. Release only
    // after successful host submission, at the same point as the old bridge.
    const int status = std::forward<Submit>(submit)(arguments_);
    if (status == 0) arguments_.clear();
    return status;
  }

 private:
  std::vector<Argument> arguments_;
  std::atomic<bool> attempted_{false};
};
}  // namespace qwen_native
#endif
