// SPDX-License-Identifier: Apache-2.0
#ifndef QWEN_PRODUCER_LAUNCH_ARGUMENTS_H
#define QWEN_PRODUCER_LAUNCH_ARGUMENTS_H

#include <atomic>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace qwen_native {
// A queued callback must never destroy a Tensor: allocator event recording can
// wait for that very queue while another thread owns the allocator mutex.
// Keep the owner on the submitting thread until its host queue has drained.
template <typename Argument>
class ProducerLaunchArguments {
 public:
  explicit ProducerLaunchArguments(std::vector<Argument> arguments)
      : arguments_(std::move(arguments)), producer_(std::this_thread::get_id()) {}

  template <typename Submit>
  int Run(Submit&& submit) {
    if (attempted_.exchange(true, std::memory_order_acq_rel)) {
      throw std::logic_error("native submission callback invoked more than once");
    }
    const int status = std::forward<Submit>(submit)(arguments_);
    succeeded_.store(status == 0, std::memory_order_release);
    return status;
  }

  void Release() {
    if (std::this_thread::get_id() != producer_ || !succeeded_.load(std::memory_order_acquire)) {
      throw std::logic_error("native arguments require successful submission and producer release");
    }
    arguments_.clear();
  }

 private:
  std::vector<Argument> arguments_;
  const std::thread::id producer_;
  std::atomic<bool> attempted_{false};
  std::atomic<bool> succeeded_{false};
};
}  // namespace qwen_native
#endif
