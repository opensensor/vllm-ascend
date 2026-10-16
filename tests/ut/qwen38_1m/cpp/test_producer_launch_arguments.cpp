// SPDX-License-Identifier: Apache-2.0
#include "tools/qwen4exp/producer_launch_arguments.h"
#include <cassert>
#include <functional>
#include <memory>

struct Operand {
  std::thread::id* destroyed;
  ~Operand() { *destroyed = std::this_thread::get_id(); }
};

int main() {
  using Owner = qwen_native::ProducerLaunchArguments<std::shared_ptr<Operand>>;
  const auto producer = std::this_thread::get_id();
  for (int mode = 0; mode < 3; ++mode) {
    std::thread::id destroyed;
    auto operand = std::shared_ptr<Operand>(new Operand{&destroyed});
    auto owner = std::make_shared<Owner>(std::vector{operand});
    operand.reset();
    try {
      owner->Release();
      assert(false);
    } catch (const std::logic_error&) {
    }
    std::function<int()> callback = [owner, mode] {
      return owner->Run([mode](const auto& values) {
        assert(values.size() == 1);
        if (mode == 2) throw std::runtime_error("submission failed");
        return mode;
      });
    };
    auto retained_copy = callback;
    std::thread consumer([&] {
      try {
        assert(callback() == mode);
      } catch (const std::runtime_error&) {
        assert(mode == 2);
      }
      try {
        owner->Release();
        assert(false);
      } catch (const std::logic_error&) {
      }
      callback = {};
    });
    consumer.join();  // Models producer host-queue drain, not device completion.
    assert(destroyed == std::thread::id{});
    try {
      retained_copy();
      assert(false);
    } catch (const std::logic_error&) {
    }
    if (mode == 0) {
      owner->Release();
      assert(destroyed == producer);
    } else {
      try {
        owner->Release();
        assert(false);
      } catch (const std::logic_error&) {
      }
      assert(destroyed == std::thread::id{});
    }
    retained_copy = {};
    owner.reset();  // Failed owners are quarantined until explicit retirement.
    assert(destroyed == producer);
  }
}
