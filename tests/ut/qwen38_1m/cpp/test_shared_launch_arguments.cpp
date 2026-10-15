// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "tools/qwen4exp/shared_launch_arguments.h"

using Payload = std::shared_ptr<int>;
using Owner = qwen_native::SharedLaunchArguments<Payload>;

void IndependentCopiesRetainAfterSubmission() {
  auto tensor = std::make_shared<int>(11);
  std::weak_ptr<int> weak = tensor;
  std::function<int()> original = [arguments = std::vector<Payload>{tensor}]() mutable {
    arguments.clear();
    return 0;
  };
  auto exported = original;
  auto queued = exported;
  tensor.reset();
  assert(queued() == 0);
  assert(!weak.expired());
  original = nullptr;
  assert(!weak.expired());
  exported = nullptr;
  assert(weak.expired());
}

void SharedCopiesReleaseTogetherAndNeverResubmit() {
  auto tensor = std::make_shared<int>(17);
  std::weak_ptr<int> weak = tensor;
  auto owner = std::make_shared<Owner>(std::vector<Payload>{tensor});
  int submissions = 0;
  std::function<int()> original = [owner, &submissions] {
    return owner->Run([&](const auto& arguments) {
      assert(arguments.size() == 1 && *arguments[0] == 17);
      ++submissions;
      return 0;
    });
  };
  auto exported = original;
  auto queued = exported;
  tensor.reset();
  owner.reset();
  assert(queued() == 0);
  assert(weak.expired());
  bool rejected = false;
  try {
    original();
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected && submissions == 1);
}

void FailedSubmissionRetainsAfterEveryHandlerIsDestroyed(bool throwing) {
  auto tensor = std::make_shared<int>(23);
  std::weak_ptr<int> weak = tensor;
  auto owner = std::make_shared<Owner>(std::vector<Payload>{tensor});
  std::vector<std::shared_ptr<Owner>> kernelFailures;
  std::function<int()> original = [owner, &kernelFailures, throwing] {
    int status;
    try {
      status = owner->Run([throwing](const auto& arguments) {
        assert(arguments.size() == 1 && *arguments[0] == 23);
        if (throwing) throw std::runtime_error("submission threw");
        return 9;
      });
    } catch (...) {
      kernelFailures.push_back(owner);
      throw;
    }
    if (status) kernelFailures.push_back(owner);
    return status;
  };
  auto exported = original;
  auto queued = exported;
  tensor.reset();
  owner.reset();
  if (throwing) {
    bool rejected = false;
    try {
      queued();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  } else {
    assert(queued() == 9);
  }
  original = nullptr;
  exported = nullptr;
  queued = nullptr;
  assert(!weak.expired() && kernelFailures.size() == 1);
  // Only the fake kernel's test teardown releases failed argument ownership.
  kernelFailures.clear();
  assert(weak.expired());
}

void ConcurrentDuplicateCannotRaceArgumentRelease() {
  auto owner = std::make_shared<Owner>(std::vector<Payload>{std::make_shared<int>(29)});
  std::atomic<int> submissions{0}, rejections{0};
  auto handler = [owner, &submissions, &rejections] {
    try {
      owner->Run([&](const auto& arguments) {
        assert(arguments.size() == 1 && *arguments[0] == 29);
        ++submissions;
        std::this_thread::yield();
        return 0;
      });
    } catch (const std::logic_error&) {
      ++rejections;
    }
  };
  std::thread first(handler), second(handler);
  first.join();
  second.join();
  assert(submissions == 1 && rejections == 1);
}

int main() {
  IndependentCopiesRetainAfterSubmission();
  SharedCopiesReleaseTogetherAndNeverResubmit();
  FailedSubmissionRetainsAfterEveryHandlerIsDestroyed(false);
  FailedSubmissionRetainsAfterEveryHandlerIsDestroyed(true);
  ConcurrentDuplicateCannotRaceArgumentRelease();
}
