// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace rtsp_scene {
// Select the first real decoded frame at/after each sampling boundary. Missing
// boundaries stay missing; never manufacture catch-up samples or timestamps.
class SampleClock {
 public:
  explicit SampleClock(double fps) : fps_(fps) {
    if (!std::isfinite(fps) || fps <= 0) throw std::invalid_argument("Invalid sample rate");
  }
  bool select(double seconds) {
    if (!std::isfinite(seconds) || seconds < previous_) throw std::runtime_error("Decoded PTS moved backwards");
    previous_ = seconds;
    const auto slot = static_cast<int64_t>(std::floor(seconds * fps_ + 1e-6));
    if (slot <= last_slot_) return false;
    missed_ += slot - last_slot_ - 1;
    last_slot_ = slot;
    return true;
  }
  int64_t missed() const { return missed_; }

 private:
  double fps_, previous_ = 0;
  int64_t last_slot_ = -1, missed_ = 0;
};
}  // namespace rtsp_scene
