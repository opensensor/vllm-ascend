// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <climits>
#include <initializer_list>
#include "checkpoint_preparation_integer.h"

void Check(int32_t previous, int32_t computed, int32_t accepted) {
  const auto result = checkpoint_preparation::Plan(previous, computed, accepted, 128);
  if (previous < -1 || computed < 0 || accepted < 1 || accepted > 3) {
    assert(result.error == 1 && !result.copy);
    return;
  }
  const int64_t current = (static_cast<int64_t>(computed) + 3 + 127) / 128 - 1;
  const bool copy = previous != -1 && previous != current;
  assert(!result.error && result.copy == copy);
  assert(result.destination == current && result.bias == accepted - 1);
  assert(result.normalized == (copy ? 1 : accepted));
}

int main() {
  for (int32_t computed = -1; computed <= 262144; ++computed) {
    for (int32_t accepted = -1; accepted <= 4; ++accepted) {
      for (int32_t previous : {-2, -1, 0, (computed + 2) / 128, INT_MAX})
        Check(previous, computed, accepted);
    }
  }
  for (int32_t computed : {INT_MAX - 2, INT_MAX - 1, INT_MAX})
    for (int32_t accepted = 1; accepted <= 3; ++accepted)
      Check(INT_MAX / 128, computed, accepted);
  assert(checkpoint_preparation::Plan(0, 0, 1, 0).error == 1);
}
