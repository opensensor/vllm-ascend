// SPDX-License-Identifier: Apache-2.0
#include "mamba_commit_integer.h"
#include <cassert>
int main() {
  for (int source = 0; source < 5; ++source)
    for (int computed = 0; computed < 262145; ++computed)
      for (int accepted = 0; accepted < 4; ++accepted) {
        auto p = mamba_commit::Plan(source, computed, 3, 2, accepted, 128);
        long running = computed + 1, aligned = (running + accepted - 1) / 128 * 128;
        long dst = aligned / 128 - 1, bias = aligned - running;
        assert(p.copy == (aligned >= running && (source != dst || bias != 0)));
        assert(p.next_count == (aligned >= running && source == dst ? 1 : accepted));
        if (p.copy) {
          assert(p.destination == dst);
          assert(p.bias == bias);
        }
      }
}
