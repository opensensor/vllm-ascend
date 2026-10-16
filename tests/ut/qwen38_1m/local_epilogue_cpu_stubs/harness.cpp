// SPDX-License-Identifier: Apache-2.0
#include "native_local_route_epilogue.cpp"

extern "C" int run_local_epilogue(int mode, int64_t rows, const half* input, const int64_t* inverse,
                                  const float* weights, const int64_t* ends, float* output, uint64_t* readBytes,
                                  uint64_t* localBytes) {
  using namespace AscendC;
  const auto registerRegion = [](const void* p, size_t bytes, size_t readable) {
    regions.push_back({reinterpret_cast<uintptr_t>(p), bytes, readable});
  };
  regions.clear();
  const int64_t active = std::max<int64_t>(0, std::min<int64_t>(rows, ends[127]));
  registerRegion(input, rows * 2560 * sizeof(half), active * 2560 * sizeof(half));
  registerRegion(inverse, rows * sizeof(int64_t), rows * sizeof(int64_t));
  registerRegion(weights, rows * sizeof(float), rows * sizeof(float));
  registerRegion(ends, 128 * sizeof(int64_t), 128 * sizeof(int64_t));
  outputBegin = reinterpret_cast<uintptr_t>(output);
  outputWrites.assign((mode == 0 ? rows : rows / 10) * 2560, 0);
  inputReadBytes = 0;
  *localBytes = 0;
  try {
    for (blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
      auto kernel = mode == 0   ? qwen_local_route_finalize_v1
                    : mode == 1 ? qwen_local_route_reduce_sequential_v1
                                : qwen_local_route_reduce_pairwise_v1;
      kernel(const_cast<half*>(input), const_cast<int64_t*>(inverse), const_cast<float*>(weights),
             const_cast<int64_t*>(ends), output, &rows);
      *localBytes = std::max<uint64_t>(*localBytes, allocatedUB);
      for (bool event : events) assert(!event);
    }
    for (unsigned count : outputWrites) assert(count == 1);
    *readBytes = inputReadBytes;
    return 0;
  } catch (const std::runtime_error&) {
    return 1;
  }
}
