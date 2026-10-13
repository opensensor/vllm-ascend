// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

constexpr unsigned int THREADS = 256;
constexpr uint8_t PAYLOAD_XOR = 0xA7;
constexpr uint8_t REJECTED_RECORD = 0xEE;
constexpr uint32_t PROBE_LAYER = 18;
constexpr uint32_t MAX_ROWS = 128;
constexpr uint64_t INPUT_ROW_BYTES = 8256;
constexpr uint64_t OUTPUT_ROW_BYTES = 16384;

struct Invocation {
  char magic[8];
  uint64_t epoch;
  uint64_t sequence;
  uint32_t layer;
  uint32_t rows;
  uint32_t slot;
  uint32_t rank;
  uint64_t input_bytes;
  uint64_t output_bytes;
  uint64_t reserved;
};
static_assert(sizeof(Invocation) == 64);

__device__ bool matches(const Invocation& record, uint64_t epoch, uint64_t sequence, uint32_t slot, uint32_t rank,
                        size_t size) {
  return record.magic[0] == 'S' && record.magic[1] == 'C' && record.magic[2] == 'M' && record.magic[3] == 'A' &&
         record.magic[4] == 'I' && record.magic[5] == 'L' && record.magic[6] == '0' && record.magic[7] == '1' &&
         record.epoch == epoch && record.sequence == sequence && record.slot == slot && record.rank == rank &&
         record.layer == PROBE_LAYER && record.rows >= 1 && record.rows <= MAX_ROWS &&
         record.input_bytes == record.rows * INPUT_ROW_BYTES && record.output_bytes == record.rows * OUTPUT_ROW_BYTES &&
         (size == record.input_bytes || size == record.output_bytes) && !record.reserved;
}

__global__ void consume(const uint8_t* source, uint8_t* destination, size_t size, const uint64_t* completion,
                        uint64_t epoch, uint64_t sequence, uint32_t slot, uint32_t rank) {
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const auto& record = *reinterpret_cast<const Invocation*>(completion + 1);
  if (offset < size) {
    destination[offset] =
        matches(record, epoch, sequence, slot, rank, size) ? source[offset] ^ PAYLOAD_XOR : REJECTED_RECORD;
  }
}

extern "C" int sidecard_mailbox_preload() {
  cudaFuncAttributes attributes;
  return cudaFuncGetAttributes(&attributes, consume);
}

extern "C" int sidecard_mailbox_consume(const uint8_t* source, uint8_t* destination, size_t size,
                                        const uint64_t* completion, uint64_t epoch, uint64_t sequence, uint32_t slot,
                                        uint32_t rank, cudaStream_t stream) {
  if (!source || !destination || !size || !completion || !epoch || !sequence || slot >= 2 || rank >= 2)
    return cudaErrorInvalidValue;
  consume<<<(size + THREADS - 1) / THREADS, THREADS, 0, stream>>>(source, destination, size, completion, epoch,
                                                                  sequence, slot, rank);
  return cudaGetLastError();
}
