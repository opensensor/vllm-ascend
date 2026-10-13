// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cstdint>

constexpr unsigned int THREADS = 256;
constexpr uint32_t WIDTH = 4096;
constexpr uint32_t INPUT_ROW_BYTES = 8256;
constexpr uint32_t OUTPUT_ROW_BYTES = WIDTH * sizeof(float);
constexpr uint32_t MAX_PACKED_ROWS = 64;
constexpr uint32_t MAX_SEPARATE_ROWS = 128;
constexpr uint32_t FIRST_COLD_LAYER = 18;
constexpr uint32_t LAST_COLD_LAYER = 32;
constexpr uint32_t RANKS = 2;

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
static_assert(MAX_SEPARATE_ROWS * OUTPUT_ROW_BYTES == 2 * 1024 * 1024);

extern "C" uint32_t sidecard_glm_completion_max_rows(uint32_t separate) {
  return separate ? MAX_SEPARATE_ROWS : MAX_PACKED_ROWS;
}

__global__ void merge_rank(const float* input, float* output, const uint64_t* packet, uint64_t epoch, uint64_t sequence,
                           uint32_t layer, uint32_t rows, uint32_t rank, const float* separate) {
  const auto& record = *reinterpret_cast<const Invocation*>(packet + 1);
  const bool valid = record.magic[0] == 'S' && record.magic[1] == 'C' && record.magic[2] == 'M' &&
                     record.magic[3] == 'A' && record.magic[4] == 'I' && record.magic[5] == 'L' &&
                     record.magic[6] == '0' && record.magic[7] == '1' && record.epoch == epoch &&
                     record.sequence == sequence && record.layer == layer && record.rows == rows &&
                     record.slot == rank && record.rank == rank && record.input_bytes == rows * INPUT_ROW_BYTES &&
                     record.output_bytes == rows * OUTPUT_ROW_BYTES && !record.reserved;
  const auto* partial = separate ? separate
                                 : reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(packet) +
                                                                  sizeof(uint64_t) + sizeof(Invocation));
  const uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < rows * WIDTH)
    output[offset] = valid ? __fadd_rn(input[offset], partial[offset]) : __int_as_float(0x7fffffff);
}

__global__ void prepare_control(uint64_t* control, uint8_t* input, uint64_t epoch, uint64_t sequence, uint32_t layer,
                                uint32_t rows, uint32_t rank) {
  Invocation record = {{'S', 'C', 'M', 'A', 'I', 'L', '0', '1'},
                       epoch,
                       sequence,
                       layer,
                       rows,
                       rank,
                       rank,
                       rows * INPUT_ROW_BYTES,
                       rows * OUTPUT_ROW_BYTES,
                       0};
  *reinterpret_cast<Invocation*>(control + 1) = record;
  *reinterpret_cast<Invocation*>(input + rows * INPUT_ROW_BYTES) = record;
  *control = 0;
}

extern "C" int sidecard_glm_completion_preload() {
  cudaFuncAttributes attributes;
  const int status = cudaFuncGetAttributes(&attributes, merge_rank);
  return status ? status : cudaFuncGetAttributes(&attributes, prepare_control);
}

extern "C" int sidecard_glm_merge_rank(const float* input, float* output, const uint64_t* packet, uint64_t epoch,
                                       uint64_t sequence, uint32_t layer, uint32_t rows, uint32_t rank,
                                       cudaStream_t stream) {
  if (!input || !output || !packet || !epoch || !sequence || layer < FIRST_COLD_LAYER || layer > LAST_COLD_LAYER ||
      !rows || rows > MAX_PACKED_ROWS || rank >= RANKS)
    return cudaErrorInvalidValue;
  merge_rank<<<(rows * WIDTH + THREADS - 1) / THREADS, THREADS, 0, stream>>>(input, output, packet, epoch, sequence,
                                                                             layer, rows, rank, nullptr);
  return cudaGetLastError();
}

extern "C" int sidecard_glm_merge_separate(const float* input, float* output, const uint64_t* control,
                                           const float* partial, uint64_t epoch, uint64_t sequence, uint32_t layer,
                                           uint32_t rows, uint32_t rank, cudaStream_t stream) {
  if (!input || !output || !control || !partial || !epoch || !sequence || layer < FIRST_COLD_LAYER ||
      layer > LAST_COLD_LAYER || !rows || rows > MAX_SEPARATE_ROWS || rank >= RANKS)
    return cudaErrorInvalidValue;
  merge_rank<<<(rows * WIDTH + THREADS - 1) / THREADS, THREADS, 0, stream>>>(input, output, control, epoch, sequence,
                                                                             layer, rows, rank, partial);
  return cudaGetLastError();
}

extern "C" int sidecard_glm_prepare_control(uint64_t* control, uint8_t* input, uint64_t epoch, uint64_t sequence,
                                            uint32_t layer, uint32_t rows, uint32_t rank, cudaStream_t stream) {
  if (!control || !input || !epoch || !sequence || layer < FIRST_COLD_LAYER || layer > LAST_COLD_LAYER || !rows ||
      rows > MAX_SEPARATE_ROWS || rank >= RANKS)
    return cudaErrorInvalidValue;
  prepare_control<<<1, 1, 0, stream>>>(control, input, epoch, sequence, layer, rows, rank);
  return cudaGetLastError();
}
