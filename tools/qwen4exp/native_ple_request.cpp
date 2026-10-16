// SPDX-License-Identifier: Apache-2.0
// Exact device hashing plus backend-neutral row IDs and compact RAM offsets.
#include "kernel_operator.h"
#define PLE_INLINE __aicore__ inline
#include "ple_request_integer.h"

using namespace AscendC;
namespace {
constexpr int64_t kMaxRequests = 6;
constexpr int64_t kMaxTokens = 2048;
constexpr int32_t kReceiptWords = 8;
constexpr int32_t kMultiplierStart = 4;
constexpr int32_t kSizeStart = 7;
constexpr int32_t kOffsetStart = 23;
}  // namespace

extern "C" __global__ __aicore__ void qwen_ple_request_v1(GM_ADDR tokens_ptr, GM_ADDR starts_ptr, GM_ADDR history_ptr,
                                                          GM_ADDR config_ptr, GM_ADDR rows_ptr, GM_ADDR offsets_ptr,
                                                          GM_ADDR receipt_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  GlobalTensor<int64_t> config, tokens, history, rows, offsets;
  GlobalTensor<int32_t> starts, receipts;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(config_ptr));
  const int64_t count = config.GetValue(0), requests = config.GetValue(1);
  // Immutable geometry is validated before graph capture by the owner.
  if (count < 1 || count > kMaxTokens || requests < 1 || requests > kMaxRequests) return;
  const int64_t eos = config.GetValue(2), vocabulary = config.GetValue(3);
  tokens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tokens_ptr));
  starts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(starts_ptr));
  history.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(history_ptr));
  rows.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(rows_ptr));
  offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(offsets_ptr));
  receipts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(receipt_ptr));
  int32_t layout_error = starts.GetValue(0) != 0 || starts.GetValue(requests) != count;
  for (int64_t r = 0; r < requests; ++r) {
    const int32_t first = starts.GetValue(r), last = starts.GetValue(r + 1);
    if (first < 0 || last < first || last > count) layout_error = 1;
  }
  TPipe pipe;
  TBuf<TPosition::VECCALC> row_buffer, offset_buffer, receipt_buffer;
  pipe.InitBuffer(row_buffer, ple_request::kHeads * sizeof(int64_t));
  pipe.InitBuffer(offset_buffer, ple_request::kHeads * sizeof(int64_t));
  pipe.InitBuffer(receipt_buffer, kReceiptWords * sizeof(int32_t));
  auto local_rows = row_buffer.Get<int64_t>();
  auto local_offsets = offset_buffer.Get<int64_t>();
  auto local_receipt = receipt_buffer.Get<int32_t>();
  for (int64_t position = GetBlockIdx(); position < count; position += GetBlockNum()) {
    int32_t error = layout_error;
    int64_t current = eos, recent = eos, older = eos;
    if (!error) {
      int64_t request = 0;
      while (request + 1 < requests && position >= starts.GetValue(request + 1)) ++request;
      const int64_t local = position - starts.GetValue(request);
      current = tokens.GetValue(position);
      recent = local >= 1 ? tokens.GetValue(position - 1) : history.GetValue(request * 2 + 1);
      older = local >= 2 ? tokens.GetValue(position - 2) : history.GetValue(request * 2 + local);
      // An older token across EOS is deliberately ignored by model math.
      if (recent == eos) older = eos;
      if (!ple_request::ValidToken(current, vocabulary) || !ple_request::ValidToken(recent, vocabulary) ||
          !ple_request::ValidToken(older, vocabulary))
        error = 2;
    }
    for (int32_t head = 0; head < ple_request::kHeads; ++head) {
      int64_t row = -1;
      if (!error) {
        const auto mixed =
            ple_request::Mix(current, recent, older, eos, config.GetValue(kMultiplierStart),
                             config.GetValue(kMultiplierStart + 1), config.GetValue(kMultiplierStart + 2), head);
        row = ple_request::Row(mixed, config.GetValue(kSizeStart + head), config.GetValue(kOffsetStart + head));
      }
      local_rows.SetValue(head, row);
      local_offsets.SetValue(head, row < 0 ? -1 : row * ple_request::kRowBytes);
    }
    for (int32_t word = 0; word < kReceiptWords; ++word) local_receipt.SetValue(word, word == 0 ? error : 0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(rows[position * ple_request::kHeads], local_rows, ple_request::kHeads);
    DataCopy(offsets[position * ple_request::kHeads], local_offsets, ple_request::kHeads);
    DataCopy(receipts[position * kReceiptWords], local_receipt, kReceiptWords);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
}
