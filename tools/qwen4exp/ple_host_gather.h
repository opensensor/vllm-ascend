// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ple_host {
constexpr size_t kHeads = 16;
constexpr size_t kRowBytes = 320;
constexpr size_t kReceiptWords = 8;
constexpr size_t kMaxTokens = 18;
constexpr int kDeviceCount = 6;
constexpr size_t kPacketBytesPerToken = kHeads * sizeof(int64_t) + kReceiptWords * sizeof(int32_t);

// Packet layout: all token/head INT64 rows, then all token INT32 receipts.
// Validate the complete request before reading any source row or writing output.
inline int validate(const void* packet, size_t tokens, size_t table_bytes) {
  if (!packet || !tokens || tokens > kMaxTokens || !table_bytes || table_bytes % kRowBytes) return -EINVAL;
  const auto* rows = static_cast<const int64_t*>(packet);
  const auto* receipts = reinterpret_cast<const int32_t*>(rows + tokens * kHeads);
  for (size_t token = 0; token < tokens; ++token)
    if (receipts[token * kReceiptWords] != 0) return -EINVAL;
  for (size_t row = 0; row < tokens * kHeads; ++row)
    if (rows[row] < 0 || static_cast<uint64_t>(rows[row]) >= table_bytes / kRowBytes) return -ERANGE;
  return 0;
}

inline int gather(const void* table, size_t table_bytes, const void* packet, size_t tokens, void* output) {
  if (!table || !output) return -EINVAL;
  const int status = validate(packet, tokens, table_bytes);
  if (status) return status;
  const auto* rows = static_cast<const int64_t*>(packet);
  for (size_t row = 0; row < tokens * kHeads; ++row)
    std::memcpy(static_cast<char*>(output) + row * kRowBytes,
                static_cast<const char*>(table) + static_cast<size_t>(rows[row]) * kRowBytes, kRowBytes);
  return 0;
}
}  // namespace ple_host
