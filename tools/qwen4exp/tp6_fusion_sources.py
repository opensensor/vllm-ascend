# SPDX-License-Identifier: Apache-2.0
"""Generate private kernel variants from pinned arithmetic; no default changes."""

import hashlib
from pathlib import Path

HEADER_ROOT = Path("csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel")
PACK_SOURCE = Path("csrc/gmm/qwen_w4_a8_swiglu_pack_v310/op_kernel/qwen_w4_a8_swiglu_pack_v310.cpp")
HASHES = {
    "native_int4_schedule.h": "f3e0b9c6c1a9bd96c14069a277857300090bd047e8df74e026bce5f4026b9673",
    "native_int4_route_groups.h": "1281fbe9a68f200931fdedfcc87bd24e3cc01cfc1bcc062116b5da431c535216",
    "qwen_w4_a8_int4_matmul_v310_tiling_data.h": "75355de444258e0edbc53dc917d3338d646a827ec8aba441782654e383686e53",
    "pack": "24bb801797a290fb960a67eb866c33f2a3218255c185bdac1304930127055397",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("pinned fusion seam changed")
    return text.replace(old, new)


def private_schedule(source):
    if digest(source) != HASHES["native_int4_schedule.h"]:
        raise ValueError("unqualified INT4 schedule")
    text = replace_once(source.decode(), " private:", " protected:")
    # Project keeps the qualified FP16 output in UB. Only this private subclass
    # accesses it; no global tensor or installed operator uses this header.
    text = replace_once(
        text,
        "      Store(out, tile, row, count);",
        "      // Paired gate/up consumes output_ in UB; no GM intermediate.",
    )
    return text


def local_pack(source):
    if digest(source) != HASHES["pack"]:
        raise ValueError("unqualified SwiGLU packing source")
    text = source.decode()
    buffers = text[text.index("    // The first 4 KiB") : text.index("    const int64_t batchesPerRow")]
    math = text[text.index("      // Match the current Python path") : text.index("\n    }\n  }\n\n private:")]
    for bank in ("high", "low"):
        math = replace_once(
            math,
            f"      DataCopy({bank}_[first * GROUP / 2], packed, count * GROUP / 2);",
            f"""      for (uint32_t i = 0; i < count; ++i) {{
        const int64_t dst = (routed ? routes[row + start + i] : row + start + i) * 5 + group;
        DataCopy({bank}_[dst * HALF_GROUP], packed[i * HALF_GROUP], HALF_GROUP);
      }}""",
        )
    math = replace_once(
        math,
        "      DataCopy(scale_[first * LANES], broadcast, count * LANES);\n"
        "      DataCopy(sum_[first * LANES], sumBroadcast, count * LANES);",
        """      for (uint32_t i = 0; i < count; ++i) {
        const int64_t dst = (routed ? routes[row + start + i] : row + start + i) * 5 + group;
        DataCopy(scale_[dst * LANES], broadcast[i * LANES], LANES);
        DataCopy(sum_[dst * LANES], sumBroadcast[i * LANES], LANES);
      }""",
    )
    return (
        """// SPDX-License-Identifier: Apache-2.0
// Generated from pinned SwiGLU/packing arithmetic; projection halves stay in UB.
#include "kernel_operator.h"
namespace tp6_pack {
using namespace AscendC;
constexpr uint32_t GROUP=128, BATCH=8, LANES=8, HALF_GROUP=64, ELEMENTS=1024;
class LocalPack {
 public:
  __aicore__ inline void Init(TPipe& pipe, GM_ADDR low, GM_ADDR high, GM_ADDR scale, GM_ADDR sum) {
    pipe.InitBuffer(storage_, 32*1024);
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    sum_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
  }
  __aicore__ inline void Emit(LocalTensor<half> gate, LocalTensor<half> up, int64_t group,
      int64_t row, uint32_t rows, const uint32_t* routes, bool routed) {
"""
        + buffers
        + """    for (uint32_t start=0; start<rows; start+=BATCH) {
      const uint32_t count = rows-start < BATCH ? rows-start : BATCH;
      Duplicate(gateInput, static_cast<half>(0), ELEMENTS);
      Duplicate(upInput, static_cast<half>(0), ELEMENTS);
      PipeBarrier<PIPE_V>();
      for (uint32_t i=0; i<count; ++i) {
        for (uint32_t nb=0; nb<8; ++nb) {
          // Native NZ tile: eight 16-column strips, each with 16 rows.
          // Signed zero is immaterial after the integer quantization boundary.
          Adds(gateInput[i*GROUP+nb*16], gate[nb*256+(start+i)*16], static_cast<half>(0), 16);
          Adds(upInput[i*GROUP+nb*16], up[nb*256+(start+i)*16], static_cast<half>(0), 16);
        }
      }
      PipeBarrier<PIPE_V>();
"""
        + math
        + """
    }
  }
 private:
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<float> scale_, sum_;
};
}
"""
    )


def sources(root):
    root = Path(root)
    headers = {}
    for name in HASHES:
        if name == "pack":
            continue
        # Keep this experiment's compiled arithmetic independent of newer
        # optional schedules. The byte-exact snapshot is from 139f20832.
        path = (
            Path("tools/qwen4exp/tp6_frozen") / f"{name}.txt"
            if name == "native_int4_schedule.h"
            else HEADER_ROOT / name
        )
        headers[name] = (root / path).read_bytes()
    for name, data in headers.items():
        if digest(data) != HASHES[name]:
            raise ValueError(f"changed pinned header: {name}")
    return {
        **headers,
        "tp6_local_schedule.h": private_schedule(headers["native_int4_schedule.h"]).encode(),
        "tp6_local_pack.h": local_pack((root / PACK_SOURCE).read_bytes()).encode(),
    }
