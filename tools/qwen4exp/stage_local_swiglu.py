# SPDX-License-Identifier: Apache-2.0
"""Reuse the native quantizer unchanged; bound work by device local-route ends."""

from pathlib import Path


def source(root: Path) -> str:
    path = root / "csrc/gmm/qwen_w4_a8_swiglu_pack_v310/op_kernel/qwen_w4_a8_swiglu_pack_v310.cpp"
    text = path.read_text()
    marker = 'extern "C" __global__ __aicore__ void qwen_w4_a8_swiglu_pack_v310('
    if text.count(marker) != 1 or "op.Run(gate_up, low, high, scale, sum, td->rows, td->groups_per_row);" not in text:
        raise ValueError("SwiGLU quantizer ABI changed; review before staging")
    return (
        text[: text.index(marker)]
        + """extern "C" __global__ __aicore__ void qwen_local_swiglu_pack_v1(
    GM_ADDR gate_up, GM_ADDR group_ends, GM_ADDR low, GM_ADDR high, GM_ADDR scale, GM_ADDR sum, GM_ADDR config) {
  AscendC::InitSocState();
  auto c = reinterpret_cast<__gm__ int64_t*>(config);
  AscendC::GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_ends));
  const int64_t activeRows = ends.GetValue(c[1] - 1);
  SwigluPackActivation op;
  op.Run(gate_up, low, high, scale, sum, activeRows, c[0]);
}
"""
    )
