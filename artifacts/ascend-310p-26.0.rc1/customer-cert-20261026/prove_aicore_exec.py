#!/usr/bin/env python3
"""Prove that the custom AscendC W2SwigluV310 AI Core kernel executes on the card.

This is the missing leg of the "custom firmware on the cards" chain:

  build  (already proven 2026-10-26: a source edit produced a new .o)
     ->
  EXECUTE on the card's AI Core   (this script)

The script calls the registered custom op `torch.ops._C_ascend.npu_w2_swiglu_310`
on real NPU memory and compares the result against a CPU reference computed
with torch on the host.

The op is a SwiGLU over a [rows, 2*width] fp16 input:

    gate = input[row, 0:width]
    up   = input[row, width:2*width]
    y[row, :] = gate * sigmoid(gate) * up

If the device-side result matches the reference, our own AscendC binary --
compiled for the card's AI Core ISA (arch 0x1029) -- executed on the card.
"""

import sys

import torch
import torch_npu  # noqa: F401

EXT = "/home/matteius/experiments/glm-decode-flags-20261006/glm_decode_flags.so"


def reference(gate_up):
    """Host reference: gate * sigmoid(gate) * up, fp16 in / fp16 out.

    The kernel's layout is SPLIT, not interleaved (see w2_swiglu_v310.cpp):
        input[row, 0:width]      -> gate
        input[row, width:2*width] -> up
    """
    width = gate_up.shape[-1] // 2
    gate = gate_up[..., :width].float()
    up = gate_up[..., width:].float()
    return (gate * torch.sigmoid(gate) * up).to(torch.float16)


def run_case(rows, width):
    torch.manual_seed(0)
    x = torch.randn(rows, 2 * width, dtype=torch.float16)

    op = torch.ops._C_ascend.npu_w2_swiglu_310
    out = op(x.npu()).float().cpu()

    ref = reference(x).float()
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-9
    rel = err / scale

    # The kernel clamps to +-FP16_MAX, so also check no NaN/Inf leaked through.
    finite = bool(torch.isfinite(out).all())

    status = "OK" if rel < 0.02 and finite else "MISMATCH"
    print(f"[rows={rows:<4d} width={width:<6d}] maxabs={err:.5f} rel={rel:.5f} finite={finite} {status}")
    return status == "OK"


def main():
    torch.ops.load_library(EXT)
    if not hasattr(torch.ops._C_ascend, "npu_w2_swiglu_310"):
        print("op npu_w2_swiglu_310 NOT REGISTERED")
        return 1

    torch.npu.set_device(0)
    print("=== custom AI Core kernel execution proof (W2SwigluV310) ===")
    print(f"device: {torch.npu.get_device_name()}")
    print()

    ok = True
    # decode-ish and prefill-ish geometries
    for rows, width in ((1, 2048), (1, 4096), (8, 2048), (32, 1024), (128, 512)):
        ok &= run_case(rows, width)

    print()
    print(
        "RESULT:",
        "AI Core custom kernel EXECUTED on the card"
        if ok
        else "FAILED -- custom kernel did not produce matching results",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
