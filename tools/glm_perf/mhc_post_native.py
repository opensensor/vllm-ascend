# SPDX-License-Identifier: Apache-2.0
"""Native post mixer retaining FP32 expert inputs and FP16-rounded mHC state."""

import json

import torch

GLM_HIDDEN_SIZE = 4096
RESIDUAL_STREAMS = 4
MAX_DECODE_ROWS = 8
PREFILL_GRAPH_ROWS = 640
FINAL_POST_ROWS = frozenset({640, 1280})


def supported_inputs(x, residual, post_mix, comb_mix, *, final_only=False):
    if (
        x.ndim != 2
        or x.dtype not in (torch.float16, torch.float32)
        or not x.is_contiguous()
        or not (
            x.shape[0] in FINAL_POST_ROWS
            if final_only
            else (1 <= x.shape[0] <= MAX_DECODE_ROWS or x.shape[0] == PREFILL_GRAPH_ROWS)
        )
        or x.shape[1] != GLM_HIDDEN_SIZE
    ):
        return False
    rows = x.shape[0]
    shapes = (
        (rows, RESIDUAL_STREAMS, GLM_HIDDEN_SIZE),
        (rows, RESIDUAL_STREAMS, 1),
        (rows, RESIDUAL_STREAMS, RESIDUAL_STREAMS),
    )
    return all(
        value.dtype == torch.float32 and value.shape == shape and value.device == x.device and value.is_contiguous()
        for value, shape in zip((residual, post_mix, comb_mix), shapes, strict=True)
    )


class NativeMhcPost:
    final_only = False

    def __init__(self, root, namespace, *, finish_only=False):
        self.finish_only = finish_only
        factory = getattr(torch.classes, namespace).Kernel
        self.kernels = {
            dtype: factory(str(root / f"mhc_post_fp{bits}.bin"), f"glm_mhc_post_fp{bits}_v1")
            for dtype, bits in ((torch.float16, 16), (torch.float32, 32))
        }
        self.launch = getattr(torch.ops, namespace).launch
        self.configs = {}

    def __call__(self, x, residual, post_mix, comb_mix):
        if not supported_inputs(x, residual, post_mix, comb_mix, final_only=self.final_only):
            raise ValueError("native mHC requires qualified GLM rows, precision and contiguous streams")
        key = (x.shape, x.device)
        if key not in self.configs:
            # Descriptor creation must also work on the first captured call.
            config = torch.empty(2, dtype=torch.int64, device=x.device)
            config[0].fill_(x.shape[0])
            config[1].fill_(x.shape[1])
            self.configs[key] = config
        if self.finish_only:
            # Preserve the backend BMM reduction exactly. Fuse only the
            # broadcast product, residual addition and state rounding.
            residual = torch.einsum("nij,nih->njh", comb_mix, residual).contiguous()
        output = torch.empty_like(residual)
        self.launch(self.kernels[x.dtype], [x, residual, post_mix, comb_mix, output, self.configs[key]], 8)
        return output


class NativeMhcFinalPost(NativeMhcPost):
    """Opt-in final mixer: bounded vector tiles and unrounded FP32 output.

    Intermediate state mixers must keep their existing rounding contract.
    Changing the reduction order requires numerical and real-model gates;
    construction alone does not authorize a serving binding.
    """

    final_only = True

    def __init__(self, root, namespace):
        provenance = json.loads((root / "mhc-provenance.json").read_text())
        if provenance.get("state_rounding") != "none_fp32" or provenance.get("finish_only") is not False:
            raise ValueError("final mixer requires a complete, unrounded FP32 build")
        super().__init__(root, namespace)


def wrap_post(original, native):
    original = getattr(original, "__glm_native_post_original__", original)

    def post(x, residual, post_mix, comb_mix):
        if supported_inputs(x, residual, post_mix, comb_mix):
            return native(x, residual, post_mix, comb_mix)
        return original(x, residual, post_mix, comb_mix)

    post.__glm_native_post_original__ = original
    return post
