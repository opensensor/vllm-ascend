# SPDX-License-Identifier: Apache-2.0
"""Use independently qualified native schedules at the existing bulk boundary."""

from tools.glm_perf.glm_fused_moe import FUSED_REDUCTION_TOKENS


class PrefillDecodeNative:
    """Select using static tensor shape; never read route tensors on the CPU.

    Both resources own their original descriptors and scratch. Resident graph
    capture freezes the selected resource, so decode retains its existing
    kernels even when the prefill schedule adds instructions or register use.
    """

    def __init__(self, prefill, decode):
        for name in ("device", "input_dtype", "activation_bits", "prepared_weight_layout", "fp16_route_workspace"):
            if getattr(prefill, name) != getattr(decode, name):
                raise ValueError("prefill/decode native resources disagree: " + name)
        self.prefill = prefill
        self.decode = decode
        self.input_dtype = prefill.input_dtype
        self.activation_bits = prefill.activation_bits
        self.prepared_weight_layout = prefill.prepared_weight_layout
        self.fp16_route_workspace = prefill.fp16_route_workspace
        self.dispatches = {"prefill": 0, "decode": 0}

    @property
    def scratch(self):
        # The decode resource also stays in native_resources, where the
        # existing graph-pool release visits its scratch independently.
        return self.prefill.scratch

    @property
    def route_workspace_dtype(self):
        return self.prefill.route_workspace_dtype

    def selected(self, tokens):
        return self.prefill if tokens > FUSED_REDUCTION_TOKENS else self.decode

    def geometry(self, x, *args):
        return self.selected(x.shape[0]).geometry(x, *args)

    def pack_weight_codes(self, signed, bits):
        return self.prefill.pack_weight_codes(signed, bits)

    def __call__(self, x, *args, **kwargs):
        native = self.selected(x.shape[0])
        self.dispatches["prefill" if native is self.prefill else "decode"] += 1
        return native(x, *args, **kwargs)

    def grouped(self, *args):
        geometry = args[-1]
        native = self.selected(geometry.tokens)
        self.dispatches["prefill" if native is self.prefill else "decode"] += 1
        return native.grouped(*args)
