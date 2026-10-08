"""Exercise GLM's grouped-MoE route tiling with the real 310P operator."""

import json
from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend._310p.quantization.methods.w2_dynamic import (
    W2_GROUPED_MAX_ROUTES,
    AscendW2DynamicFusedMoEMethod310,
    _w2_grouped_mm_op,
)
from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz
from vllm_ascend.utils import enable_custom_op


class GroupedBank(list):
    grouped_ready = True
    nz_packed_codes = True
    local_expert_offset = 0
    num_local_experts = 4


def _packed_bank() -> GroupedBank:
    experts = 4
    hidden = inter = 256
    bank = GroupedBank([SimpleNamespace(hidden=hidden, inter=inter) for _ in range(experts)])

    def projection():
        canonical = torch.randint(0, 256, (experts, inter, hidden // 2), dtype=torch.uint8)
        packed = torch.stack([_pack_codes_nz(canonical[expert], hidden) for expert in range(experts)])
        scales = (torch.rand(experts, inter // 32, hidden // 32) * 0.02 + 0.005).float()
        return packed.view(torch.int8).npu(), scales.npu()

    bank.gate_packed_bank, bank.gate_scale_bank = projection()
    bank.up_packed_bank, bank.up_scale_bank = projection()
    bank.down_packed_bank, bank.down_scale_bank = projection()
    bank.gate_up_packed_bank = torch.cat((bank.gate_packed_bank, bank.up_packed_bank), dim=1).contiguous()
    bank.gate_up_scale_bank = torch.cat((bank.gate_scale_bank, bank.up_scale_bank), dim=1).contiguous()
    return bank


def main() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("This probe requires an Ascend 310P device")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(310)
    bank = _packed_bank()
    grouped_op = _w2_grouped_mm_op()
    if grouped_op is None:
        raise RuntimeError("The 310P grouped W2/W4 operator was not loaded")
    method = AscendW2DynamicFusedMoEMethod310()
    top_k = 8
    max_tokens_per_call = W2_GROUPED_MAX_ROUTES // top_k
    results = []
    for num_tokens in (640, 641, 1280, 1281):
        x = torch.randn(num_tokens, 256, dtype=torch.float16, device="npu") * 0.1
        ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int64, device="npu").expand(num_tokens, -1)
        weights = torch.full((num_tokens, top_k), 1 / top_k, dtype=torch.float32, device="npu")
        weights[:, 1] = 0  # Exercise the peer-route sentinel used in EP.
        reference = torch.cat(
            [
                method._apply_device_grouped(
                    grouped_op,
                    bank,
                    x[start : start + max_tokens_per_call],
                    weights[start : start + max_tokens_per_call],
                    ids[start : start + max_tokens_per_call],
                    None,
                )
                for start in range(0, num_tokens, max_tokens_per_call)
            ]
        )
        actual = method._apply_device(bank, x, weights, ids, None)
        torch.testing.assert_close(actual.cpu(), reference.cpu(), atol=0, rtol=0)
        if not torch.isfinite(actual).all():
            raise AssertionError(f"Non-finite output at {num_tokens} tokens")
        results.append({"tokens": num_tokens, "routes": num_tokens * top_k, "bitwise_parity": True})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
