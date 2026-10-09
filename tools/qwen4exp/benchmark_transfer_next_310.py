# SPDX-License-Identifier: Apache-2.0
"""Deferred component gates; importing or --help submits no NPU work."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tools.glm_perf.resident_native import file_digest


def load_resources(build, *, load_library=True):
    import torch

    from tools.qwen4exp.native_cached_metadata import NativeCachedMetadata
    from tools.qwen4exp.native_state_layout import NativeStateLayout

    provenance = json.loads((build / "provenance.json").read_text())
    for entry in [provenance["bridge"], *provenance["binaries"].values()]:
        if file_digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("native binary digest mismatch")
    if load_library:
        torch.ops.load_library(provenance["bridge"]["path"])
    namespace = provenance["namespace"]
    kernel = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch
    state = provenance["binaries"]["native_state_layout"]["path"]
    return {
        "state_io": NativeStateLayout(
            kernel(state, "qwen_state_gather_v1"), kernel(state, "qwen_state_scatter_v1"), launch
        ),
        "cached_metadata": NativeCachedMetadata(
            kernel(provenance["binaries"]["native_cached_metadata"]["path"], "qwen_cached_metadata_v1"), launch
        ),
    }


def validate_resources(resources):
    import torch

    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device, pack_native_metadata, pack_native_weight

    generator = torch.Generator().manual_seed(918)
    results = []
    for heads, sequences in ((12, 1), (48, 4)):
        cache = torch.randn(8, heads, 128, 128, generator=generator).npu()
        slots = torch.tensor([6, 2, 3, 4][:sequences], dtype=torch.int32).npu()
        initialized = torch.tensor([False, True, True, False][:sequences]).npu()
        expected = cache[slots.long()].clone()
        expected[~initialized] = 0
        actual = resources["state_io"].gather(cache, slots, initialized)
        torch.testing.assert_close(actual.cpu(), expected.transpose(-1, -2).contiguous().cpu(), rtol=0, atol=0)
        resources["state_io"].scatter(cache, slots, initialized, actual)
        torch.testing.assert_close(cache[slots.long()].cpu(), expected.cpu(), rtol=0, atol=0)
        results.append({"case": "state_io", "heads": heads, "sequences": sequences, "exact": True})
    # Compare full recurrent H/O outputs and FP32 final states at sharded and
    # unsharded production geometry; IO parity alone does not qualify GDN.
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
        build_varlen_chunk_plan,
        chunk_gated_delta_rule_310,
    )

    for key_heads, value_heads in ((4, 12), (16, 48)):
        tokens = 160
        q = torch.randn(1, tokens, key_heads, 128, generator=generator).half().npu()
        k = torch.randn(1, tokens, key_heads, 128, generator=generator).half().npu()
        v = torch.randn(1, tokens, value_heads, 128, generator=generator).half().npu()
        g = torch.nn.functional.logsigmoid(torch.randn(1, tokens, value_heads, generator=generator)).npu()
        beta = torch.rand(1, tokens, value_heads, generator=generator).npu()
        cache = (torch.randn(4, value_heads, 128, 128, generator=generator) * 0.01).npu()
        slots = torch.tensor([2, 1], dtype=torch.int32).npu()
        initialized = torch.tensor([False, True]).npu()
        cu_cpu = torch.tensor([0, 64, tokens], dtype=torch.int32)
        cu = cu_cpu.npu()
        plan = build_varlen_chunk_plan(cu_cpu, 64)
        initial = cache[slots.long()].contiguous()
        initial[~initialized] = 0
        expected_out, expected_state = chunk_gated_delta_rule_310(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            chunk_plan=plan,
            use_qk_l2norm_in_kernel=True,
        )
        packed = resources["state_io"].gather(cache, slots, initialized)
        actual_out, actual_state = chunk_gated_delta_rule_310(
            q,
            k,
            v,
            g,
            beta,
            initial_state=packed,
            output_final_state=True,
            cu_seqlens=cu,
            chunk_plan=plan,
            use_qk_l2norm_in_kernel=True,
            state_is_kernel_layout=True,
        )
        resources["state_io"].scatter(cache, slots, initialized, actual_state)
        torch.testing.assert_close(actual_out.cpu(), expected_out.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(cache[slots.long()].cpu(), expected_state.cpu(), rtol=0, atol=0)
        results.append({"case": "gdn_h_o", "key_heads": key_heads, "value_heads": value_heads, "exact": True})
        del cache, initial, packed, actual_out, actual_state, expected_out, expected_state
    # Exercise both production operand geometries and N=160/N=80 specialization.
    # Each bank is temporary; no model weights, server configuration or OPP path changes.
    for outputs, width in ((1280, 2560), (2560, 640)):
        weight, ws = pack_native_weight(
            torch.randint(-128, 128, (128 * outputs, width // 2), dtype=torch.int8, generator=generator)
        )
        scales = pack_native_metadata(torch.full((128 * outputs, width // 128), 0.01, dtype=torch.float16))
        offsets = pack_native_metadata(torch.randint(-8, 8, (128 * outputs, width // 128), generator=generator).half())
        bank = SimpleNamespace(
            weight=weight.reshape(128, outputs, width // 2).npu(),
            weight_sum=ws.reshape(128, outputs, width // 128).npu(),
            weight_scale=scales.reshape(128, outputs, width // 128).npu(),
            weight_offset=offsets.reshape(128, outputs, width // 128).npu(),
        )
        for rows in (129, 321):
            # Several row tiles per expert, partial M tiles, empty groups and peer tail.
            inputs = (torch.randn(rows, width, generator=generator) * 0.1).half().npu()
            packed = pack_activation_device(inputs)
            active = rows - 13
            counts = torch.zeros(128, dtype=torch.int64)
            counts[0] = active // 2
            counts[17] = active - active // 2
            ends = counts.cumsum(0).npu()
            expected = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
                *packed, bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum, ends
            )
            actual = resources["cached_metadata"](bank, packed, ends)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
            results.append({"case": "cached_metadata", "rows": rows, "outputs": outputs, "width": width, "exact": True})
        del bank, weight, ws, scales, offsets, packed, actual, expected, inputs
    return {
        "component_correctness": results,
        "full_model_validated": False,
        "thermal_validated": False,
        "performance_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch_npu  # noqa: F401 -- explicit device gate, never import during help.

    result = validate_resources(load_resources(args.build))
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
