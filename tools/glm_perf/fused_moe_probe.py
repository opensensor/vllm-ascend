# SPDX-License-Identifier: Apache-2.0
"""Independent CPU arithmetic and full fused-MoE graph gates."""

import hashlib
import json

import torch

from .glm_fused_moe import FUSED_REDUCTION_TOKENS, INPUT_SCALE_MIN_ROWS, REDUCE_CACHE_MIN_TOKENS, NativeFusedMoE
from .glm_int4 import unpack_canonical_codes


def projection_reference(inputs, signed, scales, activation_bits):
    groups = inputs.float().reshape(inputs.shape[0], -1, 32)
    limit = (1 << (activation_bits - 1)) - 1
    maximum = groups.abs().amax(-1)
    sx = torch.where(maximum > 0, maximum / limit, torch.ones_like(maximum))
    quant = (groups / sx[..., None]).round().clamp(-limit, limit)
    sw = scales.half().float().repeat_interleave(32, 0)
    result = torch.zeros(inputs.shape[0], signed.shape[0], dtype=torch.float32)
    for group in range(signed.shape[1] // 32):
        result += (quant[:, group] @ signed[:, group * 32 : (group + 1) * 32].float().T) * (
            sx[:, group, None] * sw[None, :, group]
        )
    return result.half()


def reference(x, gate, gate_scales, down, down_scales, weights, ids, activation_bits, offset=0):
    result = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    for expert in range(gate.shape[0]):
        for token in range(x.shape[0]):
            for slot in range(ids.shape[1]):
                if ids[token, slot] != expert + offset or weights[token, slot] == 0:
                    continue
                gate_up = projection_reference(x[token : token + 1], gate[expert], gate_scales[expert], activation_bits)
                g, u = gate_up.chunk(2, -1)
                hidden = (torch.nn.functional.silu(g.float()) * u.float()).clamp(-65504, 65504).half()
                projected = projection_reference(hidden, down[expert], down_scales[expert], activation_bits)
                result[token] += projected[0].float() * weights[token, slot]
    return result


def route_metadata(ids, weights, experts, offset):
    # FP32 keys use AI Core sorting; routes remain entirely device-resident.
    flat = ids.flatten() - offset
    local = (flat >= 0) & (flat < experts) & (weights.flatten() != 0)
    keys = torch.where(local, flat, experts).to(torch.int32).float()
    order = torch.argsort(keys, stable=True).contiguous()
    ends = (
        (keys[None, :] < torch.arange(1, experts + 1, device=keys.device)[:, None]).sum(-1).to(torch.int64).contiguous()
    )
    return order, ends


def gate_case(
    native, bits, tokens=2, hidden=256, inter=256, real=None, nz_density_boundary=False, input_scale_boundary=False
):
    generator = torch.Generator().manual_seed(5700 + bits + tokens)
    experts = 3 if real is None else 1
    if real is None:
        gate = torch.randint(
            -(1 << (bits - 1)), 1 << (bits - 1), (experts, 2 * inter, hidden), generator=generator, dtype=torch.int8
        )
        down = torch.randint(
            -(1 << (bits - 1)), 1 << (bits - 1), (experts, hidden, inter), generator=generator, dtype=torch.int8
        )
        gs = torch.rand(experts, 2 * inter // 32, hidden // 32, generator=generator) * 0.01 + 0.001
        ds = torch.rand(experts, hidden // 32, inter // 32, generator=generator) * 0.01 + 0.001
    else:
        gate, gs, down, ds = real
        inter, hidden = gate.shape[1] // 2, gate.shape[2]
    x = torch.randn(tokens, hidden, generator=generator).half()
    ids = torch.randint(0, experts + 1, (tokens, 2), generator=generator, dtype=torch.int64)
    ids[0] = torch.tensor([0, 0])
    weights = torch.rand(tokens, 2, generator=generator).float()
    if tokens > 1:
        weights[-1, 0] = 0
    offset = 5
    ids += offset
    boundary = getattr(native, "nz_prefill_min_rows", 0)
    if input_scale_boundary:
        if nz_density_boundary or not getattr(native, "group_major_input_scales", False) or native.activation_bits != 4:
            raise ValueError("input scale boundary gate requires the paired A4 layout")
        boundary = INPUT_SCALE_MIN_ROWS
    boundary_gate = nz_density_boundary or input_scale_boundary
    if boundary_gate:
        if not boundary or tokens <= FUSED_REDUCTION_TOKENS or ids.numel() < boundary:
            raise ValueError("NZ boundary gate requires bulk tokens and an explicit reachable threshold")
        ids.fill_(offset + experts)
        ids.view(-1)[: boundary - 1] = offset
        weights.fill_(0.75)
    gx, gi, gw = x.npu(), ids.npu(), weights.npu()
    gc, dc = native.pack_weight_codes(gate, bits).npu(), native.pack_weight_codes(down, bits).npu()
    # Qualification is explicit and performed on the small CPU fixture. Keep
    # the independent reference's original scales and half-rounding operation.
    prepared = getattr(native, "prerounded_weight_scales", False)
    half_storage = getattr(native, "fp16_weight_scales", False)
    ggs, gds = (
        (value.half() if half_storage else value.half().float() if prepared else value).npu() for value in (gs, ds)
    )

    def pipeline():
        return native(gx, gc, ggs, dc, gds, gw, gi, offset)

    def check(actual):
        expected = reference(gx.cpu(), gate, gs, down, ds, gw.cpu(), gi.cpu(), native.activation_bits, offset)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.002, atol=0.002)

    actual = pipeline()
    check(actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = pipeline()
    gx.copy_((x * 1.25).npu())
    changed_ids = torch.full_like(ids, offset)
    if boundary_gate:
        changed_ids.fill_(offset + experts)
        changed_ids.view(-1)[:boundary] = offset
    gi.copy_(changed_ids.npu())
    gw.copy_((weights * 0.75).npu())
    gate = torch.roll(gate, 1, -1)
    down = torch.roll(down, 2, -1)
    gc.copy_(native.pack_weight_codes(gate, bits).npu())
    dc.copy_(native.pack_weight_codes(down, bits).npu())
    graph.replay()
    torch.npu.synchronize()
    check(actual)
    gi.copy_(torch.full_like(ids, offset + experts).npu())
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), torch.zeros(tokens, hidden))
    return {
        "weight_bits": bits,
        "prerounded_weight_scales": prepared,
        "fp16_weight_scales": half_storage,
        "compact_w4_scratch": getattr(native, "compact_w4_scratch", False),
        "activation_bits": native.activation_bits,
        "fused_scale_accumulation": getattr(native, "fused_scale_accumulation", False),
        "nz_prefill_min_rows": getattr(native, "nz_prefill_min_rows", 0),
        "cache_expert_ends": getattr(native, "cache_expert_ends", False),
        "prefill_reduce_meta_cache": getattr(native, "prefill_reduce_meta_cache", False),
        "direct_compact_down_scales": getattr(native, "direct_compact_down_scales", False),
        "nz_boundary_rows": [boundary - 1, boundary] if nz_density_boundary else [],
        "group_major_input_scales": getattr(native, "group_major_input_scales", False),
        "input_scale_boundary_rows": [boundary - 1, boundary] if input_scale_boundary else [],
        "tokens": tokens,
        "hidden": hidden,
        "intermediate": inter,
        "passed": True,
        "graph_changed_inputs_routes_weights": True,
        "all_peer_output_zero": True,
        "native_stages": ["input_quant_once"]
        + (
            ["route_input_pack"]
            if getattr(native, "route_packed_input", False)
            and native.activation_bits == 4
            and tokens > FUSED_REDUCTION_TOKENS
            else []
        )
        + ["gate_up_swiglu_quant", "down_weighted_reduce"]
        + (["stable_prefill_reduce"] if tokens > FUSED_REDUCTION_TOKENS else []),
        "input_quantization_passes_per_token": 1,
        "fp16_intermediate_gm_bytes": tokens * ids.shape[1] * hidden * 2
        if tokens > FUSED_REDUCTION_TOKENS and native.fp16_route_workspace
        else 0,
        "fp16_gate_up_hidden_gm_bytes": 0,
        "top_k": ids.shape[1],
        "route_workspace_dtype": str(native.route_workspace_dtype),
        "weighted_fp32_workspace_bytes": tokens * ids.shape[1] * hidden * 4
        if tokens > FUSED_REDUCTION_TOKENS and not native.fp16_route_workspace
        else 0,
        "unweighted_fp16_workspace_bytes": tokens * ids.shape[1] * hidden * 2
        if tokens > FUSED_REDUCTION_TOKENS and native.fp16_route_workspace
        else 0,
    }


def run(build_dir, output, *, checkpoint=None, real_prefixes=()):
    import torch_npu
    from safetensors import safe_open

    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    options = json.loads((build_dir / "provenance.json").read_text())["_build"]
    namespace = options["namespace"]
    bridge = f"glm_reconstruction_bridge_v{options['version']}.so"
    torch.ops.load_library(str(build_dir / bridge))
    names = (bridge, "glm_fused_gate_up.bin", "glm_fused_down.bin", "glm_fused_pack.bin")
    if options.get("specialize_w3"):
        names += ("glm_fused_gate_up_w3.bin", "glm_fused_down_w3.bin")
    if options.get("compact_w4_scratch"):
        names += ("glm_fused_gate_up_w4.bin", "glm_fused_down_w4.bin")
    if (build_dir / "glm_fused_reduce.bin").exists():
        names += ("glm_fused_reduce.bin",)
    if options.get("route_packed_input"):
        names += ("glm_fused_route_input.bin",)
    payload = {
        "schema_version": 1,
        "complete": False,
        "build_options": options,
        "binaries": {name: hashlib.sha256((build_dir / name).read_bytes()).hexdigest() for name in names},
        "records": [],
        "real_weight_records": [],
    }
    try:
        for activation_bits in (8, 4):
            native = NativeFusedMoE(
                build_dir,
                namespace=namespace,
                activation_bits=activation_bits,
                prepared_weight_layout=options.get("prepared_weight_layout", False),
                weight_decode_lut=options.get("weight_decode_lut", False),
                fp16_route_workspace=options.get("fp16_route_workspace", False),
            )
            for bits in (2, 3, 4):
                for tokens in (1, 2, 3, 17, 128):
                    record = gate_case(native, bits, tokens)
                    payload["records"].append(record)
                    print(json.dumps(record), flush=True)
                if options.get("group_major_input_scales") and activation_bits == 4:
                    record = gate_case(native, bits, tokens=31, input_scale_boundary=True)
                    payload["records"].append(record)
                    print(json.dumps(record), flush=True)
                if options.get("nz_prefill_min_rows"):
                    record = gate_case(native, bits, tokens=31, nz_density_boundary=True)
                    payload["records"].append(record)
                    print(json.dumps(record), flush=True)
            if checkpoint:
                index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
                for prefix in real_prefixes:
                    tensors = []
                    hashes = {}
                    for projection in ("gate", "up", "down"):
                        keys = [prefix + projection + "_proj" + suffix for suffix in ("_codes", "_scale")]
                        values = []
                        for key in keys:
                            with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as handle:
                                values.append(handle.get_tensor(key))
                        canonical, scale = values
                        k = scale.shape[1] * 32
                        signed = unpack_canonical_codes(canonical, k)
                        hashes[projection] = hashlib.sha256(
                            canonical.numpy().tobytes() + scale.numpy().tobytes()
                        ).hexdigest()
                        tensors.append((signed, scale.float()))
                    gate = torch.cat((tensors[0][0], tensors[1][0]), 0)[None]
                    gs = torch.cat((tensors[0][1], tensors[1][1]), 0)[None]
                    down, ds = tensors[2][0][None], tensors[2][1][None]
                    packed_k = canonical.shape[1]
                    bits = packed_k * 8 // k
                    real_tokens = (
                        (2, 31)
                        if any(
                            options.get(flag)
                            for flag in (
                                "prefill_weight_cache",
                                "fp16_route_workspace",
                                "share_gate_up_input",
                                "cache_gate_up_activations",
                                "vector_scale_products",
                                "gather_product_matrix",
                                "prefill_rows_32",
                                "active_cube_rows",
                                "direct_w4_l1",
                                "prepared_offset_tables",
                                "quad_hidden_quant",
                                "direct_hidden_gather",
                                "route_packed_input",
                                "route_packed_down",
                                "route_compact_down_scales",
                                "raw_hidden_scales",
                                "raw_input_scales",
                                "nz_prefill_accumulator",
                                "prefill_product_cast",
                            )
                        )
                        else (2,)
                    )
                    for tokens in real_tokens:
                        record = gate_case(native, bits, tokens=tokens, real=(gate, gs, down, ds))
                        record.update(tensor_prefix=prefix, checkpoint_tensor_hashes=hashes)
                        payload["real_weight_records"].append(record)
                        print(json.dumps(record), flush=True)
                    if options.get("prefill_reduce_meta_cache"):
                        record = gate_case(native, bits, tokens=REDUCE_CACHE_MIN_TOKENS, real=(gate, gs, down, ds))
                        record.update(tensor_prefix=prefix, checkpoint_tensor_hashes=hashes)
                        payload["real_weight_records"].append(record)
                        print(json.dumps(record), flush=True)
                    if options.get("group_major_input_scales") and activation_bits == 4:
                        record = gate_case(
                            native, bits, tokens=31, real=(gate, gs, down, ds), input_scale_boundary=True
                        )
                        record.update(tensor_prefix=prefix, checkpoint_tensor_hashes=hashes)
                        payload["real_weight_records"].append(record)
                        print(json.dumps(record), flush=True)
                    if options.get("nz_prefill_min_rows"):
                        record = gate_case(native, bits, tokens=31, real=(gate, gs, down, ds), nz_density_boundary=True)
                        record.update(tensor_prefix=prefix, checkpoint_tensor_hashes=hashes)
                        payload["real_weight_records"].append(record)
                        print(json.dumps(record), flush=True)
        payload["complete"] = True
    except Exception as error:
        payload["error"] = str(error)
        raise
    finally:
        output.write_text(json.dumps(payload, indent=2) + "\n")
