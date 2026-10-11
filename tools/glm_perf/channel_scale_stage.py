# SPDX-License-Identifier: Apache-2.0
"""Cold-load admission and composition for a complete GLM pipeline stage."""

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import regex as re

from .channel_scale_checkpoint import file_sha256, load_partition
from .channel_scale_math import LAYOUT
from .channel_scale_runtime import DECODE_ROWS, Geometry, scratch_shapes, shape_bytes, verify_bundle


@dataclass(frozen=True)
class ChannelBankOwner:
    layer_key: str
    local_expert_offset: int
    num_local_experts: int
    grouped_ready: bool = False

    def finalize_grouped_storage(self):
        """Native banks were loaded by the candidate, never the legacy unpacker."""


def frozen_runtime(root):
    root = Path(root).resolve()
    manifest = verify_bundle(root)
    package = manifest["namespace"] + "_helpers"
    folder = root / package
    if not manifest.get("helpers"):
        raise ValueError("cold-load candidate requires frozen Python helpers")
    for name, expected in manifest["helpers"].items():
        if Path(name).name != name:
            raise ValueError("unsafe frozen helper path")
        if file_sha256(folder / name) != expected:
            raise ValueError("frozen channel-scale helper differs: " + name)
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, folder / "__init__.py", submodule_search_locations=[str(folder)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[package]
            raise
    if Path(sys.modules[package].__file__).resolve() != (folder / "__init__.py").resolve():
        raise ValueError("channel-scale namespace is already owned by another build")
    return importlib.import_module(package + ".channel_scale_runtime")


def plan_stage(
    model,
    overlay,
    *,
    max_tokens,
    top_k,
    decode_bits=8,
    prefill_bits=8,
    decode_sizes=None,
    fused_gate_up=False,
    fused_decode=False,
    identity_smoothing=False,
    fused_prefill=False,
):
    """Fail before native allocation if coverage, topology or geometry is incomplete."""
    if decode_sizes is None:
        decode_sizes = range(1, DECODE_ROWS + 1)
    root = Path(overlay)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("layout") != LAYOUT or manifest.get("complete") is not True:
        raise ValueError("requires a complete original-FP8 channel-scale overlay")
    if fused_gate_up and manifest.get("down_group") != 256:
        raise ValueError("UB gate/up fusion requires G256 hidden quantization")
    owners, keys, resident_bytes, diagonal_bytes, geometry = [], set(), 0, 0, None
    for name, module in model.named_modules():
        bank = getattr(module, "w2_experts", None)
        if bank is None:
            continue
        match = re.fullmatch(r"layers\.(\d+)", bank.layer_key)
        if match is None or not hasattr(module, "routed_experts_forward"):
            raise ValueError("candidate requires composed GLM MoE owners with layer identities")
        if bank.grouped_ready or any(
            getattr(bank, key, None) is not None for key in ("gate_up_packed_bank", "down_packed_bank")
        ):
            raise ValueError("channel-scale candidate requires cold loading, never replacing live legacy banks")
        first, stop = bank.local_expert_offset, bank.local_expert_offset + bank.num_local_experts
        layer = int(match[1])
        if layer in keys:
            raise ValueError("duplicate channel-scale layer owner")
        keys.add(layer)
        records = [manifest["experts"].get(f"{layer}:{expert}") for expert in range(first, stop)]
        if not records or any(row is None for row in records):
            raise ValueError("channel-scale overlay does not cover every local expert and layer")
        if any(
            row["gate_shape"] != records[0]["gate_shape"] or row["down_shape"] != records[0]["down_shape"]
            for row in records
        ):
            raise ValueError("channel-scale overlay mixes expert geometry")
        _, twice_i, hidden = records[0]["gate_shape"]
        if records[0]["down_shape"] != [1, hidden, twice_i // 2]:
            raise ValueError("channel-scale gate/up and down metadata disagree")
        current = Geometry(
            stop - first,
            hidden,
            twice_i // 2,
            manifest["gate_group"],
            manifest["down_group"],
            fused_gate_up,
            fused_decode,
            fused_prefill,
        )
        h, i = current.hidden, current.intermediate
        expected_bytes = (
            3 * h * i // 2 + (2 * i * h // current.gate_group + h * i // current.down_group) * 8 + (h + i) * 4
        )
        if any(row["tensor_bytes"] != expected_bytes for row in records):
            raise ValueError("channel-scale manifest byte count differs from physical bank geometry")
        if geometry is not None and geometry != current:
            raise ValueError("candidate stage requires consistent local expert geometry")
        geometry = current
        diagonal_bytes += (stop - first) * (current.hidden + current.intermediate) * 4
        resident_bytes += sum(row["tensor_bytes"] for row in records)
        owners.append((layer, module, first, stop))
    if not owners:
        raise ValueError("candidate requires at least one local GLM MoE layer")
    disk_bytes = resident_bytes
    if identity_smoothing:
        resident_bytes -= diagonal_bytes
    maximum_bits = decode_bits if max_tokens <= DECODE_ROWS else prefill_bits
    scratch_growth = shape_bytes(scratch_shapes(max_tokens, top_k, geometry, maximum_bits))
    scratch_bytes = scratch_growth
    scratch_bytes += sum(
        shape_bytes(scratch_shapes(n, top_k, geometry, decode_bits))
        for n in set(decode_sizes)
        if 1 <= n <= min(max_tokens, DECODE_ROWS) and n != max_tokens
    )
    # Upper bound with private tables/headers per layer, output and bounded route
    # metadata. KV, dense weights and pipeline handoffs need separate reservations.
    metadata_bytes = len(owners) * (512 * 1024 + 288 + geometry.experts * 4 + 4 + max_tokens * (4 * 8 * 8 + top_k * 4))
    transient_bytes = max_tokens * geometry.hidden * 4 + max_tokens * top_k * (5 * geometry.experts + 32) * 4
    workspace_reserve = 64 * 1024**2
    return owners, {
        "resident_expert_bytes": resident_bytes,
        "disk_expert_bytes": disk_bytes,
        "scratch_growth_reserve_bytes": scratch_growth,
        "scratch_bytes": scratch_bytes,
        "metadata_bytes_upper_bound": metadata_bytes,
        "output_and_route_bytes_upper_bound": transient_bytes,
        "backend_workspace_reserve_bytes": workspace_reserve,
        "candidate_peak_bytes": resident_bytes
        + scratch_bytes
        + scratch_growth
        + metadata_bytes
        + transient_bytes
        + workspace_reserve,
        "measurement": "offline byte admission with 64 MiB backend allowance; excludes KV/dense/PP allocations",
    }


def prepare_stage(
    model, overlay, bundle, *, max_tokens, top_k, host_budget_bytes, device_budget_bytes, decode_bits=8, prefill_bits=8
):
    manifest = verify_bundle(bundle)
    owners, ledger = plan_stage(
        model,
        overlay,
        max_tokens=max_tokens,
        top_k=top_k,
        decode_bits=decode_bits,
        prefill_bits=prefill_bits,
        fused_gate_up=manifest.get("fused_gate_up", False),
        fused_decode=manifest.get("fused_decode", False),
        identity_smoothing=manifest.get("identity_smoothing", False),
        fused_prefill=manifest.get("fused_prefill", False),
    )
    if type(device_budget_bytes) is not int or ledger["candidate_peak_bytes"] > device_budget_bytes:
        raise ValueError("new MoE stage exceeds explicit device reservation before allocation")
    helper = frozen_runtime(bundle)
    arena = helper.ScratchArena(ledger["scratch_bytes"])
    for layer, module, first, stop in owners:
        gate, down = load_partition(overlay, layer, first, stop, host_budget_bytes=host_budget_bytes)
        device = next(model.parameters()).device
        native = helper.NativeChannelScaleMoE(
            bundle,
            gate,
            down,
            device=device,
            arena=arena,
            expert_offset=first,
            top_k=top_k,
            max_tokens=max_tokens,
            decode_bits=decode_bits,
            prefill_bits=prefill_bits,
        )
        module.w2_experts = ChannelBankOwner(f"layers.{layer}", first, stop - first)
        module._method = helper.ChannelScaleMoEMethod(native)
    model._channel_scale_load_report = {
        **ledger,
        "layout": LAYOUT,
        "layers": [row[0] for row in owners],
        "legacy_expert_tensors_loaded": 0,
        "runtime_weight_transforms": 0,
    }
    return model._channel_scale_load_report
