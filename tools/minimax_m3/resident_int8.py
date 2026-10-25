# SPDX-License-Identifier: Apache-2.0
"""Reversibly reorder resident W4 banks and switch their compute implementation."""

import argparse
import dataclasses
import fcntl
import inspect
import json
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3.resident import candidate_source

REPACK_EXPERT_CHUNK = 8
REPACK_MARGIN_BYTES = 256 * 1024**2
MAX_PROJECTION_NRMSE = 0.03


def reorder_codes(codes, *, inverse=False, chunk=REPACK_EXPERT_CHUNK):
    """Allocate one destination bank; keep transpose scratch bounded to a chunk."""
    import torch

    if codes.ndim != 3 or codes.dtype != torch.int8 or codes.shape[1] % 16 or codes.shape[2] % 128 or chunk < 1:
        raise ValueError("repacking requires signed-byte N16/K256 expert banks and a positive chunk")
    experts, rows, packed_cols = codes.shape
    result = torch.empty_like(codes)
    inner = (16, 32) if inverse else (32, 16)
    for start in range(0, experts, chunk):
        count = min(chunk, experts - start)
        source = codes[start : start + count].reshape(count, rows // 16, packed_cols // 128, 4, *inner)
        result[start : start + count].copy_(source.permute(0, 1, 2, 3, 5, 4).reshape_as(result[start : start + count]))
        if codes.device.type != "cpu":
            # Release each transient transpose before processing the next chunk.
            torch.npu.synchronize()
    return result


def convert_banks(model, original, proposed, enable):
    # Worker-only imports; the controller never opens a device.
    import torch

    from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts

    if not proposed.requires_int8_layout:
        raise ValueError("resident repacking requires the prepacked INT8 candidate")
    if not enable:
        # Cached quantization metadata owns code references. Release these before
        # replacing banks so rollback does not retain a second model in memory.
        proposed.banks.clear()
        proposed.scales.clear()
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for bank in model.modules():
            if not isinstance(bank, PackedM3Experts):
                continue
            if not hasattr(bank, "_m3_int8_layouts"):
                bank._m3_int8_layouts = set()
            for projection in ("w1", "w3", "w2"):
                is_int8 = projection in bank._m3_int8_layouts
                if is_int8 == enable:
                    continue
                codes = getattr(bank, projection + "_codes")
                scales = getattr(bank, projection + "_scales").float()
                free, _ = torch.npu.mem_get_info()
                reusable = torch.npu.memory_reserved() - torch.npu.memory_allocated()
                bytes_per_expert = codes[0].numel()
                required = (
                    codes.numel() + 2 * min(REPACK_EXPERT_CHUNK, len(codes)) * bytes_per_expert + REPACK_MARGIN_BYTES
                )
                if free + reusable < required:
                    raise RuntimeError(f"resident bank repack needs {required} bytes; only {free + reusable} available")
                # One small real-weight comparison per projection bank includes
                # four routed experts. No attention/cache/model generation runs.
                x = (
                    torch.randn(4, codes.shape[-1] * 2, generator=torch.Generator().manual_seed(53))
                    .half()
                    .to(codes.device)
                )
                ends = torch.arange(1, len(codes) + 1, dtype=torch.int64, device=codes.device).clamp_max(4)
                source_kernel = proposed if is_int8 else original
                expected = source_kernel(x, codes, scales, ends).cpu().float() if enable else None
                converted = reorder_codes(codes, inverse=not enable)
                setattr(bank, projection + "_codes", converted)
                if enable:
                    bank._m3_int8_layouts.add(projection)
                else:
                    bank._m3_int8_layouts.discard(projection)
                record = {"layer": bank.prefix, "projection": projection, "bytes": codes.numel(), "passed": True}
                del codes
                if enable:
                    actual = proposed(x, converted, scales, ends).cpu().float()
                    error = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
                    record.update(nrmse=error, finite=bool(actual.isfinite().all()))
                    record["passed"] = record["finite"] and error < MAX_PROJECTION_NRMSE
                records.append(record)
                if not record["passed"]:
                    raise RuntimeError(f"resident INT8 comparison failed: {record}")
    return {"passed": True, "enabled": enable, "seconds": time.monotonic() - started, "banks": records}


def conversion_replacements(native_resources, baseline, candidate, enable):
    from tools.minimax_m3.resident_int8 import convert_banks

    original, proposed = native_resources[baseline], native_resources[candidate]

    def capture(self):
        self._m3_idle()
        try:
            result = convert_banks(self.model_runner.model, original, proposed, enable)
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        return {**self.resident_status(), "conversion": result}

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture": capture}


def convert(client, baseline, candidate, enable):
    source = (
        inspect.getsource(conversion_replacements)
        + "\ndef replacements(native_resources):\n"
        + f"    return conversion_replacements(native_resources, {baseline!r}, {candidate!r}, {enable!r})\n"
    )
    control = Control(uuid.uuid4().hex, "direct-target", "int8_layout_conversion", source)
    client.rpc("resident_prepare", json.dumps(dataclasses.asdict(control)))
    client.rpc("resident_apply", control.generation)
    result = client.rpc("resident_capture")
    if not all(row["conversion"]["passed"] for row in result):
        raise RuntimeError(f"resident bank conversion did not finish: {result}")
    return result


def transition(client, manifest, baseline, enable):
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise ValueError("refusing to take ownership of an existing pause")
    before = client.rpc("resident_status")
    previous_name = baseline if enable else manifest.name
    previous = candidate_source(native_name=previous_name)
    expected = Control(before[0]["generation"], "direct-target", before[0]["candidate"], previous)
    if any(row["digest"] != expected.digest or row["native_failed"] for row in before):
        raise ValueError("transition requires the expected healthy native-only generation")
    client.request("/pause?mode=wait&clear_cache=true")
    changed = False
    try:
        prepared = client.rpc("resident_native_prepare", manifest.payload)
        if any(row["native_digest"] != manifest.digest for row in prepared):
            raise RuntimeError("native manifests disagree")
        loaded = client.rpc("resident_native_load", manifest.digest)
        if any(not row["validation"]["passed"] for row in loaded):
            raise RuntimeError("native validation failed")
        changed = True
        conversion = convert(client, baseline, manifest.name, enable)
        target = manifest.name if enable else baseline
        after = client.switch(
            Control(uuid.uuid4().hex, "direct-target", "m3_candidate", candidate_source(native_name=target))
        )
        if sorted((w["rank"], w["pid"]) for w in before) != sorted((w["rank"], w["pid"]) for w in after):
            raise RuntimeError("worker processes changed during resident conversion")
    except BaseException:
        if changed:
            convert(client, baseline, manifest.name, not enable)
            client.switch(expected)
        status = client.rpc("resident_status")
        if all(not row["native_failed"] and not row["graphs_dirty"] for row in status):
            client.resume()
        raise
    client.resume()
    return {"before": before, "after": after, "conversion": conversion, "cache_reset": True, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enable", "disable"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", default="minimax_w4_v12")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = transition(
            ResidentClient(args.url, 6),
            NativeManifest(json.loads(args.manifest.read_text())),
            args.baseline,
            args.action == "enable",
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "action": args.action, "workers": len(result["after"])}))


if __name__ == "__main__":
    main()
