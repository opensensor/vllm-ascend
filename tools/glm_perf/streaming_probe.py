# SPDX-License-Identifier: Apache-2.0
"""Bounded, watchdog-supervised native streaming gates; no serving mutations."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import regex as re

from tools.glm_perf.build_streaming import DISPATCH, MEMORY_COMPONENTS, admit_candidate
from tools.glm_perf.prefill_memory_budget import moe_scratch_bytes
from tools.glm_perf.streaming_protocol import evidence_schema, file_digest, validate_evidence

EXPERTS = 72
TOP_K = 8
MAX_TOKENS = 1280
MIB = 1024**2


def memory_envelope(evidence: dict) -> dict:
    """Whole incremental device peak with the resident server still allocated."""
    for name in ("free_bytes", "resident_allocated_bytes", "runtime_context_bytes", "safety_bytes"):
        if type(evidence.get(name)) is not int or evidence[name] <= 0:
            raise ValueError(f"measured memory evidence requires positive {name}")
    if not isinstance(evidence.get("raw_provenance"), list) or not evidence["raw_provenance"]:
        raise ValueError("memory evidence requires raw measurement provenance")
    scratch = max(moe_scratch_bytes(MAX_TOKENS, h, i)["total_bytes"] for h, i in ((4096, 2048), (2048, 4096)))
    # A8's two activation limbs and padded scale banks have no routed A4 cache.
    a8 = max(
        2 * MAX_TOKENS * h
        + MAX_TOKENS * (h // 32) * 8 * 4
        + 2 * MAX_TOKENS * TOP_K * i
        + MAX_TOKENS * TOP_K * (i // 32) * 8 * 4
        + MAX_TOKENS * TOP_K * h * 2
        for h, i in ((4096, 2048), (2048, 4096))
    )
    scratch = max(scratch, a8)
    components = {
        "bank_fixtures": EXPERTS * 3 * 4096 * 2048 // 2,
        "bank_scales": EXPERTS * 3 * (4096 // 32) * (2048 // 32) * 2,
        "reference_scratch": scratch,
        "candidate_scratch": scratch,
        "outputs": 4 * MAX_TOKENS * 4096 * 4,
        "verification": 64 * MIB,
        "graph_and_config_reserve": 128 * MIB,
        "second_process_runtime": evidence["runtime_context_bytes"],
        "safety": evidence["safety_bytes"],
    }
    required = sum(components.values())
    # Existing baseline resources are already deducted from measured free space.
    resident_required = 2 * scratch + components["outputs"] + components["verification"] + components["safety"]
    return {
        "passed": required <= evidence["free_bytes"],
        "required_free_bytes": required,
        "measured_free_bytes": evidence["free_bytes"],
        "resident_already_allocated_bytes": evidence["resident_allocated_bytes"],
        "components": components,
        "resident_memory_admission": {
            "passed": resident_required <= evidence["free_bytes"],
            "required_free_bytes": resident_required,
            "includes": sorted(MEMORY_COMPONENTS),
            "old_resources": "already included in resident_already_allocated_bytes; not double-counted",
            "new_resources": "kernel handles/configs covered by safety; two scratch pools conservatively included",
        },
    }


def cases() -> list[dict]:
    result = []
    for name in evidence_schema()["x-required-coverage"]["native_gates"]:
        match = re.fullmatch(r"w([234])_a([48])_k(2048|4096)_rows(\d+)_(balanced|skewed|all_peer)", name)
        if match:
            bits, activation, k, rows = map(int, match.group(1, 2, 3, 4))
            pattern = match[5]
            tokens = max(17, math.ceil(rows * (EXPERTS if pattern == "balanced" else 1) / TOP_K))
        else:
            match = re.fullmatch(r"w([234])_a4_chunk(640|1280)_topk8_e72", name)
            if match is None:
                raise ValueError("unsupported frozen native coverage case")
            bits, tokens = map(int, match.group(1, 2))
            activation, k, rows, pattern = 4, 4096, None, "balanced"
        result.append(dict(id=name, bits=bits, activation=activation, k=k, rows=rows, pattern=pattern, tokens=tokens))
    # A diagnostic limit must first exercise both streaming projections, not a no-op.
    smoke = "w4_a4_k4096_rows5_balanced"
    result.sort(key=lambda item: item["id"] != smoke)
    return result


def select_cases(case_ids=(), *, limit=0):
    available = cases()
    if case_ids:
        if limit:
            raise ValueError("case-id selection and case-limit are mutually exclusive")
        by_id = {case["id"]: case for case in available}
        if len(set(case_ids)) != len(case_ids) or any(name not in by_id for name in case_ids):
            raise ValueError("case-id selection must contain unique frozen native case IDs")
        return [by_id[name] for name in case_ids]
    return available[:limit] if limit else available


def route_ids(case: dict, *, replay: int = 0) -> list[list[int]]:
    """Frozen CPU fixture; row labels describe initial per-expert active counts."""
    routes = case["tokens"] * TOP_K
    ids = [EXPERTS + replay] * routes
    if case["pattern"] != "all_peer":
        rows = case["rows"]
        if rows is None:
            ids = [(slot + replay) % EXPERTS for slot in range(routes)]
        elif case["pattern"] == "balanced":
            for expert in range(EXPERTS):
                for row in range(rows):
                    ids[expert * rows + row] = (expert + replay) % EXPERTS
        else:
            for row in range(rows):
                ids[row] = replay % EXPERTS
        # Replay changes the actual count as well as expert IDs; keep the bank ABI.
        if replay and rows is not None and rows and routes > rows:
            ids[0] = EXPERTS
        if replay and rows == 0:
            ids[0] = replay % EXPERTS
    return [ids[index : index + TOP_K] for index in range(0, routes, TOP_K)]


def _package(build: Path):
    options = json.loads((build / "provenance.json").read_text())["_build"]
    name = options["helper_package"]
    root = build / name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[name].__file__).resolve() != (root / "__init__.py").resolve():
        raise ValueError("native package namespace collision")
    return options, name


def _verify_reference_build(build: Path, reference: dict) -> None:
    if json.loads((build / "provenance.json").read_text())["_build"] != reference["build_options"]:
        raise ValueError("reference build options differ from frozen v1021")
    for name, digest in reference["binaries"].items():
        if file_digest(build / name) != digest:
            raise ValueError("reference binary differs from frozen v1021")


def _bank(checkpoint: Path, reference: dict, bits: int, k: int):
    """Load prepared Cube bytes directly; one bank remains live at a time."""
    import torch
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    shards = reference["checkpoint"]["native_shards"]
    chosen = None
    for shard in shards:
        if shard.get("rank", 0) != 0:
            continue
        prefix = f"model.language_model.layers.{shard['layer']}.mlp.experts."
        name = prefix + "0.gate_proj_codes"
        if name not in index:
            continue
        with safe_open(str(checkpoint / index[name]), framework="pt", device="cpu") as handle:
            shape = handle.get_slice(name).get_shape()
        if shape[-1] * 8 // 4096 == bits:
            chosen = shard, prefix
            break
    if chosen is None:
        raise ValueError(f"no frozen rank-zero checkpoint bank for W{bits}")
    shard, prefix = chosen
    if file_digest(checkpoint / shard["file"]) != shard["sha256"]:
        raise ValueError("checkpoint native shard changed")
    h, i = (4096, 2048) if k == 4096 else (2048, 4096)
    gate = torch.empty((EXPERTS, 2 * i, h * bits // 8), dtype=torch.int8)
    down = torch.empty((EXPERTS, h, i * bits // 8), dtype=torch.int8)
    gs = torch.empty((EXPERTS, 2 * i // 32, h // 32), dtype=torch.float16)
    ds = torch.empty((EXPERTS, h // 32, i // 32), dtype=torch.float16)
    tensor_hashes = {}
    expected_shards = {
        record["file"]: record["sha256"]
        for category in ("native_shards", "scale_shards")
        for record in reference["checkpoint"][category]
    }
    for expert in range(EXPERTS):
        values = {}
        for projection in ("gate", "up", "down"):
            for suffix in ("codes", "scale"):
                key = prefix + f"{expert}.{projection}_proj_{suffix}"
                with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as handle:
                    value = handle.get_tensor(key)
                if suffix == "scale" and value.dtype != torch.float16:
                    raise ValueError("native bank requires persistent FP16 scales")
                values[projection, suffix] = value
                if index[key] not in tensor_hashes:
                    tensor_hashes[index[key]] = file_digest(checkpoint / index[key])
                    if tensor_hashes[index[key]] != expected_shards.get(index[key]):
                        raise ValueError("loaded code/scale shard differs from frozen checkpoint")
        if k == 4096:
            gate[expert, :i].copy_(values["gate", "codes"])
            gate[expert, i:].copy_(values["up", "codes"])
            gs[expert, : i // 32].copy_(values["gate", "scale"])
            gs[expert, i // 32 :].copy_(values["up", "scale"])
            down[expert].copy_(values["down", "codes"])
            ds[expert].copy_(values["down", "scale"])
        else:
            for start in (0, i):
                gate[expert, start : start + i].copy_(values["down", "codes"])
                gs[expert, start // 32 : (start + i) // 32].copy_(values["down", "scale"])
            down[expert].copy_(values["gate", "codes"])
            ds[expert].copy_(values["gate", "scale"])
    return (gate, gs, down, ds), {
        "layer": shard["layer"],
        "shard_sha256": shard["sha256"],
        "projection_mapping": "native" if k == 4096 else "down->gate/up; gate->down",
        "tensor_shard_hashes": tensor_hashes,
    }


def _clear(native, torch):
    native.scratch.clear()
    native.configs.clear()
    gc.collect()
    torch.npu.empty_cache()


def report_artifact_paths(build_dir: Path, legacy_gate_report: Path) -> dict[str, str]:
    """Admission result paths are absolute; formal artifacts keep local names.

    Readers resolve relative evidence paths against the gate report directory,
    whereas CLI paths were supplied against the producer's working directory.
    Resolve once at publication so nested report locations cannot change them.
    """
    build = Path(build_dir).resolve(strict=True)
    return {
        "build_dir": str(build),
        "build_evidence": str((build / "build.json").resolve(strict=True)),
        "legacy_gate_report": str(Path(legacy_gate_report).resolve(strict=True)),
    }


def _same(torch, actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite native checkpoint output")
    if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
        raise AssertionError(f"candidate/reference output bits differ (max abs {(actual - expected).abs().max()})")


def _audit(native, build):
    from tools.glm_perf.native_launch_audit import KERNEL_FILES, NativeLaunchAudit

    audit = NativeLaunchAudit(native, build)
    audit.begin_requests()
    labels = {
        id(getattr(native, attribute)): attribute
        for attribute, _ in KERNEL_FILES
        if getattr(native, attribute, None) is not None
    }
    original = native.launch
    timings = {}

    def launch(kernel, tensors, cores):
        start = time.perf_counter_ns()
        result = original(kernel, tensors, cores)
        label = labels[id(kernel)]
        timings[label] = timings.get(label, 0) + time.perf_counter_ns() - start
        return result

    native.launch = launch
    return audit, timings


def _eager_stages(native, args, torch):
    """One diagnostic eager pass; timing events never enter captured graphs."""
    from tools.glm_perf.native_launch_audit import KERNEL_FILES

    labels = {
        id(getattr(native, attribute)): attribute
        for attribute, _ in KERNEL_FILES
        if getattr(native, attribute, None) is not None
    }
    original = native.launch
    events = []

    def launch(kernel, tensors, cores):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = original(kernel, tensors, cores)
        end.record()
        events.append((labels[id(kernel)], start, end))
        return result

    native.launch = launch
    try:
        output = native(*args)
        torch.npu.synchronize()
        timings = [{"stage": label, "device_ms": start.elapsed_time(end)} for label, start, end in events]
        return output, timings
    finally:
        native.launch = original


def _native_case(candidate, baseline, case, bank, *, replays=4):
    import torch

    codes, gs, down, ds = bank
    generator = torch.Generator().manual_seed(8800 + case["bits"] + case["tokens"])
    x_cpu = torch.randn(case["tokens"], codes.shape[-1] * 8 // case["bits"], generator=generator).half() * 0.1
    x = x_cpu.npu()
    ids = torch.tensor(route_ids(case), dtype=torch.int64).npu()
    weights_cpu = torch.full((case["tokens"], TOP_K), 0.125, dtype=torch.float32)
    weights = weights_cpu.npu()
    args = x, codes, gs, down, ds, weights, ids
    start = time.perf_counter()
    eager_c, stages_c = _eager_stages(candidate, args, torch)
    eager_r, stages_r = _eager_stages(baseline, args, torch)
    _same(torch, eager_c, eager_r)
    torch.npu.synchronize()
    eager_ms = (time.perf_counter() - start) * 1000
    del eager_c, eager_r
    # Different scratch poisons must not affect observable outputs or padding reads.
    for native, poison in ((candidate, 37), (baseline, 71)):
        for buffers in native.scratch.values():
            for tensor in buffers:
                if tensor is not None:
                    tensor.fill_(poison)
    _same(torch, candidate(*args), baseline(*args))
    graph_c, graph_r = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(graph_c):
        out_c = candidate(*args)
    with torch.npu.graph(graph_r):
        out_r = baseline(*args)
    for replay in range(1, replays + 1):
        x.copy_((x_cpu * (1 + replay / 16)).npu())
        ids.copy_(torch.tensor(route_ids(case, replay=replay), dtype=torch.int64).npu())
        weights.copy_((weights_cpu * (1 - replay / 32)).npu())
        # Both resources share these same live bank tensors. Change small byte
        # slices without creating another full-bank device allocation.
        codes.reshape(-1)[:64].add_(1)
        down.reshape(-1)[:64].add_(1)
        gs.mul_(0.9375)
        ds.mul_(0.9375)
        graph_c.replay()
        graph_r.replay()
        torch.npu.synchronize()
        _same(torch, out_c, out_r)
    ids.fill_(EXPERTS)
    graph_c.replay()
    graph_r.replay()
    torch.npu.synchronize()
    _same(torch, out_c, out_r)
    if not torch.equal(out_c.cpu(), torch.zeros_like(out_c.cpu())):
        raise AssertionError("all-peer replay retained stale output")
    del out_c, out_r, graph_c, graph_r
    audit, submission = candidate._probe_audit
    snapshot = audit.snapshot()
    gate_label = "gate_w4_kernel" if case["bits"] == 4 else "gate_kernel"
    down_label = "down_w4_kernel" if case["bits"] == 4 else "down_kernel"
    if not snapshot["counts"].get(gate_label) or not snapshot["counts"].get(down_label):
        raise AssertionError("both native projections were not actually submitted")
    return {
        "id": case["id"],
        "passed": True,
        "same_input_bitwise": True,
        "finite": True,
        "changed_replays": replays,
        "padding_poison_independent": True,
        "all_peer_zero": True,
        "eager_pair_ms": eager_ms,
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "geometry": case,
        "initial_active_rows_per_expert": dict(
            Counter(value for row in route_ids(case) for value in row if value < EXPERTS)
        ),
        "launch_audit": snapshot,
        "stage_cpu_submission_ns": dict(submission),
        "stage_timing_semantics": "cumulative host launch submission time, not device execution time",
        "stage_device_ms": {"candidate": stages_c, "reference": stages_r},
        "stage_device_timing_semantics": "NPU launch timing events on one diagnostic eager pass; not timed E2E",
    }


def worker(args, identity, reference, memory, report):
    # This import occurs only after build identity and memory admission succeed.
    import torch
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    options_c, package_c = _package(args.build_dir)
    options_r, package_r = _package(args.reference_build_dir)
    for build, options in ((args.build_dir, options_c), (args.reference_build_dir, options_r)):
        torch.ops.load_library(str(build / f"glm_reconstruction_bridge_v{options['version']}.so"))
    import importlib  # Worker-only frozen helpers.

    klass_c = importlib.import_module(package_c + ".glm_fused_moe").NativeFusedMoE
    klass_r = importlib.import_module(package_r + ".glm_fused_moe").NativeFusedMoE
    gate_case = importlib.import_module(package_c + ".fused_moe_probe").gate_case
    unpack_cube = importlib.import_module(package_c + ".fused_weight_layout").unpack_cube
    details, legacy = (
        [],
        {
            "schema_version": 1,
            "complete": False,
            "build_options": options_c,
            "binaries": identity["binaries"],
            "records": [],
            "real_weight_records": [],
        },
    )
    selected = select_cases(args.case_id, limit=args.case_limit)
    diagnostic = bool(args.case_limit or args.case_id)
    try:
        for bits in (4, 2, 3):
            for k in (4096, 2048):
                group = [case for case in selected if case["bits"] == bits and case["k"] == k]
                if not group:
                    continue
                bank_cpu, provenance = _bank(args.checkpoint, reference, bits, k)
                bank = tuple(value.npu() for value in bank_cpu)
                for activation in (4, 8):
                    candidate = klass_c(
                        args.build_dir,
                        namespace=options_c["namespace"],
                        activation_bits=activation,
                        prepared_weight_layout=True,
                        fp16_route_workspace=True,
                        shared_prefill_max_tokens=MAX_TOKENS,
                    )
                    baseline = klass_r(
                        args.reference_build_dir,
                        namespace=options_r["namespace"],
                        activation_bits=activation,
                        prepared_weight_layout=True,
                        fp16_route_workspace=True,
                        shared_prefill_max_tokens=MAX_TOKENS,
                    )
                    candidate._probe_audit = _audit(candidate, args.build_dir)
                    for case in group:
                        if case["activation"] != activation:
                            continue
                        record = _native_case(candidate, baseline, case, bank, replays=args.replays)
                        record["checkpoint"] = provenance
                        details.append(record)
                        report["coverage"].append(case["id"])
                        print(json.dumps({key: record[key] for key in ("id", "passed", "eager_pair_ms")}), flush=True)
                        _clear(candidate, torch)
                        _clear(baseline, torch)
                        # Restore each deliberately changed bank slice and scale
                        # bank, avoiding cumulative underflow across the matrix.
                        for device, original in zip(bank, bank_cpu):
                            if device.dtype in (torch.int8, torch.uint8):
                                device.reshape(-1)[:64].copy_(original.reshape(-1)[:64].npu())
                            else:
                                device.copy_(original.npu())
                    # Small independent CPU arithmetic gates consume Cube-packed
                    # real codes through the correct inverse before slicing.
                    if k == 4096 and not diagnostic:
                        g = unpack_cube(bank_cpu[0][:1].contiguous(), 4096)[:, :512, :256].contiguous()
                        d = unpack_cube(bank_cpu[2][:1].contiguous(), 2048)[:, :256, :256].contiguous()
                        real = g, bank_cpu[1][:1, :16, :8].float(), d, bank_cpu[3][:1, :8, :8].float()
                        for tokens in (2, 128):
                            legacy["records"].append(gate_case(candidate, bits, tokens=tokens))
                            _clear(candidate, torch)
                        for tokens in (2, 31):
                            legacy["real_weight_records"].append(gate_case(candidate, bits, tokens=tokens, real=real))
                            _clear(candidate, torch)
                        if activation == 4:
                            legacy["records"].append(gate_case(candidate, bits, tokens=31, down_scale_boundary=True))
                            _clear(candidate, torch)
                            legacy["real_weight_records"].append(
                                gate_case(candidate, bits, tokens=31, real=real, down_scale_boundary=True)
                            )
                            _clear(candidate, torch)
                    del candidate, baseline
                del bank, bank_cpu
                gc.collect()
                torch.npu.empty_cache()
        legacy["complete"] = not diagnostic and len(report["coverage"]) == len(cases())
        records_file = args.output.with_name(args.output.stem + "-records.json")
        records_file.write_text(json.dumps(details, indent=2) + "\n")
        legacy_file = args.output.with_name(args.output.stem + "-legacy.json")
        legacy_file.write_text(json.dumps(legacy, indent=2) + "\n")
        report["artifacts"] += [
            {"path": path.name, "sha256": file_digest(path)} for path in (records_file, legacy_file)
        ]
        report["result"].update(
            {
                key: True
                for key in evidence_schema()["x-complete-result-rules"]["native_gates"]
                if key != "watchdog_passed"
            }
        )
        report["result"].update(
            **report_artifact_paths(args.build_dir, legacy_file),
            memory_envelope=memory,
            resident_memory_admission=memory["resident_memory_admission"],
        )
        return report
    except Exception as error:
        report["state"] = "failed"
        report["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "reference", "reference-build-dir", "checkpoint", "memory-evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[], help="Repeatable frozen diagnostic case ID")
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--watchdog-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case_limit < 0 or args.replays < 3 or args.watchdog_seconds <= 0:
        parser.error("case limit is nonnegative; at least three changed replays and a positive watchdog are required")
    select_cases(args.case_id, limit=args.case_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = admit_candidate(args.build_dir, live=False)
    reference = json.loads(args.reference.read_text())
    if reference["reference_sha256"] != identity["reference_sha256"]:
        raise ValueError("frozen reference identity mismatch")
    _verify_reference_build(args.reference_build_dir, reference)
    for name, descriptor in reference["checkpoint"]["descriptors"].items():
        if file_digest(args.checkpoint / name) != descriptor["sha256"]:
            raise ValueError("checkpoint descriptor changed")
    evidence = json.loads(args.memory_evidence.read_text())
    for record in evidence["raw_provenance"]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = args.memory_evidence.parent / path
        if file_digest(path) != record["sha256"]:
            raise ValueError("raw memory provenance changed")
    memory = memory_envelope(evidence)
    memory_copy = args.output.with_name(args.output.stem + "-memory.json")
    if args.memory_evidence.resolve() != memory_copy.resolve():
        shutil.copyfile(args.memory_evidence, memory_copy)
    report = {
        "schema_version": 1,
        "kind": "native_gates",
        "state": "pending",
        "identity": identity,
        "dispatch": dict(DISPATCH),
        "coverage": [],
        "artifacts": [{"path": memory_copy.name, "sha256": file_digest(memory_copy)}],
        "errors": [],
        "result": {"memory_envelope": memory},
    }
    if not memory["passed"]:
        report["state"] = "blocked"
        report["errors"] = ["standalone peak exceeds measured free memory; no device initialization attempted"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.worker:
        worker(args, identity, reference, memory, report)
        return
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "tools.glm_perf.streaming_probe", *sys.argv[1:], "--worker"],
            check=True,
            timeout=args.watchdog_seconds,
        )
        report = json.loads(args.output.read_text())
        report["result"]["watchdog_passed"] = True
        if not (args.case_limit or args.case_id) and set(report["coverage"]) == set(
            evidence_schema()["x-required-coverage"]["native_gates"]
        ):
            report["state"] = "complete"
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            from tools.glm_perf.fused_moe_control import manifest

            manifest(args.build_dir, Path(report["result"]["legacy_gate_report"]), streaming_gate_report=args.output)
            validate_evidence(report, kind="native_gates", bindings=identity, artifact_root=args.output.parent)
    except Exception as error:
        report = json.loads(args.output.read_text())
        report["state"] = "failed"
        report["errors"].append(f"supervisor: {type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
