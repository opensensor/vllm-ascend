# SPDX-License-Identifier: Apache-2.0
"""Measure matched GLM W4/W3/W2 expert pipelines with 72 resident experts.

Two local routes per token approximate one TP2 rank's share of top-eight.
CUDA uses the deployed packed scalar kernels; NPU uses the frozen native
bundle. Both include activation quantization, SwiGLU and weighted reduction.
Resident timing excludes upload, route sorting, RPC and output download.
"""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

from .architecture_benchmark import COPY_BYTES, SAMPLES, WARMUPS, summarize
from .device_backend import CUDA_NIBBLE_PLANES, CudaOracle
from .native_expert_probe import contained, digest_file
from .resident import NativeResidentPlan

LAYERS = (3, 8, 33)
FIRST_EXPERT = 144
EXPERTS = 72
WIDTH = 4096
TOP_K = 8
LOCAL_ROUTES = 2


def route_fixture(tokens, torch):
    torch.manual_seed(6000310 + tokens)
    hidden = torch.randn(tokens, WIDTH).half()
    ids = torch.full((tokens, TOP_K), -1, dtype=torch.int32)
    weights = torch.zeros(tokens, TOP_K)
    routes = []
    for token in range(tokens):
        for slot in range(LOCAL_ROUTES):
            expert = (token * 13 + slot * 17) % EXPERTS
            ids[token, slot] = FIRST_EXPERT + expert
            weights[token, slot] = 1 / TOP_K
            routes.append((expert, token, slot))
    routes.sort()
    row_map = torch.full((tokens, TOP_K), -1, dtype=torch.int32)
    for index, (_, token, slot) in enumerate(routes):
        row_map[token, slot] = index
    return hidden, ids, weights, routes, row_map


def cuda_measure(oracle, layer, fixture, torch, *, cache_flush=None, activation_bits=4):
    if type(activation_bits) is not int or activation_bits not in (4, 8):
        raise ValueError("CUDA expert activation precision must be A4 or A8")
    hidden, _, weights, routes, row_map = fixture
    tokens = hidden.shape[0]
    oracle.upload(oracle.buffers["input"], hidden[[row[1] for row in routes]].numpy().tobytes())
    local_ids = torch.tensor([row[0] for row in routes], dtype=torch.int32)
    oracle.upload(oracle.buffers["ids"], local_ids.numpy().tobytes())
    oracle.upload(oracle.buffers["row_map"], row_map.numpy().tobytes())
    oracle.upload(oracle.buffers["weights"], weights.numpy().tobytes())
    runtime, check = oracle.cuda.library, oracle.cuda.check
    event_type = ctypes.c_void_p
    for name, signature in {
        "cudaEventCreate": [ctypes.POINTER(event_type)],
        "cudaEventRecord": [event_type, event_type],
        "cudaEventSynchronize": [event_type],
        "cudaEventElapsedTime": [ctypes.POINTER(ctypes.c_float), event_type, event_type],
        "cudaEventDestroy": [event_type],
    }.items():
        getattr(runtime, name).argtypes = signature
        getattr(runtime, name).restype = ctypes.c_int
    begin, end = event_type(), event_type()
    check(runtime.cudaEventCreate(ctypes.byref(begin)))
    check(runtime.cudaEventCreate(ctypes.byref(end)))
    names = ("input_q", "input_scales", "gate_up", "hidden", "hidden_q", "hidden_scales", "output")
    arguments = (oracle.buffers["input"], oracle.buffers["ids"], *oracle.bank_pointers[layer])
    arguments += tuple(oracle.buffers[name] for name in names)
    device, wall = [], []
    try:
        if cache_flush:
            source = oracle.allocate("cache-flush-source", COPY_BYTES)
            target = oracle.allocate("cache-flush-target", COPY_BYTES)
            runtime.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
            runtime.cudaMemset.restype = ctypes.c_int
            check(runtime.cudaMemset(source, 181, COPY_BYTES))
        for iteration in range(WARMUPS + SAMPLES):
            if cache_flush:
                check(cache_flush(target, source, COPY_BYTES))
                check(runtime.cudaDeviceSynchronize())
            start = time.perf_counter()
            check(runtime.cudaEventRecord(begin, None))
            check(oracle.launch(*arguments, len(routes), WIDTH, 2048, activation_bits))
            check(
                oracle.reduce(
                    oracle.buffers["output"],
                    oracle.buffers["row_map"],
                    oracle.buffers["weights"],
                    oracle.buffers["partial_output"],
                    tokens,
                    WIDTH,
                    len(routes),
                )
            )
            check(runtime.cudaEventRecord(end, None))
            check(runtime.cudaEventSynchronize(end))
            elapsed = ctypes.c_float()
            check(runtime.cudaEventElapsedTime(ctypes.byref(elapsed), begin, end))
            if iteration >= WARMUPS:
                device.append(elapsed.value)
                wall.append((time.perf_counter() - start) * 1000)
        actual = (
            torch.frombuffer(bytearray(oracle.read_output("partial_output", tokens * WIDTH * 4)), dtype=torch.float32)
            .reshape(tokens, WIDTH)
            .clone()
        )
    finally:
        check(runtime.cudaEventDestroy(end))
        check(runtime.cudaEventDestroy(begin))
    return actual, device, wall


def npu_measure(native, bank, fixture, torch):
    hidden, ids, weights = (value.npu() for value in fixture[:3])
    device_bank = tuple(value.npu() for value in bank["tensors"])
    begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    device, wall = [], []
    for iteration in range(WARMUPS + SAMPLES):
        torch.npu.synchronize()
        start = time.perf_counter()
        begin.record()
        actual = native(hidden, *device_bank, weights, ids, FIRST_EXPERT)
        end.record()
        end.synchronize()
        if iteration >= WARMUPS:
            device.append(begin.elapsed_time(end))
            wall.append((time.perf_counter() - start) * 1000)
    return actual.cpu(), device, wall


def cuda_cold_measure(oracle, layer, bank, fixture, torch, *, pinned):
    """Gather selected native CPU experts, upload, losslessly pack, execute.

    Host buffers and device scratch are preallocated. Every timed iteration
    refreshes all selected weights; there is no hot-cache advantage here.
    """
    hidden, _, weights, routes, row_map = fixture
    tokens = hidden.shape[0]
    selected = sorted({route[0] for route in routes})
    index = torch.tensor(selected)
    local = {expert: slot for slot, expert in enumerate(selected)}
    ids = torch.tensor([local[route[0]] for route in routes], dtype=torch.int32)
    oracle.upload(oracle.buffers["input"], hidden[[route[1] for route in routes]].numpy().tobytes())
    oracle.upload(oracle.buffers["ids"], ids.numpy().tobytes())
    oracle.upload(oracle.buffers["row_map"], row_map.numpy().tobytes())
    oracle.upload(oracle.buffers["weights"], weights.numpy().tobytes())
    host_buffers = [torch.empty((len(selected), *value.shape[1:]), dtype=value.dtype) for value in bank["tensors"]]
    runtime, check = oracle.cuda.library, oracle.cuda.check
    runtime.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    runtime.cudaHostRegister.restype = ctypes.c_int
    runtime.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    runtime.cudaHostUnregister.restype = ctypes.c_int
    registered = []
    staging = ctypes.c_void_p()
    pack = oracle.library.sidecard_oracle_pack_cube
    pointers = oracle.bank_pointers[layer]
    names = ("input_q", "input_scales", "gate_up", "hidden", "hidden_q", "hidden_scales", "output")
    arguments = (oracle.buffers["input"], oracle.buffers["ids"], *pointers)
    arguments += tuple(oracle.buffers[name] for name in names)
    samples = []
    try:
        if pinned:
            for value in host_buffers:
                check(runtime.cudaHostRegister(value.data_ptr(), value.numel() * value.element_size(), 0))
                registered.append(value)
        check(runtime.cudaMalloc(ctypes.byref(staging), host_buffers[0].numel()))
        for iteration in range(WARMUPS + SAMPLES):
            start = time.perf_counter()
            for projection, (source, target) in enumerate(zip(bank["tensors"], host_buffers, strict=True)):
                torch.index_select(source, 0, index, out=target)
                destination = staging if projection in (0, 2) else pointers[projection]
                check(runtime.cudaMemcpy(destination, target.data_ptr(), target.numel() * target.element_size(), 1))
                if projection in (0, 2):
                    width = WIDTH if projection == 0 else 2048
                    bits = target.shape[-1] * 8 // width
                    check(pack(staging, pointers[projection], len(selected), target.shape[1], width, bits))
            check(oracle.launch(*arguments, len(routes), WIDTH, 2048, 4))
            check(
                oracle.reduce(
                    oracle.buffers["output"],
                    oracle.buffers["row_map"],
                    oracle.buffers["weights"],
                    oracle.buffers["partial_output"],
                    tokens,
                    WIDTH,
                    len(routes),
                )
            )
            check(runtime.cudaDeviceSynchronize())
            if iteration >= WARMUPS:
                samples.append((time.perf_counter() - start) * 1000)
        actual = (
            torch.frombuffer(bytearray(oracle.read_output("partial_output", tokens * WIDTH * 4)), dtype=torch.float32)
            .reshape(tokens, WIDTH)
            .clone()
        )
    finally:
        if staging.value:
            check(runtime.cudaFree(staging))
        for value in reversed(registered):
            check(runtime.cudaHostUnregister(value.data_ptr()))
    return actual, samples, sum(value.numel() * value.element_size() for value in host_buffers), len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--cold-streaming", action="store_true")
    parser.add_argument("--flush-cuda-cache", action="store_true")
    parser.add_argument("--cache-flush-library", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if args.flush_cuda_cache and (args.backend != "cuda" or args.cold_streaming or args.cache_flush_library is None):
        parser.error("cache flushing requires CUDA resident-only mode and the raw probe library")
    if args.backend == "npu" and os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        parser.error("NPU ownership must be explicitly devices 2,3")
    import torch

    plan = NativeResidentPlan(
        args.checkpoint,
        tuple((layer, expert) for layer in LAYERS for expert in range(FIRST_EXPERT, FIRST_EXPERT + EXPERTS)),
        args.checkpoint_sha256,
    )
    plan.authenticate()
    torch.set_num_threads(1)
    cache_flush = None
    if args.flush_cuda_cache:
        flush_library = ctypes.CDLL(str(args.cache_flush_library))
        cache_flush = flush_library.architecture_flush
        cache_flush.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        cache_flush.restype = ctypes.c_int
    native = None
    if args.backend == "npu":
        import torch_npu

        from tools.glm_perf.fused_moe_profile import frozen_helper

        torch.npu.set_device(0)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        bundle = contained(plan.root, plan.manifest["kernel_bundle"])
        helper, options = frozen_helper(bundle)
        native = helper.NativeFusedMoE(
            bundle,
            namespace=options["namespace"],
            activation_bits=4,
            prepared_weight_layout=True,
            **({"fp16_route_workspace": True} if options.get("fp16_route_workspace") else {}),
        )
    args.comparison_dir.mkdir(exist_ok=True)
    report = {
        "completed": False,
        "backend": args.backend,
        "checkpoint_sha256": args.checkpoint_sha256,
        "resident_experts_per_layer": EXPERTS,
        "local_routes_per_token": LOCAL_ROUTES,
        "activation_bits": 4,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "cuda_cache_flush_bytes": COPY_BYTES if args.flush_cuda_cache else 0,
        "scope": (
            "resident expert pipeline including weighted reduction, excluding host staging/RPC; "
            "NPU native host dispatch included"
        ),
        "cases": [],
        "cold_streaming_cases": [],
    }
    for layer, bank in plan.items():
        oracle = None
        try:
            if args.backend == "cuda":
                oracle = CudaOracle(
                    {layer: bank},
                    {
                        "library": str(args.library),
                        "library_sha256": digest_file(args.library),
                        "cuda_runtime": "/usr/local/cuda/lib64/libcudart.so",
                        "weight_layout": CUDA_NIBBLE_PLANES,
                        "partial_reduce": True,
                    },
                    1024,
                )
            for tokens in (1, 4, 128, 512):
                fixture = route_fixture(tokens, torch)
                # Reuse owned scratch between cases; cache eviction stays outside timing.
                if oracle and args.flush_cuda_cache:
                    for name in ("cache-flush-source", "cache-flush-target"):
                        pointer = oracle.buffers.pop(name, None)
                        if pointer is not None:
                            oracle.cuda.check(oracle.cuda.library.cudaFree(pointer))
                            oracle.allocated_bytes -= COPY_BYTES
                actual, device, wall = (
                    cuda_measure(oracle, layer, fixture, torch, cache_flush=cache_flush)
                    if oracle
                    else npu_measure(native, bank, fixture, torch)
                )
                comparison = args.comparison_dir / f"layer-{layer}-tokens-{tokens}.pt"
                validation = {"finite": bool(torch.isfinite(actual).all()), "cross_architecture_passed": False}
                if not validation["finite"]:
                    raise AssertionError("nonfinite expert output")
                if oracle:
                    torch.save(actual, comparison)
                else:
                    expected = torch.load(comparison, weights_only=True)
                    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
                    validation.update(
                        cross_architecture_passed=True, max_abs_error=float((actual - expected).abs().max())
                    )
                result = {
                    "layer": layer,
                    "weight_bits": plan.manifest["layer_bits"][str(layer)],
                    "tokens": tokens,
                    "routes": tokens * LOCAL_ROUTES,
                    "validation": validation,
                    "device": summarize(device),
                    "wall": summarize(wall),
                }
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
            if oracle and args.cold_streaming:
                for tokens in (1, 4, 128, 512):
                    for pinned in (False, True):
                        fixture = route_fixture(tokens, torch)
                        actual, samples, transferred, selected = cuda_cold_measure(
                            oracle, layer, bank, fixture, torch, pinned=pinned
                        )
                        expected = torch.load(
                            args.comparison_dir / f"layer-{layer}-tokens-{tokens}.pt", weights_only=True
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        result = {
                            "layer": layer,
                            "weight_bits": plan.manifest["layer_bits"][str(layer)],
                            "tokens": tokens,
                            "selected_experts": selected,
                            "weight_upload_bytes": transferred,
                            "pinned": pinned,
                            "wall": summarize(samples),
                            "bitwise_equal_to_resident": True,
                            "scope": "CPU weight gather + H2D + native Cube packing + complete expert pipeline",
                        }
                        report["cold_streaming_cases"].append(result)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(result), flush=True)
        finally:
            if oracle:
                oracle.close()
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
