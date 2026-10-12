# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight RTX hot / Ascend cold expert overlap experiment.

One W4 layer and eight experts are resident on each device. Synthetic inputs
exercise clipping. Persistent pinned buffers send only FP16 inputs, int32
expert IDs and original FP32 route weights; Ascend returns a weighted FP32
partial. No attention, mHC or HCCL runs here. This is not a deployed model.
"""

import argparse
import importlib
import json
import os
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

from .device_backend import CudaOracle
from .layer_actor import guard_parent
from .layer_weights import TOP_K, WIDTH, CudaExperts, NpuExperts, PlanBanks, private_native_banks
from .native_expert_probe import contained, digest_file
from .resident import NativeResidentPlan

LAYER = 18
READY = b"COLD_EXPERT_READY\n"
CONTROL = struct.Struct("!II")
TIMING = struct.Struct("!d")
MAX_ROWS = 32
INPUT_ROW_BYTES = WIDTH * 2 + TOP_K * 8
OUTPUT_ROW_BYTES = WIDTH * 4
WARMUPS = 3
SAMPLES = 12


def receive(connection, target):
    position = 0
    while position < len(target):
        count = connection.recv_into(target[position:])
        if not count:
            raise EOFError("private cold-expert peer closed")
        position += count


def selected_plan(spec):
    root = Path(spec["checkpoint"])
    plan = NativeResidentPlan(root, tuple((LAYER, expert) for expert in range(TOP_K)), spec["checkpoint_sha256"])
    plan.authenticate()
    return plan


def split_route_weights(weights, cold):
    """Preserve the original normalization when splitting selected slots."""
    if type(cold) is not int or not 0 <= cold <= TOP_K or weights.shape[-1] != TOP_K:
        raise ValueError("cold partition requires zero to eight selected slots")
    cold_weights, hot_weights = weights.clone(), weights.clone()
    cold_weights[:, cold:] = 0
    hot_weights[:, :cold] = 0
    return cold_weights, hot_weights


def validate_header(rows, cold):
    if type(rows) is not int or type(cold) is not int or not 1 <= rows <= MAX_ROWS or not 1 <= cold <= TOP_K:
        raise ValueError("cold probe packet exceeds its fixed geometry")


def worker(config, fd, device):
    guard_parent(os.getppid())
    import torch
    import torch_npu

    from tools.glm_perf.fused_moe_profile import frozen_helper

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("cold probe requires only the assigned sidecard")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    spec = json.loads(Path(config).read_bytes())
    plan = selected_plan(spec)
    bundle = contained(plan.root, plan.manifest["kernel_bundle"])
    helper, options = frozen_helper(bundle)
    override_path = Path(spec["native_gate_override"])
    override = json.loads(override_path.read_bytes())
    if (
        override["swiglu_limit"] != 10.0
        or override["original_provenance_sha256"] != plan.manifest["kernel_provenance_sha256"]
    ):
        raise ValueError("cold expert clipping provenance differs")
    for name, checksum in override["native_gate_files"].items():
        if digest_file(override_path.parent / name) != checksum:
            raise ValueError("cold expert native override failed authentication")

    def factory(path, entry):
        name = Path(path).name
        if name in override["native_gate_files"]:
            path = str(override_path.parent / name)
        return getattr(torch.classes, options["namespace"]).Kernel(path, entry)

    actor = object.__new__(NpuExperts)
    actor.torch, actor.rank, actor.reduce_results = torch, 0, False
    actor.stable_rows = helper.FUSED_REDUCTION_TOKENS + 1
    actor.native = helper.NativeFusedMoE(
        bundle,
        namespace=options["namespace"],
        activation_bits=4,
        kernel_factory=factory,
        prepared_weight_layout=True,
        fp16_route_workspace=True,
    )
    bank = next(private_native_banks(plan, torch))[1]
    actor.banks = {LAYER: tuple(value.to(f"npu:{device}") for value in bank["tensors"])}
    host_in = torch.empty(MAX_ROWS * INPUT_ROW_BYTES, dtype=torch.uint8, pin_memory=True)
    host_out = torch.empty(MAX_ROWS * OUTPUT_ROW_BYTES, dtype=torch.uint8, pin_memory=True)
    input_bytes, output_bytes = memoryview(host_in.numpy()), memoryview(host_out.numpy())
    with socket.socket(fileno=fd) as connection, torch.inference_mode():
        connection.settimeout(300)
        connection.sendall(READY)
        while True:
            header = bytearray(CONTROL.size)
            try:
                receive(connection, memoryview(header))
            except EOFError:
                break
            rows, cold = CONTROL.unpack(header)
            validate_header(rows, cold)
            receive(connection, input_bytes[: rows * INPUT_ROW_BYTES])
            hidden_size = rows * WIDTH * 2
            id_size = rows * TOP_K * 4
            x = host_in[:hidden_size].view(torch.float16).reshape(rows, WIDTH).to(f"npu:{device}")
            ids = host_in[hidden_size : hidden_size + id_size].view(torch.int32).reshape(rows, TOP_K)
            weights = host_in[hidden_size + id_size : rows * INPUT_ROW_BYTES].view(torch.float32).reshape(rows, TOP_K)
            if not torch.equal(ids, torch.arange(TOP_K).expand(rows, -1)) or bool((weights[:, cold:] != 0).any()):
                raise ValueError("probe peer changed fixed selected routes")
            ids, weights = ids.to(x.device), weights.to(x.device)
            torch.npu.synchronize()
            begin = time.perf_counter()
            partial = actor(LAYER, x, ids, weights)
            torch.npu.synchronize()
            native_ms = (time.perf_counter() - begin) * 1000
            host_out[: rows * OUTPUT_ROW_BYTES].view(torch.float32).reshape(rows, WIDTH).copy_(partial)
            torch.npu.synchronize()
            connection.sendall(output_bytes[: rows * OUTPUT_ROW_BYTES])
            connection.sendall(TIMING.pack(native_ms))


def run(config, output, device):
    """Authenticate resident banks, overlap exact partials, and gate each merge.

    The independent CPU quantizer establishes the expected complete eight-slot
    output before any timings. Persistent buffers and original route weights
    then isolate the selected cold-count cost without moving model weights.
    """
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    spec = json.loads(Path(config).read_bytes())
    plan = selected_plan(spec)
    bundle = contained(plan.root, plan.manifest["kernel_bundle"])
    provenance = json.loads((bundle / "provenance.json").read_bytes())
    package = provenance["_build"]["helper_package"]
    for name, checksum in provenance["_helpers"].items():
        if digest_file(bundle / package / name) != checksum:
            raise ValueError("cold expert CPU reference authentication failed")
    sys.path.insert(0, str(bundle))
    projection = importlib.import_module(package + ".fused_moe_probe").projection_reference
    unpack = importlib.import_module(package + ".fused_weight_layout").unpack_cube
    bank = next(plan.items())[1]
    gate, gs, down, ds = bank["tensors"]
    signed_gate, signed_down = unpack(gate, WIDTH), unpack(down, 2048)
    actor = object.__new__(CudaExperts)
    actor.torch = torch
    actor.oracle = CudaOracle(PlanBanks(plan), spec["cuda_backend"], TOP_K * MAX_ROWS)
    actor.configure_stream(True)
    parent, child = socket.socketpair()
    parent.settimeout(300)
    env = dict(os.environ)
    env.update(json.loads(Path(spec["npu_environment"]).read_bytes()))
    env["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + ":" + env.get("PYTHONPATH", "")
    output = Path(output)
    report = dict(
        passed=False,
        layer=LAYER,
        physical_npu=2 + device,
        real_checkpoint_weights=True,
        synthetic_activations=True,
        complete_model_speed_validated=False,
        gpu_hot_route_compaction=False,
        hccl_collectives=False,
        input_row_bytes=INPUT_ROW_BYTES,
        output_row_bytes=OUTPUT_ROW_BYTES,
        samples=SAMPLES,
        cases=[],
    )
    process = None
    try:
        with output.with_suffix(".log").open("x") as log:
            command = [
                spec["npu_python"],
                "-m",
                "tools.sidecard.cold_expert_probe",
                "--config",
                str(config),
                "--worker-fd",
                str(child.fileno()),
                "--device",
                str(device),
            ]
            process = subprocess.Popen(command, env=env, pass_fds=(child.fileno(),), stdout=log, stderr=log)
        child.close()
        ready = bytearray(len(READY))
        receive(parent, memoryview(ready))
        if ready != READY:
            raise RuntimeError("cold expert worker readiness differs")
        with torch.inference_mode():
            for rows in (1, MAX_ROWS):
                torch.manual_seed(39181 + rows)
                hidden = (torch.randn(rows, WIDTH) * 4).half()
                ids = torch.arange(TOP_K, dtype=torch.int32).expand(rows, -1).contiguous()
                weights = (torch.arange(1, TOP_K + 1).float() / 36 * 2.5).expand(rows, -1).contiguous()
                expected = torch.zeros(rows, WIDTH)
                clipping = 0
                for expert in range(TOP_K):
                    gu = projection(hidden, signed_gate[expert], gs[expert].float(), 4)
                    g, u = gu.float().chunk(2, -1)
                    clipping += int(((g > 10) | (u.abs() > 10)).sum())
                    activation = (torch.nn.functional.silu(g.clamp(max=10)) * u.clamp(-10, 10)).half()
                    expected += (
                        projection(activation, signed_down[expert], ds[expert].float(), 4).float()
                        * weights[:, expert, None]
                    )
                x, gpu_ids = hidden.cuda(), ids.cuda()
                host_hidden = torch.empty_like(hidden, pin_memory=True)
                host_partial = torch.empty((rows, WIDTH), dtype=torch.float32, pin_memory=True)
                gpu_partial = torch.empty_like(host_partial, device="cuda")
                for cold in (0, 1, 2, 4, TOP_K):
                    cold_weights, hot_weights = split_route_weights(weights, cold)
                    hot_weights = hot_weights.cuda()
                    control = CONTROL.pack(rows, cold)
                    tail = ids.numpy().tobytes() + cold_weights.numpy().tobytes()
                    wall, native = [], []
                    reference = None
                    for iteration in range(WARMUPS + SAMPLES):
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        if cold:
                            host_hidden.copy_(x, non_blocking=True)
                            torch.cuda.synchronize()
                            parent.sendall(control + host_hidden.numpy().tobytes() + tail)
                        hot = actor(LAYER, x, gpu_ids, hot_weights) if cold < TOP_K else torch.zeros_like(gpu_partial)
                        native_ms = 0
                        if cold:
                            receive(parent, memoryview(host_partial.numpy()).cast("B"))
                            elapsed = bytearray(TIMING.size)
                            receive(parent, memoryview(elapsed))
                            native_ms = TIMING.unpack(elapsed)[0]
                            gpu_partial.copy_(host_partial, non_blocking=True)
                            actual = hot + gpu_partial
                        else:
                            actual = hot
                        torch.cuda.synchronize()
                        duration = (time.perf_counter() - begin) * 1000
                        current = actual.cpu()
                        torch.testing.assert_close(current, expected, rtol=0.002, atol=0.002)
                        if reference is None:
                            reference = current
                        elif not torch.equal(current, reference):
                            raise AssertionError("hot/cold partial merge is nondeterministic")
                        if iteration >= WARMUPS:
                            wall.append(duration)
                            native.append(native_ms)
                    report["cases"].append(
                        dict(
                            rows=rows,
                            cold_experts=cold,
                            hot_experts=TOP_K - cold,
                            clipping_activated_elements=clipping,
                            median_ms=statistics.median(wall),
                            native_median_ms=statistics.median(native),
                            samples_ms=wall,
                            native_samples_ms=native,
                            repeat_bitwise_equal=True,
                            max_abs_error=float((reference - expected).abs().max()),
                        )
                    )
        report["passed"] = True
        return report
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        parent.close()
        child.close()
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        actor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-fd", type=int)
    parser.add_argument("--device", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.worker_fd is not None:
        worker(args.config, args.worker_fd, args.device)
    else:
        if args.output is None:
            parser.error("probe requires --output")
        run(args.config, args.output, args.device)


if __name__ == "__main__":
    main()
