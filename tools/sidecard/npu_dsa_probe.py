# SPDX-License-Identifier: Apache-2.0
"""Coordinated TP2 real-weight gate for partial Ascend DSA capture.

Run on an exclusive assigned-device window with the frozen NPU interpreter and
its existing ASCEND_RT_VISIBLE_DEVICES=2,3 environment. The parent launches both
owners jointly; independent rank reports are never qualification evidence.
Config requires checkpoint/checkpoint_sha256 (native-layout.json), ops_library/
ops_library_sha256, physical_chips [2,3]. Optional max_context (default 8192),
lengths (default dense/sparse threshold and all four pool-tail phases), steps
(default 4), repeats (default 5). All paths must be absolute authenticated files.

This compares real-weight prepare_forward outputs, selected IDs and used cache,
pool/tail bytes bitwise. One eager attention HCCL SUM per step is exercised;
experts, second SUM, full-model states, 128K capacity and service speed are NOT
qualified. Timing is synchronized attention-only wall time outside CPU checks.
--help never imports torch/torch_npu or opens a device. No global configuration,
driver state, OPP installation or hardware reset is changed by this probe.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from .npu_dsa_decode_graph import CAPTURE_SCOPE, DSA_LAYERS, NpuDsaDecodeGraph

DEFAULT_LENGTHS = (128, 2047, 2048, 2049, 2050, 2051, 2052)
PAIR_TIMEOUT_SECONDS = 1800


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(config):
    allowed = {
        "checkpoint",
        "checkpoint_sha256",
        "ops_library",
        "ops_library_sha256",
        "physical_chips",
        "max_context",
        "lengths",
        "steps",
        "repeats",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unknown or invalid DSA probe configuration")
    if config.get("physical_chips") != [2, 3]:
        raise ValueError("DSA probe requires assigned physical chips [2,3]")
    root, library = Path(config["checkpoint"]), Path(config["ops_library"])
    for path, expected in (
        (root / "native-layout.json", config["checkpoint_sha256"]),
        (library, config["ops_library_sha256"]),
    ):
        if not path.is_absolute() or not path.is_file() or sha256(path) != expected:
            raise ValueError("DSA probe input path or SHA256 differs from frozen identity")
    result = dict(config)
    for name, default, lower, upper in (
        ("max_context", 8192, 2056, 131072),
        ("steps", 4, 4, 128),
        ("repeats", 5, 3, 100),
    ):
        value = config.get(name, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
        result[name] = value
    lengths = config.get("lengths", list(DEFAULT_LENGTHS))
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(
            type(n) is not int or not 1 <= n <= result["max_context"] - result["steps"] - result["repeats"] - 2
            for n in lengths
        )
        or len(set(lengths)) != len(lengths)
        or not set(DEFAULT_LENGTHS).issubset(lengths)
    ):
        raise ValueError("lengths must include dense/sparse thresholds, changed positions and all pool-tail phases")
    result["lengths"] = lengths
    return result


def tensor_receipt(value, torch):
    # Explicit gate-only CPU observation, never part of replay timing.
    cpu = value.detach().contiguous().cpu()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(cpu).all()) if cpu.is_floating_point() else True,
    }


def state_receipt(layer, outputs, selected, torch):
    indexer = layer.indexer
    values = {
        **{f"output_{i}": value for i, value in enumerate(outputs)},
        "cache": layer.cache[: layer.length],
        "pools": indexer.pools[: indexer.pool_count],
        "tail_keys": indexer.tail_keys[: indexer.tail_length],
        "tail_gates": indexer.tail_gates[: indexer.tail_length],
    }
    return {
        "length": layer.length,
        "pool_count": indexer.pool_count,
        "tail_length": indexer.tail_length,
        "tensors": {name: tensor_receipt(value, torch) for name, value in values.items()},
        "selected": None if selected is None else tensor_receipt(selected, torch),
    }


def pair_agreement(torch, device):
    def agree(layer_index, ready):
        # Host calls occur in identical layer order, outside graph capture.
        value = torch.tensor([int(ready), layer_index, layer_index * layer_index], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(value)
        total, index_sum, index_square_sum = value.cpu().tolist()
        if (index_sum, index_square_sum) != (2 * layer_index, 2 * layer_index * layer_index):
            raise RuntimeError("TP2 capture owners entered different layer schedules")
        return total == 2

    return agree


def owner_gate(config, output, rank, rendezvous):
    # Process-local lazy imports preserve parent/device isolation.
    import torch
    import torch_npu

    from .layer_math import HIDDEN, STREAMS, TextLayer, validate_config
    from .layer_weights import load_dense

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("frozen NPU environment must expose only assigned physical chips 2,3")
    device = f"npu:{rank}"
    torch.npu.set_device(rank)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(config["ops_library"])
    torch.distributed.init_process_group(
        "hccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=PAIR_TIMEOUT_SECONDS),
    )
    report = {"rank": rank, "physical_chip": 2 + rank, "scope": CAPTURE_SCOPE, "ok": False, "cases": []}
    graphs = []
    try:
        root = Path(config["checkpoint"])
        model = json.loads((root / "config.json").read_text())["text_config"]
        validate_config(model, config["max_context"])
        weights, _ = load_dense(root, DSA_LAYERS, device, torch, tp_rank=rank)
        with torch.inference_mode():
            for index in DSA_LAYERS:
                layer = TextLayer(
                    index,
                    weights[index],
                    model,
                    device,
                    torch,
                    None,
                    config["max_context"],
                    native_sinkhorn=torch.ops._C_ascend.mhc_sinkhorn_310,
                    native_round=torch.ops._C_ascend.mhc_bf16_round_310,
                    tp_rank=rank,
                )
                graph = NpuDsaDecodeGraph(layer, torch, agree=pair_agreement(torch, device))
                graphs.append(graph)
                if not graph.enabled:
                    raise RuntimeError(f"paired capture rejected layer {index}: {graph.failure}")
                report["cases"].extend(layer_cases(layer, graph, config, torch, HIDDEN, STREAMS))
                graph.close()
        report["ok"] = all(row["bitwise_equal"] and row["finite"] for row in report["cases"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        for graph in graphs:
            graph.close()
        torch.distributed.destroy_process_group()
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def layer_cases(layer, graph, config, torch, hidden, streams):
    """Reset/replay identical CPU-seeded inputs; hash only used state, never duplicate device caches."""
    rows = []
    selected = [None]
    select = layer.indexer.select

    def observe(*args):
        value = select(*args)
        selected[0] = value
        return value

    layer.indexer.select = observe
    try:
        for length in config["lengths"]:
            modes = {}
            timings = {}
            for label, operation in (
                ("eager", lambda state: layer.prepare_forward(*state)),
                ("graph", graph.attention),
            ):
                layer.reset()
                generator = torch.Generator().manual_seed(3500 + layer.index)

                def inputs(count, generator=generator):
                    return (
                        torch.randn(count, hidden, generator=generator).half().to(layer.device),
                        torch.randn(count, streams, hidden, generator=generator).to(layer.device),
                        torch.ones(count, streams, 1, device=layer.device),
                        torch.eye(streams, device=layer.device)[None].expand(count, -1, -1).contiguous(),
                    )

                remaining = length
                while remaining:
                    count = min(32, remaining)
                    layer.prepare_forward(*inputs(count))
                    remaining -= count
                records = []
                # Changed one-row inputs/positions and an explicit multi-row fallback.
                for count in (*([1] * config["steps"]), 2):
                    outputs = operation(inputs(count))
                    records.append(state_receipt(layer, outputs, selected[0], torch))
                modes[label] = records
                samples = []
                for _ in range(config["repeats"]):
                    state = inputs(1)
                    torch.npu.synchronize()
                    begin = time.perf_counter()
                    operation(state)
                    torch.npu.synchronize()
                    samples.append((time.perf_counter() - begin) * 1000)
                timings[label] = samples
            rows.append(
                {
                    "layer": layer.index,
                    "prefill_tokens": length,
                    "bitwise_equal": modes["eager"] == modes["graph"],
                    "finite": all(
                        v["finite"] for records in modes.values() for row in records for v in row["tensors"].values()
                    ),
                    "reference": modes["eager"],
                    "candidate": modes["graph"],
                    "attention_synchronized_wall_ms": timings,
                }
            )
    finally:
        layer.indexer.select = select
        layer.reset()
    return rows


def run(config_path, output):
    config = configuration(json.loads(Path(config_path).read_text()))
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    private = destination.parent / (destination.stem + "-owners")
    private.mkdir(mode=0o700)  # Refuse stale reports or a reused rendezvous.
    frozen = private / "config.json"
    frozen.write_text(json.dumps(config, indent=2) + "\n")
    frozen.chmod(0o600)
    rendezvous = (private / "hccl-rendezvous").as_uri()
    processes = []
    handles = []
    try:
        for rank in (0, 1):
            handle = (private / f"rank{rank}.log").open("w")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tools.sidecard.npu_dsa_probe",
                        "--config",
                        str(frozen),
                        "--output",
                        str(private / f"rank{rank}.json"),
                        "--worker-rank",
                        str(rank),
                        "--rendezvous",
                        rendezvous,
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [process.wait() for process in processes]
    finally:
        # No blind termination/reset: ownership remains explicit if launch fails.
        for handle in handles:
            handle.close()
    owners = []
    for rank in (0, 1):
        path = private / f"rank{rank}.json"
        owners.append(
            {"path": str(path), "sha256": sha256(path), "report": json.loads(path.read_text())}
            if path.exists()
            else {"path": str(path), "missing": True}
        )
    report = {
        "schema_version": 1,
        "coordinated_tp2": True,
        "physical_chips": [2, 3],
        "scope": CAPTURE_SCOPE,
        "configuration_sha256": sha256(frozen),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "ops_library_sha256": config["ops_library_sha256"],
        "exit_codes": codes,
        "owners": owners,
        "same_math_comparison": "bitwise",
        "collectives_captured": False,
        "full_attention_captured": False,
        "full_model_qualified": False,
        "context_capacity_qualified": False,
        "service_performance_qualified": False,
        "ok": codes == [0, 0] and all(owner.get("report", {}).get("ok") for owner in owners),
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-rank", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--rendezvous", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_rank is not None:
        report = owner_gate(
            configuration(json.loads(args.config.read_text())), args.output, args.worker_rank, args.rendezvous
        )
    else:
        report = run(args.config, args.output)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
