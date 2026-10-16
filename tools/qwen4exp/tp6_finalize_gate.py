# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only finalizer gate using private tensors in a drained resident.

This module never binds a model method, recaptures graphs, or resets caches.
Numerical rejection is a completed gate, not a failed device transaction.
"""

import hashlib
import inspect
import json
import statistics
import textwrap
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace

import torch

CHUNK_SHA256 = "a27b331cbf44aeed8d02a3387f0d1c17a4bdea0a57757454b070653dcd7954d5"
MIN_FREE_BYTES = 1024**3
NAMESPACE = "qwen_hc_residual_producer_v1"
SYNTHETIC_TOKENS = (1, 13, 65, 256, 1024)


def checksum(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def storage_metadata(value):
    """Read addresses and layout only; never copy KV or recurrent contents."""
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return [(repr(key), storage_metadata(item)) for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [storage_metadata(item) for item in value]
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    raise TypeError(f"unsupported cache metadata: {type(value).__name__}")


def worker_snapshot(worker):
    runner = worker.model_runner
    tiers = getattr(runner, "_prefix_mamba_tiers", {})
    graphs = []
    for wrapper in worker._resident_wrappers():
        entries = wrapper.entries
        graphs.append((id(wrapper), repr(wrapper.graph_pool), [(repr(k), id(v)) for k, v in entries.items()]))
    return dict(
        kv=checksum(storage_metadata(runner.kv_caches)),
        tiers={
            str(key): dict(
                status=tier.cache_status(),
                metadata=checksum(
                    storage_metadata(
                        (
                            tier._resident,
                            tier._device_archive_resident,
                            tier._host,
                            tier.layer_states,
                            tier._device_archive,
                        )
                    )
                ),
            )
            for key, tier in tiers.items()
        },
        graphs=checksum(graphs),
        graph_entries=sum(len(wrapper.entries) for wrapper in worker._resident_wrappers()),
    )


def scheduler_snapshot(engine):
    pool = engine.scheduler.kv_cache_manager.block_pool
    cache = pool.cached_block_hash_to_block._cache
    bindings = [
        (repr(key), sorted(blocks) if isinstance(blocks, dict) else [blocks.block_id]) for key, blocks in cache.items()
    ]
    return dict(
        cached_hashes=len(cache),
        hash_bindings=checksum(sorted(bindings)),
        alias_blocks=len(pool.cached_block_hashes_by_block),
        hashes=checksum(sorted((k, sorted(map(repr, v))) for k, v in pool.cached_block_hashes_by_block.items())),
        blocks=checksum([(b.block_id, b.ref_cnt, repr(b.block_hash)) for b in pool.blocks]),
    )


def projection_only(original):
    """Clone the inspected chunk through projection; leave the live class alone."""
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != CHUNK_SHA256:
        raise ValueError("unqualified live grouped chunk")
    seam = '        if self.grouped_finalize == "torch":\n'
    if source.count(seam) != 1:
        raise ValueError("ambiguous live finalization seam")
    source = source.split(seam)[0] + "        return output, dispatch, group_ends\n"
    namespace = dict(original.__globals__)
    exec(compile(textwrap.dedent(source), "<tp6-private-projection-gate>", "exec"), namespace)
    compiled = namespace[original.__name__]
    return FunctionType(compiled.__code__, original.__globals__, compiled.__name__)


def compare(actual, expected):
    # These diagnostic synchronizations only execute with the engine drained.
    actual, expected = actual.cpu(), expected.cpu()
    difference = (actual - expected).abs()
    return dict(
        bitwise=bool(torch.equal(actual.view(torch.int32), expected.view(torch.int32))),
        unequal=int((actual != expected).sum()),
        elements=actual.numel(),
        max_abs=float(difference.max()),
    )


def measurements(operations):
    samples = {name: [] for name in operations}
    for operation in operations.values():
        operation()
    torch.npu.synchronize()
    for iteration in range(6):
        names = list(operations)
        if iteration % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            operations[name]()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
    return {name: dict(median_ms=statistics.median(rows), samples_ms=rows) for name, rows in samples.items()}


def run_gate(worker, config, resource_source, owner):
    # Lazy imports occur inside each worker; the engine need not import models.
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, finalize_grouped_routes, route_topk

    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("worker has pending execution")
    modules = [
        module
        for wrapper in worker._resident_wrappers()
        for module in wrapper.runnable.modules()
        if isinstance(module, W4SparseMoE)
    ]
    if not modules or any(module.num_local_experts not in (85, 86) for module in modules):
        raise ValueError("expected TP6 W4 modules")
    project = projection_only(W4SparseMoE._forward_grouped_chunk)
    binary, bridge = (Path(config[key]).resolve(strict=True) for key in ("binary", "bridge"))
    for key, path in (("binary", binary), ("bridge", bridge)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[key + "_sha256"]:
            raise ValueError(f"changed {key}")
    if torch.npu.mem_get_info()[0] < MIN_FREE_BYTES:
        raise RuntimeError("insufficient private-tensor headroom")
    owner["before"] = worker_snapshot(worker)
    owner["namespace"] = namespace = {}
    exec(compile(resource_source, "<tp6-private-finalizer-resource>", "exec"), namespace)
    # Bridge is already loaded by the retained HC producer. Never dlopen a new
    # operator namespace while captured graphs are live.
    owner["kernel"] = getattr(torch.classes, NAMESPACE).Kernel(str(binary), "qwen_tp6_route_finalize_v1")
    owner["resource"] = resource = namespace["Finalize"](owner["kernel"], getattr(torch.ops, NAMESPACE).launch)
    device = worker.model_runner.device
    generator = torch.Generator(device="cpu").manual_seed(61016 + worker.rank)
    rows = []

    def check(label, routed, dispatch, weights, ends, timing=False):
        operations = dict(
            torch=lambda: finalize_grouped_routes(routed, dispatch, weights, torch.float32, "torch"),
            retained=lambda: finalize_grouped_routes(
                routed, dispatch, weights, torch.float32, modules[0].grouped_finalize
            ),
            candidate=lambda: resource(routed, dispatch.inverse_order, dispatch.route_weights, ends),
        )
        actual = operations["candidate"]()
        row = dict(
            label=label,
            torch=compare(actual, operations["torch"]()),
            retained=compare(actual, operations["retained"]()),
        )
        if timing:
            row["timing"] = measurements(operations)
        rows.append(row)

    with torch.inference_mode():
        for tokens in SYNTHETIC_TOKENS:
            count = tokens * 10
            inverse_cpu = torch.randperm(count, generator=generator)
            inverse = inverse_cpu.to(device)
            weights = torch.rand((tokens, 10), generator=generator).to(device)
            weights /= weights.sum(1, keepdim=True)
            dispatch = SimpleNamespace(
                inverse_order=inverse, order=torch.argsort(inverse_cpu).to(device), route_weights=weights.reshape(-1, 1)
            )
            for active in (0, count // 6, count):
                routed_cpu = torch.randn((count, 2560), generator=generator, dtype=torch.float16)
                routed_cpu[active:] = 0
                routed = routed_cpu.to(device)
                ends = torch.full((modules[0].num_local_experts,), active, dtype=torch.int64, device=device)
                check(
                    f"synthetic/{tokens}/{active}",
                    routed,
                    dispatch,
                    weights,
                    ends,
                    timing=tokens == 256 and active == count // 6,
                )
                # The native path must not read inactive peer rows.
                routed[active:] = float("nan")
                poisoned = resource(routed, inverse, dispatch.route_weights, ends)
                clean = routed_cpu.to(device)
                reference = finalize_grouped_routes(clean, dispatch, weights, torch.float32, "torch")
                rows[-1]["poisoned_peers"] = compare(poisoned, reference)
                del routed, poisoned, clean, reference
        for index in (0, len(modules) // 2, len(modules) - 1):
            module = modules[index]
            inputs = torch.randn((65, 2560), generator=generator, dtype=torch.float16).to(device)
            weights, ids = route_topk(
                torch.nn.functional.linear(inputs, module.gate),
                module.top_k,
                renormalize=module.renormalize,
                routed_scaling_factor=module.routed_scaling_factor,
            )
            routed, dispatch, ends = project(module, inputs, weights, ids)
            check(f"real_weight_component/{index}/65", routed, dispatch, weights, ends)
        torch.npu.synchronize()
    owner["after"] = worker_snapshot(worker)
    owner["rows"] = rows
    return dict(
        rank=worker.rank,
        passed=owner["before"] == owner["after"],
        before=owner["before"],
        after=owner["after"],
        rows=rows,
        retained_policy=modules[0].grouped_finalize,
        torch_exact=all(
            row["torch"]["bitwise"] and row.get("poisoned_peers", {"bitwise": True})["bitwise"] for row in rows
        ),
        retained_exact=all(row["retained"]["bitwise"] for row in rows),
        installed=False,
    )


def resource_source():
    """Bundle existing resource definitions without importing absent build tools."""
    import ast

    source = Path(__file__).with_name("tp6_fusions.py").read_text()
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in (
            "tools.qwen4exp.build_tp6_fusions",
            "tools.qwen4exp.tp6_fusion_sources",
        ):
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("load_resources", "GatePack"):
            continue
        selected.append(ast.get_source_segment(source, node))
    return "\n\n".join(selected) + "\n"


ENGINE_SOURCE = """
def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    if not hasattr(worker, "_tp6_finalizer_gate_owners"):
        worker._tp6_finalizer_gate_owners = []
    owner = {}
    worker._tp6_finalizer_gate_owners.append(owner)
    try:
        owner["gate_namespace"] = namespace = {}
        exec(compile(GATE_SOURCE, "<tp6-finalizer-gate>", "exec"), namespace)
        return namespace["run_gate"](worker, CONFIG, RESOURCE_SOURCE, owner)
    except Exception as error:
        owner["failure"] = error  # Retain traceback/operands after uncertain work.
        return dict(rank=worker.rank, passed=False, error=repr(error))

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError("drain engine without clearing caches before gate")
        if not hasattr(self, "result"):
            namespace = {}
            exec(compile(GATE_SOURCE, "<tp6-finalizer-gate>", "exec"), namespace)
            before = namespace["scheduler_snapshot"](engine)
            workers = engine.model_executor.collective_rpc(operation)
            after = namespace["scheduler_snapshot"](engine)
            expected = engine.vllm_config.parallel_config.tensor_parallel_size
            passed = len(workers) == expected == 6 and {r["rank"] for r in workers} == set(range(expected))
            self.result = dict(passed=passed and all(r["passed"] for r in workers) and before == after,
                before=before, after=after, workers=workers, installed=False)
        return self.result
    def apply(self, engine):
        raise RuntimeError("diagnostic gate cannot install a candidate")
def prepare(engine):
    return Candidate()
"""


def engine_source(config):
    gate = Path(__file__).read_text()
    return f"CONFIG = {config!r}\nRESOURCE_SOURCE = {resource_source()!r}\nGATE_SOURCE = {gate!r}\n" + ENGINE_SOURCE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(engine_source(json.loads(args.config.read_text())))
