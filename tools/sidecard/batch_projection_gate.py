# SPDX-License-Identifier: Apache-2.0
"""Gate larger grouped CUDA batches against the deployed 32-token boundary."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from .device_backend import (
    CUDA_COMPACT_PLANES,
    CUDA_NIBBLE_PLANES,
    CUDA_ROW_MAJOR,
    CUDA_WORD_PLANES,
    CudaOracle,
    owned_keys,
)
from .native_expert_probe import digest_file
from .one_card_partition import PARTITIONS, SEED_PARTITION, one_card_partition
from .ordered_tensor import combine_compact
from .reference import ExpertReply, ExpertRequest, Route
from .resident import CudaResidentBackend, NativeResidentPlan
from .service_adapter import service_placement
from .tensor_wire import TensorPacket, decode_tensor, encode_tensor
from .wire import PacketLimits, WireIdentity

LAYERS = (3, 8, 33)
BATCH_TOKENS = (128, 512, 640)
ROUTES_PER_TOKEN = (1, 2, 8)
WIDTH = 4096
REFERENCE_TOKENS = 32
MAX_ROUTES = 5120
SHORT_ROUTES = (1, 4, 8, 16, 31, 32, 33)


def short_projection_cases(oracle, layer, generator):
    """Paired decode/tile-boundary checks against the original packed kernel."""
    cases = []
    grouped, split = oracle.grouped_launch, oracle.split_launch
    try:
        for rows in SHORT_ROUTES:
            hidden = (generator.standard_normal((rows, WIDTH)) * 0.1).astype("<f2").tobytes()
            experts = tuple(sorted(generator.integers(0, 144, size=rows).tolist()))
            for bits in (4, 8):
                samples = {"candidate": [], "scalar": []}
                expected = None
                for iteration in range(3):
                    labels = ("candidate", "scalar") if iteration % 2 else ("scalar", "candidate")
                    for label in labels:
                        oracle.grouped_launch = grouped if label == "candidate" else None
                        oracle.split_launch = split if label == "candidate" else None
                        started = time.perf_counter()
                        actual = oracle.compute_bytes(layer, hidden, experts, bits)
                        samples[label].append((time.perf_counter() - started) * 1000)
                        if expected is None:
                            expected = actual
                        elif actual != expected:
                            raise AssertionError("short-batch CUDA projections differ bitwise")
                cases.append(
                    dict(
                        layer=layer,
                        routes=rows,
                        activation_bits=bits,
                        bitwise_equal=True,
                        candidate_ms=statistics.median(samples["candidate"]),
                        scalar_ms=statistics.median(samples["scalar"]),
                    )
                )
    finally:
        oracle.grouped_launch, oracle.split_launch = grouped, split
    return cases


def round_trip(packet, identity, limits):
    return decode_tensor(
        encode_tensor(packet, identity, limits),
        identity,
        limits,
        reply_fp16=packet.reply_fp16,
        partial_reduce=packet.partial_reduce,
    )


def run(
    checkpoint,
    checkpoint_sha,
    reference,
    candidate,
    output,
    *,
    weight_layout=CUDA_ROW_MAJOR,
    reply_fp16=False,
    partial_reduce=False,
    bank_cache=None,
    one_card=False,
    experimental_projection=False,
    cuda_partition=SEED_PARTITION,
):
    # CUDA-only process; CPU Torch unpacks authenticated native banks.
    import numpy as np
    import torch

    torch.set_num_threads(8)
    partition = one_card_partition(cuda_partition)
    if partition.name != SEED_PARTITION and (
        not one_card or weight_layout != CUDA_COMPACT_PLANES or bank_cache is not None
    ):
        raise ValueError("precision gate requires uncached one-card compact CUDA")
    cuda_ids = {layer: partition.cuda_ids(layer) if one_card else tuple(range(72)) for layer in LAYERS}
    cuda_experts = 144 if one_card and partition.name == SEED_PARTITION else 72 if not one_card else None
    plan = NativeResidentPlan(
        checkpoint, tuple((layer, expert) for layer in LAYERS for expert in cuda_ids[layer]), checkpoint_sha
    )
    if bank_cache is None:
        plan.authenticate()
        cached = None
    else:
        from .cuda_bank_cache import MANIFEST_NAME, CachedCudaBanks

        cached = CachedCudaBanks(
            bank_cache,
            digest_file(bank_cache / MANIFEST_NAME),
            checkpoint_sha,
            owned_keys(service_placement(), "cuda:0"),
            weight_layout,
        )
    identity = WireIdentity.for_placement(
        service_placement(one_card=one_card, cuda_partition=partition.name), checkpoint_sha
    )
    old_limits, new_limits = PacketLimits(WIDTH, 32, 256), PacketLimits(WIDTH, 640, MAX_ROUTES)
    oracles = {}
    report = dict(
        passed=False,
        end_to_end_speedup_validated=False,
        cases=[],
        checkpoint_sha256=checkpoint_sha,
        candidate_weight_layout=weight_layout,
        candidate_reply_fp16=reply_fp16,
        candidate_partial_reduce=partial_reduce,
        cuda_experts_per_layer=cuda_experts,
        candidate_experimental_projection=experimental_projection,
        candidate_cuda_partition=partition.name,
        cuda_expert_ids_by_layer={str(layer): list(ids) for layer, ids in cuda_ids.items()},
    )
    try:
        for label, library, routes in (("reference", reference, 256), ("candidate", candidate, MAX_ROUTES)):
            specification = dict(
                library=str(library),
                library_sha256=digest_file(library),
                cuda_runtime="/usr/local/cuda/lib64/libcudart.so",
                cuda_memory_budget_bytes=24 * 1024**3,
            )
            if label == "candidate":
                specification["weight_layout"] = weight_layout
                specification["partial_reduce"] = partial_reduce
                specification["experimental_projection"] = experimental_projection
            bank_layout = specification.get("weight_layout", CUDA_ROW_MAJOR)
            banks = (
                plan
                if cached is None
                else SimpleNamespace(items=lambda layout=bank_layout: cached.selected_items(LAYERS, layout=layout))
            )
            oracles[label] = CudaOracle(banks, specification, routes)
            report[label + "_sha256"] = specification["library_sha256"]
        backend = CudaResidentBackend.__new__(CudaResidentBackend)
        backend.cuda, backend.plan = oracles["candidate"], plan
        backend.specification = {"reply_fp16": reply_fp16, "partial_reduce": partial_reduce}
        backend.metrics = dict(requests=0, routes=0, compute_total_ms=0.0, compute_max_ms=0.0)
        generator = np.random.default_rng(20261011)
        for layer in LAYERS:
            expert_index = {expert: index for index, expert in enumerate(cuda_ids[layer])}
            if getattr(oracles["candidate"], "split_launch", None) is not None:
                report.setdefault("short_cases", []).extend(
                    short_projection_cases(oracles["candidate"], layer, generator)
                )
            for count in BATCH_TOKENS:
                hidden = (generator.standard_normal((count, WIDTH)) * 0.1).astype("<f2")
                for top_k in ROUTES_PER_TOKEN:
                    selected = [generator.choice(cuda_ids[layer], top_k, replace=False).tolist() for _ in range(count)]
                    route_weights = generator.random((count, top_k), dtype=np.float32)
                    started = time.perf_counter()
                    pieces = []
                    for start in range(0, count, REFERENCE_TOKENS):
                        stop = min(count, start + REFERENCE_TOKENS)
                        routes = tuple(
                            Route(token - start, slot, expert)
                            for token in range(start, stop)
                            for slot, expert in enumerate(selected[token])
                        )
                        control = ExpertRequest("batch-gate", layer, "cuda:0", routes, tuple(range(stop - start)), ())
                        packet = round_trip(TensorPacket(control, hidden[start:stop].tobytes()), identity, old_limits)
                        inputs = np.frombuffer(packet.tensor, dtype="<f2").reshape(stop - start, WIDTH)
                        raw = oracles["reference"].compute_bytes(
                            layer,
                            inputs[[r.token for r in routes]].tobytes(),
                            tuple(expert_index[r.expert] for r in routes),
                            4,
                        )
                        reply = ExpertReply(control.execution_id, layer, control.owner, routes, ())
                        encoded = np.frombuffer(raw, dtype="<f2").astype("<f4").tobytes()
                        pieces.append(round_trip(TensorPacket(reply, encoded), identity, old_limits).tensor)
                    reference_ms = (time.perf_counter() - started) * 1000
                    expected = b"".join(pieces)
                    routes = tuple(
                        Route(token, slot, expert)
                        for token in range(count)
                        for slot, expert in enumerate(selected[token])
                    )
                    control = ExpertRequest("batch-gate", layer, "cuda:0", routes, tuple(range(count)), ())
                    if partial_reduce:
                        padded_weights = np.zeros((count, 8), dtype=np.float32)
                        padded_weights[:, :top_k] = route_weights
                        expected = combine_compact(routes, expected, padded_weights, WIDTH).tobytes()
                    started = time.perf_counter()
                    packet = round_trip(
                        TensorPacket(
                            control,
                            hidden.tobytes(),
                            partial_reduce=partial_reduce,
                            route_weights=route_weights.tobytes() if partial_reduce else b"",
                        ),
                        identity,
                        new_limits,
                    )
                    result = round_trip(backend.compute_tensor(packet), identity, new_limits)
                    elapsed = (time.perf_counter() - started) * 1000
                    actual = (
                        np.frombuffer(result.tensor, dtype="<f2").astype("<f4").tobytes()
                        if reply_fp16
                        else result.tensor
                    )
                    if actual != expected:
                        raise AssertionError("grouped batch expert output differs bitwise")
                    case = dict(
                        layer=layer,
                        tokens=count,
                        routes=len(routes),
                        bitwise_equal=True,
                        reference_ms=reference_ms,
                        candidate_ms=elapsed,
                        speedup=reference_ms / elapsed,
                        reply_bytes=len(result.tensor),
                    )
                    grouped_launch = getattr(oracles["candidate"], "grouped_launch", None)
                    if grouped_launch is not None:
                        # Compare the two paths with the same resident banks,
                        # packet, arithmetic flags, and transfers. Alternate
                        # their order to avoid attributing warm-up to WMMA.
                        samples = {"grouped": [], "scalar": []}
                        try:
                            for iteration in range(3):
                                labels = ("scalar", "grouped") if iteration % 2 else ("grouped", "scalar")
                                for label in labels:
                                    oracles["candidate"].grouped_launch = grouped_launch if label == "grouped" else None
                                    started = time.perf_counter()
                                    comparison = backend.compute_tensor(packet)
                                    samples[label].append((time.perf_counter() - started) * 1000)
                                    if comparison.tensor != result.tensor:
                                        raise AssertionError("Tensor Core and scalar resident replies differ bitwise")
                        finally:
                            oracles["candidate"].grouped_launch = grouped_launch
                        case.update(
                            grouped_ms=statistics.median(samples["grouped"]),
                            scalar_ms=statistics.median(samples["scalar"]),
                            grouped_speedup=statistics.median(samples["scalar"])
                            / statistics.median(samples["grouped"]),
                            grouped_scalar_bitwise_equal=True,
                        )
                    report["cases"].append(case)
                    Path(output).write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(case), flush=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        for oracle in reversed(tuple(oracles.values())):
            oracle.close()
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-layout",
        choices=(CUDA_ROW_MAJOR, CUDA_WORD_PLANES, CUDA_NIBBLE_PLANES, CUDA_COMPACT_PLANES),
        default=CUDA_ROW_MAJOR,
    )
    parser.add_argument("--one-card", action="store_true")
    parser.add_argument("--cuda-partition", choices=PARTITIONS, default=SEED_PARTITION)
    parser.add_argument("--experimental-projection", action="store_true")
    parser.add_argument("--reply-fp16", action="store_true")
    parser.add_argument("--partial-reduce", action="store_true")
    parser.add_argument("--bank-cache", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.checkpoint,
                args.checkpoint_sha256,
                args.reference,
                args.candidate,
                args.output,
                weight_layout=args.weight_layout,
                reply_fp16=args.reply_fp16,
                partial_reduce=args.partial_reduce,
                bank_cache=args.bank_cache,
                one_card=args.one_card,
                experimental_projection=args.experimental_projection,
                cuda_partition=args.cuda_partition,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
