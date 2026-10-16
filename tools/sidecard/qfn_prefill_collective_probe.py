# SPDX-License-Identifier: Apache-2.0
"""Offline-prepared resident gate for the observed large QFN prefill SUMs.

Run only after controller-authorized pause(wait) and completed request/device
drain. A pause(keep) with suspended requests is NOT sufficient. This module
never resumes inference, creates processes/groups, resets devices, or changes
the serving collective. Each geometry gets an independent eager private owner.
Failed private owners and tensors remain strongly retained on the worker.
"""

import hashlib
import json
import secrets
import time
from pathlib import Path

from tools.sidecard.npu_collective_schedule import BinomialSchedule
from tools.sidecard.qfn_allocator_diagnostic import P2P_MEMORY_CATEGORIES, read_p2p_pools

HIDDEN_SIZE = 2560
WORKSPACE_BYTES = 1 << 20
MAX_ELEMENTS = 16 * 1024 * 1024
WARMUP_CALLS = 2
TIMING_PAIRS = 5
TIMING_BATCH = 3
CORRECTNESS_CASES = 2
RESERVE_BYTES = 64 << 20
SDK_ALIGNMENT_BYTES = 32
SDK_PADDING_BYTES = 32
# Deltas from native-timeline-v1/receipt.json, identical on all four TP ranks.
PREFILL_CASES = (
    (65, "float16", 51),
    (65, "float32", 49),
    (640, "float16", 51),
    (640, "float32", 49),
    (2048, "float16", 153),
    (2048, "float32", 147),
)
MODES = ("custom_out_of_place", "hccl_model_in_place", "hccl_out_of_place")


def memory_budget(elements, dtype, peers=2, batch=TIMING_BATCH):
    """SDK allocations plus bounded source/results; no graph allocations."""
    if dtype not in ("float16", "float32") or not 0 < elements <= MAX_ELEMENTS:
        raise ValueError("unsupported native tensor geometry")
    if type(peers) is not int or peers < 0 or type(batch) is not int or batch < 1:
        raise ValueError("invalid owner degree or timing batch")
    payload = elements * (2 if dtype == "float16" else 4)
    owner = (2 + 2 * peers) * payload + WORKSPACE_BYTES
    tensors = (1 + max(batch, 2)) * payload
    return dict(
        payload_bytes=payload,
        owner_bytes=owner,
        tensor_bytes=tensors,
        required_free_bytes=owner + tensors + RESERVE_BYTES,
    )


def private_config(live_config, elements, dtype, output_root):
    memory_budget(elements, dtype)
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("private receipts require an absolute output root")
    result = json.loads(json.dumps(live_config))
    result.update(
        geometries=[dict(dtype=dtype, elements=elements)],
        execution_mode="eager",
        graph_replay_tracking=False,
        output_root=str(root),
    )
    return result


def require_p2p_capacity(pools, elements, dtype, *, peers=2):
    """Reject known-insufficient policy-3 pools; passing cannot prove allocation.

    The installed SDK specifies 32-byte alignment plus 32-byte padding. Physical
    page rounding, fragmentation, handles and concurrent consumers can require
    more. Only P2P huge/normal pools count toward these policy-3 owner buffers.
    """
    budget = memory_budget(elements, dtype, peers=peers)
    # Two work buffers plus two inboxes per schedule peer.
    payload_buffers = (budget["owner_bytes"] - WORKSPACE_BYTES) // budget["payload_bytes"]
    sizes = [budget["payload_bytes"]] * payload_buffers + [WORKSPACE_BYTES]
    padded = [
        (size + SDK_ALIGNMENT_BYTES - 1) // SDK_ALIGNMENT_BYTES * SDK_ALIGNMENT_BYTES + SDK_PADDING_BYTES
        for size in sizes
    ]
    expected = dict(P2P_MEMORY_CATEGORIES)
    if len(pools) != len(expected) or {item.get("attribute") for item in pools} != set(expected):
        raise RuntimeError("P2P capacity requires exactly the supported huge and normal pool snapshots")
    for item in pools:
        free, total = item.get("free_bytes"), item.get("total_bytes")
        if item.get("status") != 0 or type(free) is not int or type(total) is not int or not 0 <= free <= total:
            raise RuntimeError("P2P pool capacity is unavailable; aggregate free memory is insufficient evidence")
    free = [item["free_bytes"] for item in pools]
    required = sum(padded)
    if sum(free) < required:
        raise RuntimeError(f"insufficient P2P pool capacity: {sum(free)} free, at least {required} bytes required")
    if max(free) < max(padded):
        raise RuntimeError("no individual P2P pool can fit the largest required allocation")
    return dict(
        minimum_p2p_bytes=required,
        largest_p2p_allocation_bytes=max(padded),
        p2p_free_bytes=sum(free),
        allocation_guaranteed=False,
    )


def profile_deltas(receipt):
    """Extract exact all-rank matching fallback deltas from controller metadata."""
    phases = []
    for phase in ("before", "after"):
        ranks = {}
        for item in receipt[phase]:
            status = item["custom_collective"]
            rank = status["rank"]
            if rank in ranks:
                raise ValueError("duplicate rank receipt")
            ranks[rank] = {(v["dtype"], v["elements"], v["contiguous"]): v["calls"] for v in status["fallback_schemas"]}
        phases.append(ranks)
    before, after = phases
    if set(before) != set(after) or not before:
        raise ValueError("before/after membership differs")
    common = None
    for rank, current in after.items():
        if not set(before[rank]) <= set(current):
            raise ValueError("fallback counters disappeared")
        delta = {key: count - before[rank].get(key, 0) for key, count in current.items()}
        if any(value < 0 for value in delta.values()):
            raise ValueError("fallback counter decreased")
        delta = {key: value for key, value in delta.items() if value}
        if common is not None and delta != common:
            raise ValueError("rank fallback deltas differ")
        common = delta
    result = []
    for (dtype, elements, contiguous), calls in sorted(common.items()):
        if not contiguous or dtype not in ("float16", "float32") or elements % HIDDEN_SIZE:
            raise ValueError("unexpected prefill SUM schema")
        result.append(dict(dtype=dtype, elements=elements, rows=elements // HIDDEN_SIZE, calls=calls))
    return result


def _exchange(live, phase, payload):
    packets = [None] * len(live.members)
    live.dist.all_gather_object(packets, dict(rank=live.rank, phase=phase, payload=payload), group=live.cpu_group)
    if any(
        packet["rank"] != rank or packet["phase"] != phase for rank, packet in zip(live.members, packets, strict=True)
    ):
        raise RuntimeError("private benchmark all-rank phase mismatch")
    return [packet["payload"] for packet in packets]


def _serving_identity(live):
    observer = live.graph_observer
    return (
        live.identity,
        live.state,
        live.claimed_stream_id,
        tuple((key, id(owner), owner.calls) for key, owner in live.owners.items()),
        None if observer is None else (id(observer), observer.snapshot()),
        dict(live.fallbacks),
    )


def _checked(live, retained, phase, action):
    result, error = None, None
    try:
        result = action()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    errors = _exchange(live, phase, error)
    if any(value is not None for value in errors):
        retained["errors"] = errors
        raise RuntimeError(f"private benchmark failed at {phase}: {errors}")
    return result


def _pattern(torch, elements, rank_position, ranks, case, dtype):
    base = torch.arange(elements, dtype=torch.int32).remainder(31).float().sub_(15).div_(8)
    source = (base + rank_position / 4 + case / 8).to(dtype)
    expected = (base * ranks + sum(range(ranks)) / 4 + ranks * case / 8).to(dtype)
    return source, expected


def _verify(torch, tensor, expected):
    result = tensor.cpu()
    if result.dtype != expected.dtype or result.shape != expected.shape or not torch.equal(result, expected):
        raise AssertionError("complete finite SUM output differs from the exact reference")
    return hashlib.sha256(result.numpy().tobytes()).hexdigest()


def summarize_timing(rank_samples):
    """Shared-host monotonic completion span, retaining arrival skew explicitly."""
    if not rank_samples or any(sample["count"] != rank_samples[0]["count"] for sample in rank_samples):
        raise ValueError("rank timing counts differ")
    count = rank_samples[0]["count"]
    if count < 1 or any(not x["started_ns"] <= x["submitted_ns"] <= x["completed_ns"] for x in rank_samples):
        raise ValueError("invalid monotonic completion timestamps")
    return dict(
        completed_us_per_call=(
            max(x["completed_ns"] for x in rank_samples) - min(x["started_ns"] for x in rank_samples)
        )
        / count
        / 1000,
        arrival_spread_us=(max(x["started_ns"] for x in rank_samples) - min(x["started_ns"] for x in rank_samples))
        / 1000,
        rank_event_us_per_call=[x["event_us_per_call"] for x in rank_samples],
        rank_submission_us_per_call=[(x["submitted_ns"] - x["started_ns"]) / count / 1000 for x in rank_samples],
    )


def _exercise(live, group, private, retained, geometry):
    torch, dist = live.torch, live.dist
    elements, dtype = geometry["elements"], getattr(torch, geometry["dtype"])
    device = f"npu:{live.device_index}"

    def synchronize():
        torch.npu.synchronize(live.device_index)

    def unexpected_fallback(value):
        raise RuntimeError("private registered shape selected fallback")

    def submit(mode, source):
        if mode == "custom_out_of_place":
            result = private.all_reduce(source, unexpected_fallback)
            retained["tensors"].append(result)
        else:
            result = source if mode == "hccl_model_in_place" else source.clone()
            retained["tensors"].append(result)
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group.device_group)
        return result

    correctness = []
    for case in range(CORRECTNESS_CASES):
        host_source, expected = _checked(
            live,
            retained,
            f"host-pattern-{case}",
            lambda case=case: _pattern(torch, elements, live.members.index(live.rank), len(live.members), case, dtype),
        )
        for mode in MODES:

            def prepare_source(host_source=host_source):
                source = host_source.to(device)
                retained["tensors"] = [source]
                synchronize()
                return source

            source = _checked(live, retained, f"correctness-ready-{case}-{mode}", prepare_source)

            def check(mode=mode, source=source, expected=expected, host_source=host_source):
                result = submit(mode, source)
                synchronize()
                digest = _verify(torch, result, expected)
                if mode != "hccl_model_in_place":
                    _verify(torch, source, host_source)
                return digest

            digest = _checked(live, retained, f"correctness-{case}-{mode}", check)
            correctness.append(dict(case=case, mode=mode, output_sha256=digest, exact=True))
    # Function defaults retain their tensors independently of the ownership
    # ledger. Drop the final correctness source before allocating timing input.
    del check, prepare_source, source, host_source, expected
    retained["tensors"].clear()

    # Zero inputs keep repeated in-place reductions finite without extra timed
    # copies. The primary baseline matches the real model's in-place semantics;
    # the secondary baseline explicitly includes its out-of-place clone cost.
    def timing_source():
        source = torch.zeros(elements, dtype=dtype, device=device)
        retained["tensors"] = [source]
        synchronize()
        return source

    source = _checked(live, retained, "timing-source", timing_source)
    samples = []
    for pair in range(TIMING_PAIRS):
        for mode in MODES if pair % 2 == 0 else reversed(MODES):

            def warm(mode=mode):
                for _ in range(WARMUP_CALLS):
                    submit(mode, source)
                synchronize()

            _checked(live, retained, f"warm-{pair}-{mode}", warm)
            retained["tensors"] = [source]

            def timing_events():
                retained["events"] = []
                for _ in range(2):
                    retained["events"].append(torch.npu.Event(enable_timing=True))
                return retained["events"]

            begin, end = _checked(live, retained, f"timing-events-{pair}-{mode}", timing_events)
            _exchange(live, f"timing-ready-{pair}-{mode}", None)

            def timed(mode=mode, begin=begin, end=end):
                started = time.monotonic_ns()
                begin.record()
                for _ in range(TIMING_BATCH):
                    submit(mode, source)
                end.record()
                submitted = time.monotonic_ns()
                synchronize()
                completed = time.monotonic_ns()
                return dict(
                    count=TIMING_BATCH,
                    started_ns=started,
                    submitted_ns=submitted,
                    completed_ns=completed,
                    event_us_per_call=begin.elapsed_time(end) * 1000 / TIMING_BATCH,
                )

            local = _checked(live, retained, f"timed-{pair}-{mode}", timed)
            ranks = _exchange(live, f"timing-result-{pair}-{mode}", local)
            samples.append(dict(pair=pair, mode=mode, **summarize_timing(ranks)))
            retained["tensors"] = [source]
            retained["events"].clear()
    _checked(live, retained, "exercise-final-drain", synchronize)
    return dict(geometry=geometry, correctness=correctness, timings=samples)


def run_probe(worker, *, members=(0, 1, 2, 3), algorithm="recursive-doubling-fixed", cases=PREFILL_CASES):
    """All ranks must call together, with suspended requests already drained."""
    from vllm.distributed import get_tp_group

    from tools.sidecard.qfn_private_collective_lifecycle import create_private_collective

    if getattr(worker, "_qfn_prefill_collective_probe", None) is not None:
        raise RuntimeError("prior private benchmark still owns resources; do not retry")
    group = get_tp_group()
    live = group.device_communicator.custom_sum
    if live is None or live.state != "active" or tuple(live.members) != tuple(members):
        raise RuntimeError("requires the requested active serving collective")
    if live.config["algorithm"] != algorithm:
        raise RuntimeError("serving algorithm differs from the requested private gate")
    # Validate every requested shape and derive this rank's actual degree before
    # acquiring resources. TP6 binomial roots need more inboxes than TP4 RD.
    schedules = [
        BinomialSchedule(tuple(members), dtype, rows * HIDDEN_SIZE, algorithm=algorithm) for rows, dtype, _ in cases
    ]
    if not schedules:
        raise ValueError("at least one private geometry is required")
    retained = dict(private=None, tensors=[], events=[], state="running", completed=[])
    worker._qfn_prefill_collective_probe = retained
    try:
        _checked(live, retained, "serving-context", live._context)
        before = _checked(live, retained, "serving-identity", lambda: _serving_identity(live))
        _checked(live, retained, "initial-drain", lambda: live.torch.npu.synchronize(live.device_index))
        nonce = _exchange(live, "nonce", secrets.token_hex(16) if live.rank == live.members[0] else None)[0]
        root = Path(live.config["output_root"]) / "private-prefill-probe" / nonce
        for index, (rows, dtype, calls) in enumerate(cases):
            geometry = dict(dtype=dtype, elements=rows * HIDDEN_SIZE, rows=rows, observed_calls=calls)
            peers = len(schedules[index].peers(live.rank))
            budget = memory_budget(geometry["elements"], dtype, peers=peers)

            def capacity_check(budget=budget, geometry=geometry, peers=peers):
                free, _ = live.torch.npu.mem_get_info(live.device_index)
                if free < budget["required_free_bytes"]:
                    raise RuntimeError("insufficient free device memory for bounded private benchmark")
                pools = read_p2p_pools(live)
                admission = dict(geometry=geometry, aggregate_free_bytes=int(free), p2p_pools=pools)
                retained.setdefault("admission", []).append(admission)
                admission.update(require_p2p_capacity(pools, geometry["elements"], geometry["dtype"], peers=peers))
                return admission

            _checked(live, retained, f"memory-{index}", capacity_check)
            config = _checked(
                live,
                retained,
                f"config-{index}",
                lambda geometry=geometry, dtype=dtype, index=index: private_config(
                    live.config, geometry["elements"], dtype, root / str(index)
                ),
            )
            private = create_private_collective(
                config,
                live.members,
                live.rank,
                live.cpu_group,
                live.device_index,
                torch=live.torch,
                dist=live.dist,
                stream_getter=live.stream_getter,
                checked=lambda phase, action, index=index: _checked(live, retained, f"private-{index}-{phase}", action),
                retain=lambda owner: retained.update(private=owner),
            )
            report = _exercise(live, group, private, retained, geometry)
            private.close(graphs_retired=True)  # This separate eager facade owns no graphs.
            if private.state != "closed":
                raise RuntimeError("private collective did not close")
            report.update(memory_budget=budget, private_owner_closed=True)
            retained["completed"].append(report)
            retained["tensors"].clear()
            retained["private"] = None

        def unchanged():
            if _serving_identity(live) != before:
                raise RuntimeError("serving collective identity, counters or graph ownership changed")

        _checked(live, retained, "serving-unchanged", unchanged)
        result = dict(
            passed=True,
            scope="resident_private_eager_prefill_SUM_not_model_or_graph_qualification",
            algorithms=dict(custom=live.config["algorithm"], reference="existing TP HCCL"),
            serving_collective_replaced=False,
            new_process_groups=False,
            timing_input="zero; repeated in-place SUM stays finite",
            records=retained["completed"],
        )
        worker._qfn_prefill_collective_probe = None
        return result
    except Exception:
        retained["state"] = "failed-retained"
        raise
