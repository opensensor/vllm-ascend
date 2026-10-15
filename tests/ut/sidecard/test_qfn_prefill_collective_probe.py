# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the private resident benchmark; no NPU initialization."""

import copy
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qfn_prefill_collectives as rpc
from tools.sidecard import qfn_prefill_collective_probe as probe
from tools.sidecard import qfn_private_collective_lifecycle as lifecycle


def test_largest_owner_and_private_tensor_budget():
    budget = probe.memory_budget(2048 * 2560, "float32")
    assert budget == dict(
        payload_bytes=20 << 20,
        owner_bytes=121 << 20,
        tensor_bytes=80 << 20,
        required_free_bytes=265 << 20,
    )
    assert probe.memory_budget(2048 * 2560, "float16")["owner_bytes"] == 61 << 20


def p2p_pools(huge, normal):
    return [
        dict(attribute=attribute, status=0, free_bytes=free, total_bytes=free)
        for attribute, free in ((6, huge), (7, normal))
    ]


def test_p2p_requirement_includes_all_seven_padded_buffers():
    result = probe.require_p2p_capacity(p2p_pools(0, 122 << 20), 2048 * 2560, "float32")
    assert result["minimum_p2p_bytes"] == (121 << 20) + 7 * 32
    assert result["largest_p2p_allocation_bytes"] == (20 << 20) + 32
    assert result["allocation_guaranteed"] is False


@pytest.mark.parametrize(
    "ordinary_normal,p2p_normal",
    [
        (3207012352, 2088960),
        (3226841088, 2097152),
        (3213115392, 2088960),
        (3208888320, 2097152),
    ],
)
def test_measured_four_rank_pool_receipt_rejects_failed_geometry(ordinary_normal, p2p_normal):
    # Exact counters copied from qsa-resident-v1/prefill-allocator-inspect.json.
    assert ordinary_normal > 3_000_000_000
    with pytest.raises(RuntimeError, match="insufficient P2P pool capacity"):
        probe.require_p2p_capacity(p2p_pools(0, p2p_normal), 640 * 2560, "float16")


@pytest.mark.parametrize("failure", ["status", "missing", "duplicate", "negative", "noninteger"])
def test_p2p_admission_fails_closed_on_unavailable_or_malformed_pool_data(failure):
    pools = p2p_pools(1 << 30, 1 << 30)
    if failure == "status":
        pools[0]["status"] = 207000
    elif failure == "missing":
        pools.pop()
    elif failure == "duplicate":
        pools[1]["attribute"] = 6
    elif failure == "negative":
        pools[0]["free_bytes"] = -1
    else:
        pools[0]["free_bytes"] = True
    with pytest.raises(RuntimeError, match="P2P"):
        probe.require_p2p_capacity(pools, 640 * 2560, "float16")


def test_p2p_admission_rejects_total_shortage_and_single_pool_shortage():
    with pytest.raises(RuntimeError, match="insufficient P2P pool capacity"):
        probe.require_p2p_capacity(p2p_pools(0, 2 << 20), 640 * 2560, "float16")
    with pytest.raises(RuntimeError, match="individual P2P pool"):
        probe.require_p2p_capacity(p2p_pools(600000, 600000), 1, "float16")


@pytest.mark.parametrize("elements,dtype", [(0, "float32"), (probe.MAX_ELEMENTS + 1, "float16"), (1, "int8")])
def test_memory_budget_rejects_native_unsupported_geometry(elements, dtype):
    with pytest.raises(ValueError):
        probe.memory_budget(elements, dtype)


def test_private_config_does_not_mutate_live_geometry_or_paths():
    live = dict(geometries=[dict(dtype="float16", elements=2560)], execution_mode="graph", output_root="/live")
    before = copy.deepcopy(live)
    private = probe.private_config(live, 5242880, "float32", "/private/run")
    assert live == before
    assert private["geometries"] == [dict(dtype="float32", elements=5242880)]
    assert private["execution_mode"] == "eager" and private["graph_replay_tracking"] is False
    assert private["output_root"] == "/private/run"
    with pytest.raises(ValueError):
        probe.private_config(live, 10, "float16", "relative")


def receipt():
    return {
        "before": [dict(custom_collective=dict(rank=r, fallback_schemas=[])) for r in range(4)],
        "after": [
            dict(
                custom_collective=dict(
                    rank=r,
                    fallback_schemas=[
                        dict(dtype=dtype, elements=rows * 2560, calls=calls, contiguous=True)
                        for rows, dtype, calls in probe.PREFILL_CASES
                    ],
                )
            )
            for r in range(4)
        ],
    }


def test_exact_profile_deltas_account_for_all_500_reductions():
    rows = probe.profile_deltas(receipt())
    assert sum(x["calls"] for x in rows) == 500
    assert {(x["rows"], x["dtype"], x["calls"]) for x in rows} == set(probe.PREFILL_CASES)


@pytest.mark.parametrize("failure", ["rank", "dtype", "negative", "missing", "duplicate"])
def test_profile_deltas_reject_bad_evidence(failure):
    values = receipt()
    if failure == "rank":
        values["after"][0]["custom_collective"]["fallback_schemas"][0]["calls"] += 1
    elif failure == "dtype":
        for rank in values["after"]:
            rank["custom_collective"]["fallback_schemas"][0]["dtype"] = "int8"
    elif failure == "negative":
        for rank in values["after"]:
            rank["custom_collective"]["fallback_schemas"][0]["calls"] = -1
    elif failure == "missing":
        values["after"].pop()
    else:
        values["after"].append(values["after"][0])
    with pytest.raises(ValueError):
        probe.profile_deltas(values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nonconstant_finite_reference_is_exact_across_all_ranks(dtype):
    for case in range(probe.CORRECTNESS_CASES):
        sources = [probe._pattern(torch, 100, rank, 4, case, dtype) for rank in range(4)]
        actual = (sources[0][0] + sources[1][0]) + (sources[2][0] + sources[3][0])
        assert torch.equal(actual, sources[0][1])
        assert probe._verify(torch, actual, sources[0][1])
        with pytest.raises(AssertionError):
            probe._verify(torch, actual + 1, sources[0][1])


def test_timing_summary_reports_complete_group_and_arrival_skew():
    samples = [
        dict(count=2, started_ns=1000, submitted_ns=2000, completed_ns=7000, event_us_per_call=2),
        dict(count=2, started_ns=3000, submitted_ns=4000, completed_ns=9000, event_us_per_call=3),
    ]
    result = probe.summarize_timing(samples)
    assert result["completed_us_per_call"] == 4
    assert result["arrival_spread_us"] == 2
    assert result["rank_event_us_per_call"] == [2, 3]
    assert result["rank_submission_us_per_call"] == [0.5, 0.5]
    samples[0]["completed_ns"] = 0
    with pytest.raises(ValueError):
        probe.summarize_timing(samples)


def fake_runtime(monkeypatch, *, fail_create=False, fail_exercise=False, fail_close=False, low_memory=False):
    events, created = [], []

    def gather(output, packet, *, group):
        output[:] = [dict(packet, rank=rank) for rank in range(4)]

    live = SimpleNamespace(
        state="active",
        members=(0, 1, 2, 3),
        rank=0,
        device_index=0,
        cpu_group=object(),
        config=dict(
            algorithm="recursive-doubling-fixed", output_root="/live", geometries=[dict(dtype="float16", elements=2560)]
        ),
        torch=SimpleNamespace(
            npu=SimpleNamespace(
                synchronize=lambda device: events.append("drain"),
                mem_get_info=lambda device: (0 if low_memory else 1 << 40, 1 << 40),
            )
        ),
        dist=SimpleNamespace(all_gather_object=gather),
        stream_getter=object(),
        _context=lambda: None,
        identity="serving",
        claimed_stream_id=7,
        owners={},
        fallbacks={},
        graph_observer=None,
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(custom_sum=live), device_group=object())
    distributed = ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: group
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    worker = SimpleNamespace(rank=0)

    def create(config, *args, **kwargs):
        owner = SimpleNamespace(state="active", config=config)
        created.append(owner)
        kwargs["retain"](owner)

        def close(*, graphs_retired):
            assert graphs_retired is True
            events.append("close")
            if fail_close:
                raise RuntimeError("uncertain close")
            owner.state = "closed"

        owner.close = close
        events.append("create")
        if fail_create:
            error = RuntimeError("partial setup")
            error.collective_owner = owner
            raise error
        return owner

    def exercise(live, group, private, retained, geometry):
        events.append("exercise")
        retained["tensors"].append(object())
        if fail_exercise:
            raise RuntimeError("uncertain submission")
        return dict(geometry=geometry)

    monkeypatch.setattr(lifecycle, "create_private_collective", create)
    monkeypatch.setattr(probe, "_exercise", exercise)
    monkeypatch.setattr(probe, "read_p2p_pools", lambda live: p2p_pools(1 << 40, 1 << 40))
    return worker, live, events, created


def test_success_closes_each_private_owner_before_next_allocation(monkeypatch):
    worker, live, events, created = fake_runtime(monkeypatch)
    original = copy.deepcopy(live.config)
    result = probe.run_probe(worker)
    assert result["passed"] and len(result["records"]) == 6
    assert worker._qfn_prefill_collective_probe is None
    assert live.config == original
    assert events == ["drain", *(["create", "exercise", "close"] * 6)]
    assert all(owner.state == "closed" and owner.config["execution_mode"] == "eager" for owner in created)
    assert len({owner.config["output_root"] for owner in created}) == 6


@pytest.mark.parametrize("failure", ["create", "exercise", "close"])
def test_failures_retain_private_owners_and_forbid_retry(monkeypatch, failure):
    worker, live, events, created = fake_runtime(
        monkeypatch,
        fail_create=failure == "create",
        fail_exercise=failure == "exercise",
        fail_close=failure == "close",
    )
    with pytest.raises(RuntimeError):
        probe.run_probe(worker)
    retained = worker._qfn_prefill_collective_probe
    assert retained["state"] == "failed-retained"
    assert retained["private"] is created[0]
    assert events.count("close") == (1 if failure == "close" else 0)
    assert len(created) == 1 and live.state == "active"
    if failure != "create":
        assert retained["tensors"]
    with pytest.raises(RuntimeError, match="do not retry"):
        probe.run_probe(worker)


def test_memory_rejection_allocates_no_owner(monkeypatch):
    worker, _, _, created = fake_runtime(monkeypatch, low_memory=True)
    with pytest.raises(RuntimeError, match="insufficient free"):
        probe.run_probe(worker)
    assert not created


@pytest.mark.parametrize("error", ["insufficient free device memory", "insufficient P2P pool capacity"])
def test_one_peer_capacity_failure_rejects_allocation_on_healthy_rank(monkeypatch, error):
    worker, live, _, created = fake_runtime(monkeypatch)
    original = live.dist.all_gather_object

    def fail_peer(output, packet, *, group):
        original(output, packet, group=group)
        if packet["phase"] == "memory-0":
            output[2]["payload"] = error

    live.dist.all_gather_object = fail_peer
    with pytest.raises(RuntimeError, match=error):
        probe.run_probe(worker)
    assert not created


def test_abundant_ordinary_memory_does_not_admit_exhausted_p2p_pool(monkeypatch):
    worker, _, _, created = fake_runtime(monkeypatch)
    monkeypatch.setattr(probe, "read_p2p_pools", lambda live: p2p_pools(0, 2 << 20))
    with pytest.raises(RuntimeError, match="insufficient P2P pool capacity"):
        probe.run_probe(worker)
    assert not created
    retained = worker._qfn_prefill_collective_probe
    assert retained["private"] is None
    assert retained["admission"][0]["aggregate_free_bytes"] == 1 << 40
    assert retained["admission"][0]["p2p_pools"] == p2p_pools(0, 2 << 20)


def test_serving_counter_change_is_detected_before_success(monkeypatch):
    worker, live, _, created = fake_runtime(monkeypatch)
    original = probe._exercise

    def mutate(*args):
        result = original(*args)
        live.fallbacks[("float32", 1, True)] = 1
        return result

    monkeypatch.setattr(probe, "_exercise", mutate)
    with pytest.raises(RuntimeError, match="serving collective identity"):
        probe.run_probe(worker)
    assert all(owner.state == "closed" for owner in created)
    assert worker._qfn_prefill_collective_probe["state"] == "failed-retained"


def test_checked_phase_reports_peer_failure_and_retains_diagnostics():
    def gather(output, packet, *, group):
        output[:] = [dict(packet, rank=0), dict(packet, rank=1, payload="peer failure")]

    live = SimpleNamespace(rank=0, members=(0, 1), cpu_group=None, dist=SimpleNamespace(all_gather_object=gather))
    retained = {}
    with pytest.raises(RuntimeError, match="peer failure"):
        probe._checked(live, retained, "gate", lambda: 123)
    assert retained["errors"] == [None, "peer failure"]


def test_correctness_tensor_closures_do_not_inflate_timing_budget(monkeypatch):
    live_tensors = weakref.WeakSet()
    peak_counts = []

    class DeviceTensor:
        def __init__(self):
            live_tensors.add(self)
            peak_counts.append(len(live_tensors))

        def clone(self):
            return DeviceTensor()

    class HostTensor:
        def to(self, device):
            return DeviceTensor()

    def zeros(*args, **kwargs):
        assert not live_tensors, "correctness closures still own device tensors"
        return DeviceTensor()

    def gather(output, packet, *, group):
        output[:] = [packet]

    torch_stub = SimpleNamespace(
        float32="float32",
        zeros=zeros,
        npu=SimpleNamespace(
            synchronize=lambda device: None,
            Event=lambda **kwargs: SimpleNamespace(record=lambda: None, elapsed_time=lambda end: 1.0),
        ),
    )
    live = SimpleNamespace(
        torch=torch_stub,
        dist=SimpleNamespace(
            all_gather_object=gather, all_reduce=lambda *args, **kwargs: None, ReduceOp=SimpleNamespace(SUM="sum")
        ),
        device_index=0,
        rank=0,
        members=(0,),
        cpu_group=None,
    )
    retained = dict(tensors=[], events=[])
    private = SimpleNamespace(all_reduce=lambda source, fallback: DeviceTensor())
    monkeypatch.setattr(probe, "_pattern", lambda *args: (HostTensor(), HostTensor()))
    monkeypatch.setattr(probe, "_verify", lambda *args: "exact")
    result = probe._exercise(
        live, SimpleNamespace(device_group=None), private, retained, dict(elements=1, dtype="float32")
    )
    assert len(result["correctness"]) == 6
    assert len(result["timings"]) == 15
    assert max(peak_counts) <= 4
    retained["tensors"].clear()
    assert not live_tensors


def test_rpc_delegation_success_and_profiler_rejection():
    worker = SimpleNamespace(rank=2)
    function = rpc.make_rpc(lambda self, output: output, probe=lambda self: dict(passed=True))
    assert function(worker) == dict(rank=2, prefill_collectives=dict(passed=True))
    assert function(worker, "report.json") == "report.json"
    worker._qfn_native_profile_session = object()
    result = function(worker)
    assert not result["prefill_collectives"]["passed"]
    assert "finish profiling" in result["prefill_collectives"]["error"]
