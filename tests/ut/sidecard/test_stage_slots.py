# SPDX-License-Identifier: Apache-2.0
"""Pure evidence transitions: transfers, caches, consumers and retained owners."""

import copy
import json
import threading
from dataclasses import FrozenInstanceError, replace

import pytest

from tools.sidecard.stage_slots import (
    DEFAULT_CAPACITY,
    Quarantined,
    RankLease,
    Registration,
    StageSlots,
    WouldBlock,
    memory_requirements,
)
from tools.sidecard.stage_wire import descriptors


def registration(**changes):
    return Registration("session", "nonce", "a" * 64, ("allocation-set-a", "allocation-set-b"), **changes)


def controller(*, start_request=True):
    model = StageSlots(registration())
    for rank in range(2):
        model.offer_mapping(rank)
        model.register(rank, model.registration.packet(rank))
    if start_request:
        begin(model)
    return model


def begin(model, request="request", reservation=131072):
    generation = model.begin_request(request, reservation)
    for owner in ("cuda", "rank0", "rank1"):
        model.reset_started(request, generation, owner)
        model.reset_complete(request, generation, owner)
    return generation


def submit(model, position=0, rows=32, **options):
    return model.submit(first_position=position, rows=rows, payload_bytes=descriptors(rows)[1], **options)


def prefix(model, item):
    model.prefix_started(item)
    model.producer_ready(item)


def upload(model, item):
    for rank in range(2):
        model.upload_started(item.for_rank(rank))
        model.upload_complete(item.for_rank(rank))


def compute(model, item):
    for rank in range(2):
        model.compute_started(item.for_rank(rank))
    for rank in range(2):
        model.compute_complete(item.for_rank(rank))


def result(model, item):
    model.return_started(item.for_rank(0))
    model.return_complete(item.for_rank(0))


def consume(model, item):
    assert model.receive() == item
    model.cuda_visible(item)
    model.suffix_started(item)
    model.retire(item)


def complete(model, item):
    prefix(model, item)
    upload(model, item)
    compute(model, item)
    result(model, item)
    consume(model, item)


def unmap(model, rank):
    model.mapping_released(rank, model.registration.packet(rank), guard_poison_exact=True)


def test_two_slot_prefix_middle_overlap_and_ordered_retirement():
    model = controller()
    a, b = submit(model), submit(model, 32)
    prefix(model, a)
    upload(model, a)
    for rank in range(2):
        model.compute_started(a.for_rank(rank))
    prefix(model, b)  # RTX prefix B while Ascend owns A.
    with pytest.raises(WouldBlock):
        submit(model, 64)
    with pytest.raises(RuntimeError, match="overlaps rank"):
        model.upload_started(b.for_rank(0))
    for rank in range(2):
        model.compute_complete(a.for_rank(rank))
    result(model, a)
    upload(model, b)
    compute(model, b)
    result(model, b)
    consume(model, a)
    consume(model, b)
    assert model.snapshot()["pending"] == []
    assert model.snapshot()["next_compute_sequence"] == model.snapshot()["next_receive_sequence"] == 2
    model.close_request()


def test_result_dma_does_not_retire_a_cuda_consumer():
    model = controller()
    a = submit(model)
    prefix(model, a)
    upload(model, a)
    compute(model, a)
    result(model, a)
    with pytest.raises(RuntimeError, match="suffix consumer"):
        model.retire(a)
    assert model.receive() == a
    with pytest.raises(RuntimeError, match="visibility"):
        model.suffix_started(a)
    model.cuda_visible(a)
    model.suffix_started(a)
    b = submit(model, 32)
    with pytest.raises(RuntimeError, match="overlapping"):
        model.prefix_started(b)
    with pytest.raises(WouldBlock):
        submit(model, 64)
    model.retire(a)
    prefix(model, b)
    assert model.snapshot()["pending"][0]["sequence"] == b.sequence


def test_ascend_cache_mutation_and_cuda_receipt_cannot_reorder():
    model = controller()
    a, b = submit(model), submit(model, 32)
    with pytest.raises(RuntimeError, match="prefix cache order"):
        model.prefix_started(b)
    prefix(model, a)
    prefix(model, b)
    upload(model, b)
    with pytest.raises(RuntimeError, match="cache order"):
        model.compute_started(b.for_rank(0))
    upload(model, a)
    compute(model, a)
    compute(model, b)
    result(model, b)  # Separate result allocations may complete out of order.
    with pytest.raises(WouldBlock):
        model.receive()
    result(model, a)
    assert model.receive() == a
    with pytest.raises(WouldBlock):
        model.receive()
    model.cuda_visible(a)
    model.suffix_started(a)
    model.retire(a)
    consume(model, b)


def test_both_rank_upload_and_compute_quorums_are_required():
    model = controller()
    a = submit(model)
    with pytest.raises(RuntimeError, match="producer readiness"):
        model.upload_started(a.for_rank(0))
    prefix(model, a)
    model.upload_started(a.for_rank(0))
    model.upload_complete(a.for_rank(0))
    with pytest.raises(RuntimeError, match="upload quorum"):
        model.compute_started(a.for_rank(0))
    model.upload_started(a.for_rank(1))
    model.upload_complete(a.for_rank(1))
    model.compute_started(a.for_rank(0))
    model.compute_started(a.for_rank(1))
    model.compute_complete(a.for_rank(0))
    with pytest.raises(RuntimeError, match="both-rank"):
        model.return_started(a.for_rank(0))
    model.compute_complete(a.for_rank(1))
    with pytest.raises(RuntimeError, match="rank0"):
        model.return_started(a.for_rank(1))
    result(model, a)


@pytest.mark.parametrize("phase", ["upload_complete", "compute_complete", "return_complete"])
def test_completion_without_matching_started_operation_is_rejected(phase):
    model = controller()
    a = submit(model)
    with pytest.raises(RuntimeError):
        model.accept(phase, a.for_rank(0).packet())


@pytest.mark.parametrize(
    "field,value",
    [
        ("session", "old-session"),
        ("nonce", "old-nonce"),
        ("request", "other"),
        ("request_generation", 0),
        ("sequence", 1),
        ("generation", 0),
        ("slot", 1),
        ("mapping_id", "other-mapping"),
        ("first_position", 1),
        ("rows", 1),
        ("reservation", 65536),
        ("payload_bytes", 1),
        ("mode", "decode"),
        ("physical_chip", 3),
        ("generation", True),
        ("rank", False),
    ],
)
def test_altered_or_stale_lease_cannot_acknowledge_live_work(field, value):
    model = controller()
    a = submit(model)
    prefix(model, a)
    lease = replace(a.for_rank(0), **{field: value})
    before = model.snapshot()
    with pytest.raises((ValueError, RuntimeError)):
        model.upload_started(lease)
    assert model.snapshot() == before


def test_slot_reuse_rejects_previous_generation_even_in_the_same_request():
    model = controller()
    a = submit(model)
    complete(model, a)
    b = submit(model, 32)
    assert a.slot == b.slot and b.generation == a.generation + 1
    prefix(model, b)
    with pytest.raises(ValueError, match="stale"):
        model.upload_started(a.for_rank(0))
    with pytest.raises(ValueError, match="stale"):
        model.producer_ready(a)


def test_request_reuse_requires_new_generation_and_completed_resets():
    model = controller()
    a = submit(model)
    with pytest.raises(RuntimeError, match="still owns"):
        model.close_request()
    with pytest.raises(RuntimeError, match="another request"):
        model.begin_request("new", 131072)
    complete(model, a)
    model.close_request()
    generation = model.begin_request("request", 131072)
    assert generation == a.request_generation + 1
    with pytest.raises(ValueError, match="stale request"):
        model.reset_complete("request", a.request_generation, "rank0")
    with pytest.raises(RuntimeError, match="not reset"):
        submit(model)
    for owner in ("cuda", "rank0", "rank1"):
        model.reset_started("request", generation, owner)
        model.reset_complete("request", generation, owner)
    b = submit(model)
    assert b.sequence == a.sequence + 1
    with pytest.raises(ValueError, match="replayed cache reset"):
        model.reset_complete("request", generation, "rank0")


@pytest.mark.parametrize(
    "position,rows,payload,mode",
    [
        (1, 32, 100, "prefill"),
        (0, 33, 100, "prefill"),
        (0, 0, 100, "prefill"),
        (False, 1, 100, "prefill"),
        (0, 1, DEFAULT_CAPACITY + 1, "prefill"),
        (0, 1, 0, "prefill"),
        (0, 2, 100, "decode"),
        (0, 1, 100, "other"),
    ],
)
def test_invalid_submission_does_not_consume_credit_or_advance_position(position, rows, payload, mode):
    model = controller()
    before = model.snapshot()
    with pytest.raises(ValueError):
        model.submit(first_position=position, rows=rows, payload_bytes=payload, mode=mode)
    assert model.snapshot() == before


def test_full_context_reservation_remains_131072_and_cannot_overflow():
    model = controller(start_request=False)
    begin(model, reservation=32)
    complete(model, submit(model))
    with pytest.raises(ValueError, match="overflow"):
        submit(model, 32, 1)
    with pytest.raises(ValueError, match="full 131072"):
        registration(max_context=65536)


def test_decode_admission_stays_serialized_in_both_directions():
    model = controller()
    decode = submit(model, rows=1, mode="decode")
    with pytest.raises(WouldBlock):
        submit(model, 1)
    complete(model, decode)
    prefill = submit(model, 1)
    with pytest.raises(WouldBlock):
        submit(model, 33, rows=1, mode="decode")
    complete(model, prefill)


def test_cancelled_drain_waits_for_both_ranks_then_discards_without_suffix():
    model = controller()
    a, b = submit(model), submit(model, 32)
    prefix(model, a)
    prefix(model, b)
    upload(model, a)
    model.begin_drain(now=100, timeout=5, cancelled=True)
    assert not model.poll_drain(101)
    with pytest.raises(RuntimeError, match="draining"):
        submit(model, 64)
    with pytest.raises(RuntimeError, match="discard"):
        model.discard_cancelled(a)
    for item in (a, b):
        if item == b:
            upload(model, item)
        compute(model, item)
        result(model, item)
        assert model.receive() == item
        model.cuda_visible(item)
        model.discard_cancelled(item)
    assert model.poll_drain(102)
    assert not model.snapshot()["owners_releasable"]
    unmap(model, 0)
    unmap(model, 1)
    assert model.snapshot()["owners_releasable"]
    model.release_owners()
    assert model.snapshot()["state"] == "closed"


def test_cancel_does_not_discard_an_inflight_cuda_consumer():
    model = controller()
    a = submit(model)
    prefix(model, a)
    upload(model, a)
    compute(model, a)
    result(model, a)
    model.receive()
    model.cuda_visible(a)
    model.suffix_started(a)
    model.begin_drain(now=1, timeout=5, cancelled=True)
    with pytest.raises(RuntimeError, match="no CUDA consumer"):
        model.discard_cancelled(a)
    assert not model.poll_drain(2)
    model.retire(a)
    assert model.poll_drain(3)


@pytest.mark.parametrize("failure", ["worker loss", "uncertain native completion", "client socket lost"])
def test_uncertain_native_failure_quarantines_and_retains_all_owners(failure):
    model = controller()
    a = submit(model)
    prefix(model, a)
    model.upload_started(a.for_rank(0))
    before = model.snapshot()
    model.quarantine(failure)
    after = model.snapshot()
    assert after["pending"] == before["pending"]
    assert after["native_busy"] == before["native_busy"]
    assert after["state"] == "quarantined" and not after["owners_releasable"]
    for action in (lambda: model.upload_complete(a.for_rank(0)), model.release_owners, lambda: unmap(model, 0)):
        with pytest.raises(Quarantined):
            action()
    json.dumps(after, allow_nan=False)


@pytest.mark.parametrize("inflight", [False, True])
def test_bounded_drain_timeout_keeps_pending_consumers_or_foreign_mappings(inflight):
    model = controller()
    if inflight:
        submit(model)
    model.begin_drain(now=10, timeout=2)
    with pytest.raises(Quarantined, match="expired"):
        model.poll_drain(12)
    assert not model.snapshot()["owners_releasable"]
    assert len(model.snapshot()["pending"]) == int(inflight)


def test_clean_shutdown_requires_both_mapping_acks_and_guard_integrity():
    model = controller()
    complete(model, submit(model))
    with pytest.raises(RuntimeError, match="drained"):
        unmap(model, 0)
    model.begin_drain(now=0, timeout=30)
    assert model.poll_drain(1)
    unmap(model, 0)
    with pytest.raises(RuntimeError, match="outlive"):
        model.release_owners()
    with pytest.raises(RuntimeError, match="matching"):
        unmap(model, 0)
    unmap(model, 1)
    model.release_owners()
    with pytest.raises(RuntimeError, match="closed"):
        model.receive()


def test_modified_guard_quarantines_without_acknowledging_mapping_release():
    model = controller()
    model.begin_drain(now=0, timeout=30)
    with pytest.raises(Quarantined, match="guard"):
        model.mapping_released(0, model.registration.packet(0), guard_poison_exact=False)
    assert model.snapshot()["unmapped_ranks"] == []


@pytest.mark.parametrize("change", ["version", "uapi", "chip", "nonce", "extra", "boolean-rank", "mapping-order"])
def test_versioned_registration_rejects_altered_rank_identity(change):
    model = StageSlots(registration())
    model.offer_mapping(0)
    packet = model.registration.packet(0)
    if change == "version":
        packet["schema_version"] += 1
    elif change == "uapi":
        packet["bridge_uapi_version"] -= 1
    elif change == "chip":
        packet["physical_chip"] = 0
    elif change == "nonce":
        packet["nonce"] = "stale"
    elif change == "extra":
        packet["untrusted"] = True
    elif change == "boolean-rank":
        packet["rank"] = False
    else:
        packet["mapping_ids"].reverse()
    with pytest.raises(ValueError, match="registration"):
        model.register(0, packet)
    assert model.snapshot()["registered_ranks"] == []


def test_duplicate_registration_and_unregistered_request_are_rejected():
    model = StageSlots(registration())
    model.offer_mapping(0)
    model.register(0, model.registration.packet(0))
    with pytest.raises(ValueError, match="replayed"):
        model.register(0, model.registration.packet(0))
    with pytest.raises(RuntimeError, match="another request"):
        model.begin_request("req", 131072)
    model.begin_drain(now=0, timeout=30)
    unmap(model, 0)  # Rank1 never registered/mapped, so it has no release duty.
    model.release_owners()


def test_exact_wire_roundtrip_is_immutable_and_replay_is_rejected():
    model = controller()
    a = submit(model)
    prefix(model, a)
    packet = json.loads(json.dumps(a.for_rank(0).packet()))
    assert RankLease.from_packet(packet) == a.for_rank(0)
    model.accept("upload_started", packet)
    model.accept("upload_complete", packet)
    with pytest.raises(RuntimeError, match="stale"):
        model.accept("upload_complete", packet)
    with pytest.raises(FrozenInstanceError):
        a.generation = 123
    packet["extra"] = 0
    with pytest.raises(ValueError, match="envelope"):
        model.accept("upload_complete", packet)


@pytest.mark.parametrize("now,timeout", [(float("nan"), 1), (0, float("inf")), (0, 0), (0, 301), (True, 1)])
def test_invalid_deadlines_do_not_change_state(now, timeout):
    model = controller()
    with pytest.raises(ValueError):
        model.begin_drain(now=now, timeout=timeout)
    assert model.snapshot()["state"] == "open"


def test_drain_deadline_cannot_extend_or_run_backwards():
    model = controller()
    model.begin_drain(now=10, timeout=30)
    with pytest.raises(RuntimeError, match="extended"):
        model.begin_drain(now=11, timeout=30)
    with pytest.raises(ValueError, match="monotonic"):
        model.poll_drain(9)


def test_control_model_enforces_owner_thread_without_calling_any_device_runtime():
    model = controller()
    errors = []

    def foreign():
        try:
            model.receive()
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=foreign)
    worker.start()
    worker.join()
    assert errors == ["control transitions require the owning thread"]


def test_memory_formula_separates_cuda_hbm_and_host_guards():
    budget = memory_requirements(cuda_scratch_bytes=17, rank_scratch_bytes=(19, 23))
    assert budget["cuda_input_result_bytes"] == 16 * 1024**2
    assert budget["cuda_total_extra_bytes"] == 16 * 1024**2 + 17
    assert budget["rank_hbm_input_result_bytes"] == [16 * 1024**2] * 2
    assert budget["rank_hbm_total_extra_bytes"] == [16 * 1024**2 + 19, 16 * 1024**2 + 23]
    assert budget["rank_host_guard_bytes"] == [8 * 1024**2] * 2
    assert budget["host_guard_total_bytes"] == 16 * 1024**2
    assert budget["hardware_validated"] is False
    with pytest.raises(ValueError):
        memory_requirements(rank_scratch_bytes=[0, 0])


def test_snapshot_is_detached_from_internal_control_state():
    model = controller()
    a = submit(model)
    receipt = copy.deepcopy(model.snapshot())
    receipt["pending"][0]["generation"] = 100
    assert model.snapshot()["pending"][0]["generation"] == a.generation


def test_completed_cancel_can_reuse_registered_owners_only_after_fresh_resets():
    model = controller()
    a = submit(model)
    prefix(model, a)
    upload(model, a)
    model.begin_drain(now=0, timeout=30, cancelled=True)
    with pytest.raises(RuntimeError, match="every consumer"):
        model.resume_after_cancel(1)
    compute(model, a)
    result(model, a)
    model.receive()
    model.cuda_visible(a)
    model.discard_cancelled(a)
    model.resume_after_cancel(2)
    assert model.snapshot()["state"] == "open"
    assert model.snapshot()["registered_ranks"] == [0, 1]
    generation = model.begin_request("next", 131072)
    with pytest.raises(RuntimeError, match="not reset"):
        submit(model)
    for owner in ("cuda", "rank0", "rank1"):
        model.reset_started("next", generation, owner)
        model.reset_complete("next", generation, owner)
    b = submit(model)
    assert b.sequence == 1
    complete(model, b)


def test_partially_unmapped_cancel_cannot_resume():
    model = controller()
    model.begin_drain(now=0, timeout=30, cancelled=True)
    unmap(model, 0)
    with pytest.raises(RuntimeError, match="cannot resume"):
        model.resume_after_cancel(1)


def test_noncanonical_payload_is_rejected_even_when_it_fits_the_slot():
    model = controller()
    with pytest.raises(ValueError, match="canonical"):
        model.submit(first_position=0, rows=32, payload_bytes=1024)


def test_export_offer_without_ready_ack_still_retains_allocation_owners():
    model = StageSlots(registration())
    model.offer_mapping(0)  # Before sending FD: retain even if ready ACK is lost.
    model.begin_drain(now=0, timeout=2)
    assert model.snapshot()["registered_ranks"] == []
    assert not model.snapshot()["owners_releasable"]
    with pytest.raises(RuntimeError, match="outlive"):
        model.release_owners()
    with pytest.raises(Quarantined):
        model.poll_drain(2)
    assert model.snapshot()["offered_ranks"] == [0]


def test_partially_registered_rank_can_close_only_with_authenticated_unmap_ack():
    model = StageSlots(registration())
    model.offer_mapping(0)
    model.begin_drain(now=0, timeout=2)
    packet = model.registration.packet(0)
    packet["nonce"] = "wrong"
    with pytest.raises(RuntimeError, match="matching"):
        model.mapping_released(0, packet, guard_poison_exact=True)
    unmap(model, 0)
    model.release_owners()
    assert model.snapshot()["state"] == "closed"


def test_registration_without_recording_export_ownership_is_rejected():
    model = StageSlots(registration())
    with pytest.raises(ValueError, match="registration"):
        model.register(0, model.registration.packet(0))
    model.offer_mapping(0)
    with pytest.raises(ValueError, match="replayed mapping offer"):
        model.offer_mapping(0)


def test_cache_reset_ack_without_an_active_request_is_rejected():
    model = controller(start_request=False)
    with pytest.raises(ValueError, match="stale request"):
        model.reset_complete(None, 0, "cuda")


@pytest.mark.parametrize("owner", ["cuda", "rank0", "rank1"])
def test_cache_reset_work_retains_owners_through_cancel_and_shutdown(owner):
    model = controller(start_request=False)
    generation = model.begin_request("request", 131072)
    with pytest.raises(RuntimeError, match="lacks its started"):
        model.reset_complete("request", generation, owner)
    model.reset_started("request", generation, owner)
    with pytest.raises(RuntimeError, match="overlaps"):
        model.reset_started("request", generation, owner)
    model.begin_drain(now=0, timeout=30, cancelled=True)
    assert not model.poll_drain(1)
    with pytest.raises(RuntimeError, match="still owns"):
        model.close_request()
    with pytest.raises(RuntimeError, match="matching"):
        unmap(model, 0)
    model.reset_complete("request", generation, owner)
    assert model.poll_drain(2)
    model.resume_after_cancel(3)
    begin(model, request="next")


def test_closed_receipt_does_not_advertise_repeated_allocation_cleanup():
    model = controller()
    model.begin_drain(now=0, timeout=30)
    unmap(model, 0)
    unmap(model, 1)
    assert model.snapshot()["owners_releasable"]
    model.release_owners()
    assert not model.snapshot()["owners_releasable"]


def test_drain_does_not_start_new_cache_reset_work_after_mapping_release():
    model = controller(start_request=False)
    generation = model.begin_request("request", 131072)
    model.begin_drain(now=0, timeout=30)
    unmap(model, 0)
    with pytest.raises(RuntimeError, match="reset overlaps"):
        model.reset_started("request", generation, "rank0")
