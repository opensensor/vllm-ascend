# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.sidecard.qfn_resident_lifecycle import renew_collective_graphs, reset_compact_mamba_tables, reset_prefix_tier


def test_recapture_discards_global_ids_without_reallocating_compact_tables():
    def table(is_mamba):
        return SimpleNamespace(
            is_mamba_group=is_mamba,
            block_table=SimpleNamespace(np=np.full((6, 4), 132, dtype=np.int32), gpu=torch.full((6, 4), 93)),
            num_blocks_per_row=np.full(6, 4, dtype=np.int32),
        )

    attention, mamba = table(False), table(True)
    runner = SimpleNamespace(
        supports_compact_mamba_state=True,
        requests={},
        execute_model_state=None,
        input_batch=SimpleNamespace(req_ids=[], block_table=SimpleNamespace(block_tables=[attention, mamba])),
    )
    host, device = mamba.block_table.np, mamba.block_table.gpu
    reset_compact_mamba_tables(runner)
    assert mamba.block_table.np is host and mamba.block_table.gpu is device
    assert not host.any() and not device.any() and not mamba.num_blocks_per_row.any()
    assert (attention.block_table.np == 132).all() and (attention.block_table.gpu == 93).all()
    # Simulate dummy capture committing host IDs to a 105-slot state pool.
    state = torch.zeros(105, 2)
    state[torch.from_numpy(host[:, 0]).long()] = 1
    assert state[0].tolist() == [1, 1] and not state[1:].any()


@pytest.mark.parametrize("live", ["requests", "batch", "execution"])
def test_compact_table_reset_refuses_live_ownership(live):
    runner = SimpleNamespace(
        supports_compact_mamba_state=True,
        requests={},
        execute_model_state=None,
        input_batch=SimpleNamespace(req_ids=[]),
    )
    if live == "requests":
        runner.requests["request"] = object()
    elif live == "batch":
        runner.input_batch.req_ids.append("request")
    else:
        runner.execute_model_state = object()
    with pytest.raises(RuntimeError):
        reset_compact_mamba_tables(runner)


def test_noncompact_runner_does_not_require_mamba_tables():
    reset_compact_mamba_tables(SimpleNamespace(supports_compact_mamba_state=False))


def fixture_objects(failure=None):
    events = []
    graphs = [object(), object()]
    observer = SimpleNamespace(retired=False, track_replays=False, _records={id(graph): graph for graph in graphs})

    def retire(synchronize):
        synchronize()
        events.append("native-reset-and-release")
        if failure == "retire":
            raise RuntimeError("native reset failed")
        observer.retired = True
        return dict(retired=True, native_leases_acquired=28, native_leases_released=28)

    observer.retire = retire
    facade = SimpleNamespace(state="active", graph_observer=observer, owners=object())
    wrappers = [
        SimpleNamespace(_collective_graph_observer=observer, entries={3: SimpleNamespace(capture=graph)})
        for graph in graphs
    ]

    def clear(wrappers):
        events.append("clear-and-renew-pool")
        if failure == "clear":
            raise RuntimeError("graph pool failed")
        for wrapper in wrappers:
            wrapper.entries.clear()

    def factory(owner, *, track_replays):
        events.append("new-observer")
        assert owner is facade and track_replays is False
        return SimpleNamespace(retired=False)

    options = dict(synchronize=lambda: events.append("drain"), clear_state=clear, observer_factory=factory)
    return facade, wrappers, observer, events, options


def test_reload_releases_graphs_before_rebinding_and_retains_transport():
    facade, wrappers, old, events, options = fixture_objects()
    owners = facade.owners
    receipt = renew_collective_graphs(facade, wrappers, **options)
    assert events == ["drain", "native-reset-and-release", "clear-and-renew-pool", "new-observer"]
    assert receipt["native_leases_released"] == 28
    assert facade.state == "active" and facade.owners is owners
    assert facade.graph_observer is not old
    assert all(wrapper._collective_graph_observer is facade.graph_observer for wrapper in wrappers)
    assert all(wrapper._collective_graph_tracking is False and not wrapper.entries for wrapper in wrappers)


def test_eager_draft_stays_unbound_while_target_graphs_are_renewed():
    facade, wrappers, old, events, options = fixture_objects()
    eager = SimpleNamespace(entries={})
    wrappers.append(eager)
    renew_collective_graphs(facade, wrappers, eager_wrappers=(eager,), **options)
    assert facade.graph_observer is not old
    assert all(w._collective_graph_observer is facade.graph_observer for w in wrappers[:-1])
    assert "_collective_graph_observer" not in vars(eager)
    assert not eager.entries
    assert events == ["drain", "native-reset-and-release", "clear-and-renew-pool", "new-observer"]


@pytest.mark.parametrize("reason", ["observer", "capture", "empty_entry", "unknown", "duplicate", "all"])
def test_eager_exclusion_cannot_hide_graph_ownership(reason):
    facade, wrappers, old, events, options = fixture_objects()
    eager = SimpleNamespace(entries={})
    wrappers.append(eager)
    excluded = (eager,)
    if reason == "observer":
        eager._collective_graph_observer = old
    elif reason in ("capture", "empty_entry"):
        eager.entries[3] = SimpleNamespace(capture=object() if reason == "capture" else None)
    elif reason == "unknown":
        excluded = (SimpleNamespace(entries={}),)
    elif reason == "duplicate":
        excluded = (eager, eager)
    elif reason == "all":
        wrappers = [eager]
    with pytest.raises(RuntimeError):
        renew_collective_graphs(facade, wrappers, eager_wrappers=excluded, **options)
    assert not events and facade.graph_observer is old


@pytest.mark.parametrize("failure", ["retire", "clear"])
def test_failure_retains_old_observer_and_transport(failure):
    facade, wrappers, old, events, options = fixture_objects(failure)
    owners = facade.owners
    with pytest.raises(RuntimeError):
        renew_collective_graphs(facade, wrappers, **options)
    assert facade.graph_observer is old and facade.owners is owners
    assert "new-observer" not in events
    assert all(wrapper._collective_graph_observer is old for wrapper in wrappers)


@pytest.mark.parametrize("reason", ["tracked", "retired", "quarantined", "unmanaged", "different_observer"])
def test_invalid_ownership_rejected_before_drain(reason):
    facade, wrappers, old, events, options = fixture_objects()
    if reason == "tracked":
        old.track_replays = True
    elif reason == "retired":
        old.retired = True
    elif reason == "quarantined":
        facade.state = "quarantined"
    elif reason == "unmanaged":
        old._records[123] = object()
    else:
        wrappers[0]._collective_graph_observer = object()
    with pytest.raises(RuntimeError):
        renew_collective_graphs(facade, wrappers, **options)
    assert events == []
    assert facade.graph_observer is old


def test_legacy_tier_reset_invalidates_slots_without_replacing_storage():
    events = []
    tensor = SimpleNamespace(zero_=lambda: events.append("zero"))
    tier = SimpleNamespace(
        layer_states=((tensor,),),
        num_slots=4,
        device_archive_slots=2,
        _resident={123: 1},
        _device_archive_resident={124: 0},
        _host={125: object()},
        _unused_slots=[2, 3],
        _unused_device_archive_slots=[1],
    )
    slots = tier._unused_slots
    archive_slots = tier._unused_device_archive_slots
    reset_prefix_tier(tier)
    assert tier.layer_states[0][0] is tensor and events == ["zero"]
    assert not tier._resident and not tier._device_archive_resident and not tier._host
    assert tier._unused_slots is slots and slots == [3, 2, 1]
    assert tier._unused_device_archive_slots is archive_slots and archive_slots == [1, 0]


def test_unknown_tier_fails_before_reset():
    with pytest.raises(RuntimeError, match="unsupported"):
        reset_prefix_tier(SimpleNamespace())
