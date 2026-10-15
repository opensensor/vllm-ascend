# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tools.sidecard.qfn_resident_lifecycle import renew_collective_graphs, reset_prefix_tier


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
