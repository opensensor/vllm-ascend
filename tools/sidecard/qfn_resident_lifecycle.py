# SPDX-License-Identifier: Apache-2.0
"""Cold graph replacement without closing custom collective allocations."""


def reset_prefix_tier(tier):
    """Invalidate the qualified legacy tier without reallocating its storage."""
    if callable(getattr(tier, "reset", None)):
        tier.reset()
        return
    required = (
        "_resident",
        "_device_archive_resident",
        "_host",
        "_unused_slots",
        "_unused_device_archive_slots",
        "num_slots",
        "device_archive_slots",
        "layer_states",
    )
    if any(not hasattr(tier, name) for name in required):
        raise RuntimeError("unsupported prefix Mamba tier reset contract")
    for states in tier.layer_states:
        for state in states:
            state.zero_()
    tier._resident.clear()
    tier._device_archive_resident.clear()
    tier._host.clear()
    tier._unused_slots[:] = range(tier.num_slots - 1, 0, -1)
    tier._unused_device_archive_slots[:] = range(tier.device_archive_slots - 1, -1, -1)


def renew_collective_graphs(facade, wrappers, *, synchronize, clear_state, observer_factory):
    """Called only after the controller drains inference on every rank.

    Reset every owned native graph and release its leases before clearing Python
    caches. The communicator and its imported buffers remain allocated. Failure
    propagates; callers must keep inference paused and retain the old observer.
    """
    observer = facade.graph_observer
    if facade.state != "active" or observer is None or observer.retired or observer.track_replays:
        raise RuntimeError("resident reload requires an active unobserved graph collective")
    if not wrappers or any(wrapper.__dict__.get("_collective_graph_observer") is not observer for wrapper in wrappers):
        raise RuntimeError("resident wrappers do not share the current collective observer")
    captured = {
        id(entry.capture) for wrapper in wrappers for entry in wrapper.entries.values() if entry.capture is not None
    }
    if captured != set(observer._records):
        raise RuntimeError("unmanaged or failed captures prevent resident graph replacement")
    receipt = observer.retire(synchronize)
    if not receipt["retired"] or receipt["native_leases_acquired"] != receipt["native_leases_released"]:
        raise RuntimeError("native graph retirement did not release all capture leases")
    clear_state(wrappers)
    replacement = observer_factory(facade, track_replays=False)
    facade.graph_observer = replacement
    for wrapper in wrappers:
        wrapper._collective_graph_observer = replacement
        wrapper._collective_graph_tracking = False
    return receipt
