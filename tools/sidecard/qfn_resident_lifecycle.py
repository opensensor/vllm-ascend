# SPDX-License-Identifier: Apache-2.0
"""Cold graph replacement without closing custom collective allocations."""


def reset_compact_mamba_tables(runner):
    """Remove scheduler IDs before a drained compact-state graph recapture.

    Removing requests leaves block-table storage intact. Dummy capture bypasses
    the normal scheduler-to-compact-slot remapping, so stale global block IDs
    can index beyond the compact state pool. Null slots match cold capture;
    the ordinary resident reset zeros state again after capture.
    """
    if not getattr(runner, "supports_compact_mamba_state", False):
        return
    if runner.requests or any(req is not None for req in runner.input_batch.req_ids):
        raise RuntimeError("drain requests before resetting compact Mamba tables")
    if getattr(runner, "execute_model_state", None) is not None:
        raise RuntimeError("finish model execution before resetting compact Mamba tables")
    for table in runner.input_batch.block_table.block_tables:
        if table.is_mamba_group:
            table.block_table.np.fill(0)
            table.block_table.gpu.zero_()
            table.num_blocks_per_row.fill(0)


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


def renew_collective_graphs(facade, wrappers, *, synchronize, clear_state, observer_factory, eager_wrappers=()):
    """Called only after the controller drains inference on every rank.

    Reset every owned native graph and release its leases before clearing Python
    caches. The communicator and its imported buffers remain allocated. Failure
    propagates; callers must keep inference paused and retain the old observer.
    """
    observer = facade.graph_observer
    if facade.state != "active" or observer is None or observer.retired or observer.track_replays:
        raise RuntimeError("resident reload requires an active unobserved graph collective")
    eager_ids = {id(wrapper) for wrapper in eager_wrappers}
    if len(eager_ids) != len(eager_wrappers) or not eager_ids.issubset({id(wrapper) for wrapper in wrappers}):
        raise RuntimeError("eager graph exclusions must identify distinct resident wrappers")
    if any(
        wrapper.__dict__.get("_collective_graph_observer") is not None or wrapper.entries for wrapper in eager_wrappers
    ):
        raise RuntimeError("an eager wrapper must own neither captures nor a collective observer")
    managed = [wrapper for wrapper in wrappers if id(wrapper) not in eager_ids]
    if not managed or any(wrapper.__dict__.get("_collective_graph_observer") is not observer for wrapper in managed):
        raise RuntimeError("resident wrappers do not share the current collective observer")
    captured = {
        id(entry.capture) for wrapper in managed for entry in wrapper.entries.values() if entry.capture is not None
    }
    if captured != set(observer._records):
        raise RuntimeError("unmanaged or failed captures prevent resident graph replacement")
    receipt = observer.retire(synchronize)
    if not receipt["retired"] or receipt["native_leases_acquired"] != receipt["native_leases_released"]:
        raise RuntimeError("native graph retirement did not release all capture leases")
    clear_state(wrappers)
    replacement = observer_factory(facade, track_replays=False)
    facade.graph_observer = replacement
    for wrapper in managed:
        wrapper._collective_graph_observer = replacement
        wrapper._collective_graph_tracking = False
    return receipt
