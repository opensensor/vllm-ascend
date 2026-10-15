# SPDX-License-Identifier: Apache-2.0
"""Lazy worker RPCs for the explicitly enabled QFN collective harness.

Status reads host metadata only. Before engine shutdown, invoke close through
an all-worker collective RPC so every TP rank participates in the facade's
drain/import-closure protocol. The caller must first retire eager work, graphs,
and queued handlers. Errors propagate while the communicator retains owners;
this extension never shuts down the engine or frees resources itself.
"""

FACADE_STATES = frozenset({"new", "active", "closed", "quarantined"})
GRAPH_COUNTERS = frozenset(
    {
        "registered_graphs",
        "captured_graphs",
        "failed_graphs",
        "replay_submissions",
        "captured_calls",
        "replayed_calls",
        "reset_graphs",
        "native_leases_acquired",
        "native_leases_released",
    }
)
FALLBACK_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "torch.float16",
        "torch.float32",
        "torch.bfloat16",
        "torch.float64",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.uint8",
        "torch.bool",
        "str",
        "list",
        "tuple",
        "dict",
        "int",
        "float",
        "bool",
        "NoneType",
    }
)


def _communicator():
    # Worker isolation: importing this extension must not import vLLM or Torch.
    from vllm.distributed import get_tp_group

    return get_tp_group().device_communicator


def _counter(value):
    if type(value) is not int or value < 0:
        raise ValueError("collective status requires exact nonnegative integer counters")
    return value


def _identity(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("collective status identity must be lowercase SHA256")
    return value


def validate_graph_snapshot(snapshot):
    """Copy the bounded public observer schema; reject unknown/private data."""
    flags = {"active_capture", "retired", "lifecycle_only", "tracking_enabled"}
    if (
        type(snapshot) is not dict
        or set(snapshot) != GRAPH_COUNTERS | flags | {"count_semantics"}
        or any(type(snapshot[name]) is not bool for name in flags)
        or snapshot["lifecycle_only"] is not True
    ):
        raise ValueError("graph status requires sanitized lifecycle counters and exact boolean flags")
    semantics = "capture-and-replay-submissions" if snapshot["tracking_enabled"] else "capture-only"
    if type(snapshot["count_semantics"]) is not str or snapshot["count_semantics"] != semantics:
        raise ValueError("graph count semantics differ from explicit tracking mode")
    if not snapshot["tracking_enabled"] and (snapshot["replay_submissions"] != 0 or snapshot["replayed_calls"] != 0):
        raise ValueError("untracked graph replays must not present counted replay submissions")
    return dict(
        {name: _counter(snapshot[name]) for name in GRAPH_COUNTERS},
        **{name: snapshot[name] for name in flags},
        count_semantics=semantics,
    )


def _graph_status(observer):
    return validate_graph_snapshot(observer.snapshot())


def _clear_graph_wrapper_caches(observer):
    # Lazy worker-only import: the eager/baseline extension remains standalone.
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    from vllm_ascend.compilation import acl_graph

    for wrapper in tuple(acl_graph._acl_graph_wrappers):
        if wrapper.__dict__.get("_collective_graph_observer") is observer:
            wrapper.concrete_aclgraph_entries.clear()
            wrapper.first_run_finished = False
    for wrapper in tuple(BreakableCUDAGraphWrapper._all_instances):
        if wrapper.__dict__.get("_collective_graph_observer") is observer:
            wrapper.clear_graphs()


def _retire_graph_observer(worker, facade):
    observer = getattr(facade, "graph_observer", None)
    if observer is None:
        return
    # ACLGraphWrapper capture/replay is the admitted observer path. Reject
    # direct runner graph-manager caches rather than claiming they retired.
    runners, seen = [getattr(worker, "model_runner", None)], set()
    while runners:
        runner = runners.pop()
        if runner is None or id(runner) in seen:
            continue
        seen.add(id(runner))
        for name in ("cudagraph_manager", "encoder_cudagraph_manager"):
            manager = getattr(runner, name, None)
            if manager is not None and getattr(manager, "graphs", None):
                raise RuntimeError("unmanaged runner graphs prevent custom collective retirement")
        runners.extend((getattr(runner, "drafter", None), getattr(runner, "model_runner", None)))
    observer.retire(lambda: facade.torch.npu.synchronize(facade.device_index))
    _clear_graph_wrapper_caches(observer)


def _status(communicator):
    facade = getattr(communicator, "custom_sum", None)
    result = dict(schema_version=1, enabled=facade is not None)
    if facade is None:
        return dict(result, state="disabled", geometries=[], fallback_schemas=[])
    members = facade.members
    if (
        type(members) not in (tuple, list)
        or not members
        or any(type(rank) is not int or rank < 0 for rank in members)
        or len(set(members)) != len(members)
        or type(facade.rank) is not int
        or facade.rank not in members
    ):
        raise ValueError("collective status requires exact unique group members and rank")
    if type(facade.state) is not str or facade.state not in FACADE_STATES:
        raise ValueError("unknown collective status state")
    geometries = []
    for key, owner in facade.owners.items():
        if (
            type(key) is not tuple
            or len(key) != 2
            or type(key[0]) is not str
            or key[0] not in ("float16", "float32")
            or type(key[1]) is not int
            or key[1] <= 0
            or (owner.schedule.dtype, owner.schedule.elements) != key
        ):
            raise ValueError("invalid fixed collective status geometry")
        geometries.append(
            dict(
                dtype=key[0],
                elements=key[1],
                schedule_sha256=_identity(owner.schedule.identity),
                calls=_counter(owner.calls),
            )
        )
    fallbacks = []
    for key, calls in facade.fallbacks.items():
        if (
            type(key) is not tuple
            or len(key) != 3
            or type(key[0]) is not str
            or key[1] is not None
            and (type(key[1]) is not int or key[1] < 0)
            or key[2] is not None
            and type(key[2]) is not bool
        ):
            raise ValueError("invalid collective fallback status geometry")
        fallbacks.append(
            dict(
                dtype=key[0] if key[0] in FALLBACK_DTYPES else "unsupported",
                elements=key[1],
                contiguous=key[2],
                calls=_counter(calls),
            )
        )
    status = dict(
        result,
        state=facade.state,
        rank=facade.rank,
        members=list(members),
        config_sha256=_identity(facade.identity),
        geometries=geometries,
        fallback_schemas=fallbacks,
    )
    observer = getattr(facade, "graph_observer", None)
    if observer is not None:
        status["graph_observer"] = _graph_status(observer)
    return status


class QfnCollectiveWorkerExtension:
    """Mixin supplied as worker_extension_cls by the offline QFN harness."""

    def npu_collective_status(self):
        """Return copied JSON metadata, excluding paths, IPC keys and handles."""
        return _status(_communicator())

    def npu_collective_close(self):
        """All-rank RPC before engine shutdown; never swallow close errors."""
        communicator = _communicator()
        if getattr(communicator, "custom_sum", None) is not None:
            _retire_graph_observer(self, communicator.custom_sum)
            communicator.close_custom_collective()
            result = _status(communicator)
            if result["state"] != "closed":
                raise RuntimeError("custom collective close did not establish closed ownership")
            return result
        return _status(communicator)
