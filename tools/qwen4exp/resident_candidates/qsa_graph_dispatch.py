# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BLOCKED experimental QSA prefix capture: resident boundary gate failed.

Targets the qualified qfn-pinned-position-runtime wrapper (which inherits its
__call__ unchanged). Install only while requests are paused and existing graphs
have been retired. No native replay submission counters or observer hooks are
introduced. Parent/controller owns serialization across all TP ranks.
The default factory is blocked pending a resident transition probe exercising
the exact non-default-stream window below. CPU tests cannot qualify that gate.
"""

import inspect
import textwrap
from collections import namedtuple
from contextlib import contextmanager
from types import FunctionType

import torch

from tools.qwen4exp.resident_candidates.qsa_graph_selection import prefix_selection

RESIDENT_RECAPTURE_WINDOW_QUALIFIED = False
TARGET_MODEL_CLASSES = frozenset(("AscendQwen4ExpForConditionalGeneration", "AscendQwen4ExpForCausalLM"))
SCRATCH_CALLBACKS = frozenset(
    (
        "_PLEInjection.forward.<locals>.run_ple_eager",
        "_QSAAttention.forward.<locals>.select_current_groups",
        # Source factory compiles the qualified method outside its class.
        "forward.<locals>.select_current_groups",
    )
)


def is_target_wrapper(wrapper):
    runnable = wrapper.runnable
    root = runnable if callable(getattr(runnable, "modules", None)) else getattr(runnable, "__self__", None)
    return type(root).__name__ in TARGET_MODEL_CLASSES


def validate_target_capture(capture):
    """Only qualified target scratch overwrites may execute twice at recapture.

    PLE hashes/gathers immutable rows and overwrites its output; its convolution
    is stateless. QSA selection reads caches and overwrites four scratch tensors.
    Unknown eager callbacks, including draft expert dispatch, fail closed.
    """
    graphs = eager = 0
    for segment in capture.segments:
        owner = getattr(segment, "__self__", None)
        if type(owner).__name__ == "NPUGraph" and getattr(segment, "__name__", None) == "replay":
            graphs += 1
        elif (
            getattr(segment, "__module__", None) == "vllm_ascend.models.qwen4_exp.model"
            and getattr(segment, "__qualname__", None) in SCRATCH_CALLBACKS
        ):
            eager += 1
        else:
            raise RuntimeError("unreviewed eager callback prevents target boundary recapture")
    if graphs != capture.num_graphs or eager != capture.num_eager_breaks:
        raise RuntimeError("target capture segment inventory is inconsistent")


def require_empty_graph_parameters():
    """Reject global attention task ownership before replacing one capture."""
    from vllm_ascend.compilation.acl_graph import (
        get_draft_graph_params,
        get_draft_graph_prefill_params,
        get_graph_params,
    )

    for getter in (get_graph_params, get_draft_graph_params, get_draft_graph_prefill_params):
        validate_empty_graph_parameters(getter())


def validate_empty_graph_parameters(parameters):
    if parameters is not None:
        if any(
            values for name in ("events", "handles", "attn_params") for values in getattr(parameters, name).values()
        ):
            raise RuntimeError("global attention tasks prevent isolated target capture replacement")
        if any(value is not None for value in parameters.workspaces.values()):
            raise RuntimeError("global attention workspaces prevent isolated target capture replacement")


def prefix_inventory(self):
    """Cold controller evidence; no model addresses or hot replay counters."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    from tools.qwen4exp.resident_candidates.qsa_admission_audit import inventory_with_audit

    inventory = inventory_with_audit(self)
    inventory.pop("qsa_admission_audit", None)
    inventory["qsa_prefix_dispatch"] = []
    for wrapper in tuple(BreakableCUDAGraphWrapper._all_instances):
        dispatcher = wrapper.__dict__.get("_qfn_qsa_prefix_dispatch")
        if dispatcher is not None:
            inventory["qsa_prefix_dispatch"].append(
                {
                    "model": type(wrapper.runnable).__name__,
                    "failed": dispatcher.failed,
                    "contracts": [
                        {"descriptor": str(descriptor), "mode": contract.mode}
                        for descriptor, contract in dispatcher.contracts.items()
                    ],
                    "retirements": list(dispatcher.retirement_receipts),
                }
            )
    return inventory


class PrefixGeometry(namedtuple("PrefixGeometryBase", "compress_ratio token_topk max_context")):
    """Immutable geometry without a sys.modules-dependent dataclass loader."""

    __slots__ = ()

    def __new__(cls, compress_ratio=4, token_topk=2048, max_context=262144):
        if min(compress_ratio, token_topk, max_context) <= 0 or token_topk % compress_ratio:
            raise ValueError("invalid QSA prefix geometry")
        return super().__new__(cls, compress_ratio, token_topk, max_context)

    @property
    def max_prefix_length(self):
        return self.token_topk + self.compress_ratio - 1


class Admission(namedtuple("AdmissionBase", "prefix_eligible binding")):
    __slots__ = ()


class CapturedContract(namedtuple("CapturedContractBase", "capture mode binding")):
    __slots__ = ()


def _cpu_lengths(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("QSA admission requires scheduler-owned CPU lengths")
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA admission lengths must be one-dimensional integers")
    return tuple(value.tolist())


def _tensor_binding(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("QSA graph metadata must contain device tensors")
    return value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device


def inspect_admission(context, geometries, args, kwargs):
    """Check values only on CPU; inspect device addresses/geometry without sync.

    Metadata identity itself is not required to be stable: scheduler objects
    can change while their device buffers retain the same addresses. Shared
    metadata objects are inspected once, regardless of QSA layer count.
    """
    metadata_by_layer = getattr(context, "attn_metadata", None)
    if not isinstance(metadata_by_layer, dict):
        raise ValueError("QSA graph admission requires layer metadata")
    seen, layers = {}, []
    eligible = True
    for prefix, geometry in geometries.items():
        if prefix not in metadata_by_layer:
            raise ValueError(f"missing QSA metadata for {prefix}")
        metadata = metadata_by_layer[prefix]
        cached = seen.get(id(metadata))
        if cached is None:
            lengths = _cpu_lengths(getattr(metadata, "seq_lens_cpu", None))
            queries = _cpu_lengths(getattr(metadata, "query_lens_cpu", None))
            if not lengths or len(lengths) != len(queries) or any(q < 0 for q in queries):
                raise ValueError("invalid packed QSA metadata")
            tokens = getattr(metadata, "num_actual_tokens", None)
            if type(tokens) is not int or tokens <= 0 or sum(queries) != tokens:
                raise ValueError("packed QSA lengths do not cover the captured token geometry")
            state = getattr(getattr(metadata, "attn_state", None), "name", "")
            shape = (tokens, queries, getattr(metadata, "num_prefills", 0), getattr(metadata, "num_decodes", 0), state)
            buffers = tuple(
                (name, _tensor_binding(getattr(metadata, name, None)))
                for name in ("block_tables", "query_start_loc", "slot_mapping", "seq_lens")
            )
            cached = lengths, shape, buffers
            seen[id(metadata)] = cached
        lengths, shape, buffers = cached
        if any(length < 0 or length > geometry.max_context for length in lengths):
            raise ValueError("QSA sequence length exceeds the configured context")
        eligible = eligible and shape[2] == 0 and max(lengths) <= geometry.max_prefix_length
        layers.append((prefix, geometry, shape, buffers))
    if not layers:
        raise ValueError("QSA prefix dispatch needs explicit layer geometries")
    inputs = tuple(_tensor_binding(value) for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
    return Admission(eligible, (tuple(layers), inputs))


def retire_capture(observer, capture, synchronize):
    """Drain/reset one capture, release its native tickets, keep peers live.

    This cold path intentionally mirrors the qualified observer's ownership
    protocol. It removes a fully released record so later global retire cannot
    release the same tickets twice. Failure keeps the record and all remaining
    leases, quarantines owners, and forbids an automatic cleanup retry.
    """
    observer._require_live()
    if observer.track_replays:
        raise RuntimeError("QSA replacement requires unobserved collective replay")
    if observer.active_capture is not None or any(record.replay_pending for record in observer._records.values()):
        raise RuntimeError("cannot replace a capture while capture/submission is active")
    if not callable(synchronize):
        raise ValueError("capture retirement requires an explicit device drain")
    record = observer._record(capture)
    observer.collective._context()
    observer._retiring = True
    try:
        synchronize()
        if getattr(capture, "capture_token", None) is not None:
            raise RuntimeError("native graph still owns an active capture token")
        capture.reset()
        record.reset = True
        for name in ("tasks", "provider_sizes", "capture_resources"):
            metadata = getattr(capture, name, None)
            if metadata is not None:
                metadata.clear()
        for lease in record.leases:
            if not lease.released:
                lease.owner.release_graph_lease(lease.ticket)
                lease.released = True
        receipt = {
            "captures_reset": 1,
            "native_leases_released": len(record.leases),
            "other_captures": len(observer._records) - 1,
        }
        del observer._records[id(capture)]
        return receipt
    except BaseException:
        observer._quarantine(record)
        raise
    finally:
        observer._retiring = False


class PrefixDispatcher:
    """One wrapper, existing descriptors only; no additional graph variants.

    Prefix -> general is one-way until an explicit paused reset/reload. General
    contexts retain the original graph callback path. Address/packed-layout
    changes fail before submission and require coordinated reset; silently
    recapturing one rank on a pointer mismatch could diverge from its peers.
    """

    def __init__(self, geometries, *, observer=None):
        self.geometries = dict(geometries)
        self.observer = observer
        self.contracts = {}
        self.retirement_receipts = []
        self.failed = False
        # Retained for the entire dispatcher generation, including failure.
        # Native captures must be drained/reset before generation replacement.
        self.recapture_resources = None

    def call(
        self,
        wrapper,
        context,
        args,
        kwargs,
        invoke,
        retire,
        prepare_recapture=None,
        recapture_window=None,
        validate_capture=None,
    ):
        if self.failed:
            raise RuntimeError("QSA dispatcher requires coordinated reset after a failed transition")
        descriptor = context.batch_descriptor
        admission = inspect_admission(context, self.geometries, args, kwargs)
        entry = wrapper.entries.get(descriptor)
        capture = None if entry is None else entry.capture
        prior = self.contracts.get(descriptor)
        if capture is None and prior is not None:
            # The controller may have retired and cleared wrapper graphs.
            del self.contracts[descriptor]
            prior = None
        if capture is not None:
            if prior is None or prior.capture is not capture:
                raise RuntimeError("install QSA dispatch only after retiring existing graph entries")
            if prior.binding != admission.binding:
                raise RuntimeError("QSA input/metadata addresses or packed geometry changed; pause and reset all ranks")
        mode = (
            "general"
            if prior is not None and prior.mode == "general"
            else ("prefix" if admission.prefix_eligible else "general")
        )
        if prior is not None and mode == prior.mode:
            # Admission is the only hot host check. Native replay stays the
            # original call; no capture flags, counters or contracts mutate.
            return invoke()
        if prior is not None and mode != prior.mode and recapture_window is not None:
            with recapture_window():
                return self.call(
                    wrapper, context, args, kwargs, invoke, retire, prepare_recapture, validate_capture=validate_capture
                )
        boundary_recapture = prior is not None and mode != prior.mode
        if prior is not None and mode != prior.mode:
            # All ranks derive this one-way transition from the same scheduler
            # lengths. Caller serializes admission, drain and capture globally.
            if prepare_recapture is not None:
                prepare_recapture()
            if validate_capture is not None:
                validate_capture(capture)
            try:
                receipt = retire(capture)
                self.retirement_receipts.append(receipt)
                entry.capture, entry.output, entry.input_addresses = None, None, None
                del self.contracts[descriptor]
            except BaseException:
                self.failed = True
                raise
        attribute = "_qfn_qsa_prefix_capture"
        present = hasattr(context, attribute)
        previous = getattr(context, attribute, None)
        setattr(context, attribute, self.geometries if mode == "prefix" else None)
        try:
            result = invoke()
        except BaseException:
            self.failed = True
            raise
        finally:
            if present:
                setattr(context, attribute, previous)
            else:
                delattr(context, attribute)
        current = wrapper.entries.get(descriptor)
        if current is None or current.capture is None:
            self.failed = True
            raise RuntimeError("graph dispatch did not produce a captured entry")
        self.contracts[descriptor] = CapturedContract(current.capture, mode, admission.binding)
        if boundary_recapture:
            # Hardware probe: capture records graph ops without executing them;
            # only eager callbacks ran. Execute the new graph exactly once for
            # this real request. Initial startup capture preserves its original
            # no-replay behavior. Native replay tracking remains disabled.
            try:
                if validate_capture is not None:
                    validate_capture(current.capture)
                return invoke()
            except BaseException:
                self.failed = True
                raise
        return result


def prefix_selection_for_capture(context, prefix, logical_positions, *, compress_ratio, token_topk):
    """Model seam: only called while recording the selected prefix graph."""
    geometries = getattr(context, "_qfn_qsa_prefix_capture", None)
    geometry = None if geometries is None else geometries.get(prefix)
    if geometry is None or (compress_ratio, token_topk) != (geometry.compress_ratio, geometry.token_topk):
        raise RuntimeError("QSA prefix capture geometry was not admitted")
    return prefix_selection(
        logical_positions,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        capacity_groups=token_topk // compress_ratio,
    )


def qualified_wrapper_call(self, *args, **kwargs):
    """Replacement for the qualified Ascend wrapper's inherited __call__.

    Parent installs _qfn_qsa_prefix_dispatch while paused, before recapture.
    This function deliberately does not replace newer wrapper implementations.
    """
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    invoke = lambda: BreakableCUDAGraphWrapper.__call__(self, *args, **kwargs)
    if _EXTRA_CTX.is_draft_model or not is_target_wrapper(self):
        return invoke()
    if not is_forward_context_available():
        return invoke()
    context = get_forward_context()
    if context.cudagraph_runtime_mode == CUDAGraphMode.NONE or (
        self.runtime_mode is not None and self.runtime_mode != context.cudagraph_runtime_mode
    ):
        return invoke()
    observer = ACLGraphWrapper._resolve_collective_graph_observer(self)
    if observer is None or observer.track_replays:
        raise RuntimeError("prefix graph dispatch requires an installed unobserved collective owner")
    dispatcher = dispatcher_for_observer(self, observer)
    if dispatcher is None:
        return invoke()
    return dispatcher.call(
        self,
        context,
        args,
        kwargs,
        invoke,
        lambda capture: retire_capture(observer, capture, torch.npu.synchronize),
        prepare_recapture=lambda: (validate_cudagraph_capturing_enabled(), require_empty_graph_parameters()),
        recapture_window=lambda: serialized_recapture_window(dispatcher),
        validate_capture=validate_target_capture,
    )


def dispatcher_for_observer(wrapper, observer):
    """Cold generation renewal after resident graph/lease retirement.

    Same-observer calls only compare identity. A new observer alone is not proof
    of retirement: require the prior observer's completed native receipt and an
    empty wrapper cache before discarding old contracts or failure state.
    """
    if wrapper.__dict__.get("_qfn_qsa_prefix_skip_observer") is observer:
        return None
    dispatcher = wrapper.__dict__.get("_qfn_qsa_prefix_dispatch")
    previous = None if dispatcher is None else getattr(dispatcher, "observer", None)
    if dispatcher is not None and previous is observer:
        return dispatcher
    if wrapper.entries:
        raise RuntimeError("new QSA observer generation requires empty retired graph caches")
    if dispatcher is not None:
        if previous is None:
            raise RuntimeError("unbound legacy QSA dispatcher requires an explicit paused reset")
        receipt = previous.snapshot()
        if not receipt["retired"] or receipt["native_leases_acquired"] != receipt["native_leases_released"]:
            raise RuntimeError("previous QSA observer has not completed native lease retirement")
    observer._require_live()
    if observer.track_replays or observer.active_capture is not None:
        raise RuntimeError("new QSA observer must be live, unobserved and outside capture")
    geometries = discover_geometries(wrapper.runnable, wrapper.vllm_config.model_config.max_model_len)
    if not geometries:
        wrapper._qfn_qsa_prefix_skip_observer = observer
        return None
    if getattr(wrapper.vllm_config.scheduler_config, "async_scheduling", False):
        raise RuntimeError("QSA prefix replacement requires serialized synchronous workers")
    dispatcher = PrefixDispatcher(geometries, observer=observer)
    wrapper._qfn_qsa_prefix_dispatch = dispatcher
    return dispatcher


@contextmanager
def serialized_recapture_window(dispatcher):
    """Retained non-default capture stream with explicit producer/consumer order.

    The worker owns model-call serialization and the same scheduler transition
    on every rank. This is never enabled for async_scheduling in this candidate.
    Two ordinary events and one stream are reused across descriptor transitions,
    and retained until the generation's graphs/leases have been drained/reset.
    On failure, do not enqueue new work into a potentially failed capture stream;
    retain resources for quarantine/diagnosis and restore host context only.
    """
    from vllm.compilation import monitor

    if dispatcher.failed:
        raise RuntimeError("failed recapture window requires coordinated retirement")
    if dispatcher.recapture_resources is None:
        dispatcher.recapture_resources = (
            torch.npu.Stream(),
            torch.npu.Event(),
            torch.npu.Event(),
        )
    capture_stream, producer_ready, consumer_ready = dispatcher.recapture_resources
    producer_stream = torch.npu.current_stream()
    if capture_stream == torch.npu.default_stream() or capture_stream == producer_stream:
        raise RuntimeError("boundary recapture requires a distinct non-default stream")
    previous = monitor.cudagraph_capturing_enabled
    try:
        producer_ready.record(producer_stream)
        capture_stream.wait_event(producer_ready)
        with torch.npu.stream(capture_stream):
            monitor.set_cudagraph_capturing_enabled(True)
            yield
            consumer_ready.record(capture_stream)
        producer_stream.wait_event(consumer_ready)
    except BaseException:
        dispatcher.failed = True
        raise
    finally:
        monitor.set_cudagraph_capturing_enabled(previous)


def discover_geometries(runnable, max_context):
    """Inspect model modules once after a paused patch, not on each token."""
    root = runnable if callable(getattr(runnable, "modules", None)) else getattr(runnable, "__self__", None)
    if root is None or not callable(getattr(root, "modules", None)):
        return {}
    geometries = {}
    for layer in root.modules():
        if type(layer).__name__ == "_QSAAttention":
            geometries[layer.prefix] = PrefixGeometry(
                layer.indexer.compress_ratio, layer.indexer.token_topk, max_context
            )
    return geometries


def replacements():
    """Create an installable payload from the QUALIFIED runtime's own source.

    Only two exact seams are replaced. Fail closed when the expected source is
    absent instead of copying the newer local model into the resident process.
    Compilation is isolated; the installed method retains the original globals
    so later resident replacements of module helpers remain visible.
    """
    if not RESIDENT_RECAPTURE_WINDOW_QUALIFIED:
        raise RuntimeError(
            "QSA prefix factory blocked: default-stream boundary capture failed; "
            "the exact resident recapture window requires hardware qualification"
        )
    return {
        "vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward": build_qualified_forward(),
        "vllm_ascend.compilation.breakable_aclgraph:BreakableACLGraphWrapper.__call__": qualified_wrapper_call,
        "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_graph_inventory": prefix_inventory,
    }


def build_qualified_forward():
    """Build the shared source seam without authorizing either dispatch policy."""
    from vllm_ascend.models.qwen4_exp import model

    original = getattr(model._QSAAttention.forward, "_qfn_qsa_base_forward", model._QSAAttention.forward)
    source = textwrap.dedent(inspect.getsource(original))
    projection = "needs_sparse_index_query = self._needs_sparse_index_query(metadata, self.indexer.token_topk)"
    replacement_projection = (
        projection
        + """
    if (getattr(get_forward_context(), "_qfn_qsa_prefix_capture", None) is not None
            or getattr(get_forward_context(), "_qfn_qsa_force_sparse_capture", False)):
        needs_sparse_index_query = True"""
    )
    selection = """        capture = BreakableCUDAGraphCapture.current()
        if capture is not None and capture._capturing:
            # A graph captured for the first decode step cannot freeze"""
    replacement_selection = """        capture = BreakableCUDAGraphCapture.current()
        prefix_geometries = getattr(get_forward_context(), "_qfn_qsa_prefix_capture", None)
        if capture is not None and capture._capturing and prefix_geometries is not None:
            from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import prefix_selection_for_capture

            selection = prefix_selection_for_capture(
                get_forward_context(), self.prefix, logical_positions,
                compress_ratio=self.indexer.compress_ratio, token_topk=self.indexer.token_topk,
            )
        elif capture is not None and capture._capturing:
            # A graph captured for the first decode step cannot freeze"""
    if source.count(projection) != 1 or source.count(selection) != 1:
        raise RuntimeError("qualified QSA source seams do not match; retain the previous resident implementation")
    source = source.replace(projection, replacement_projection).replace(selection, replacement_selection)
    namespace = dict(model.__dict__)
    exec(compile(source, "<qsa-prefix-qualified-forward>", "exec"), namespace)
    compiled = namespace["forward"]
    forward = FunctionType(compiled.__code__, original.__globals__, original.__name__, compiled.__defaults__)
    forward.__kwdefaults__ = compiled.__kwdefaults__
    forward.__annotations__ = compiled.__annotations__
    forward.__qualname__ = original.__qualname__
    forward._qfn_qsa_base_forward = original
    return forward
