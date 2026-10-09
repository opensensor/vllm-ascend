# SPDX-License-Identifier: Apache-2.0
"""Bounded eager MoE communication schedule with explicit rank agreement.

The default 2560-token geometry matches the reference. Smaller chunks are
separate accuracy/performance candidates, not automatic improvements. This module
imports no tensor/device runtime. Default-stream waits/copies and allocator stream
ownership are supplied explicitly; CPU tests cannot prove physical overlap.
"""

from collections import deque
from dataclasses import asdict, dataclass

from tools.qwen4exp.streaming_protocol import canonical_sha256

MIN_CHUNK_TOKENS = 128
MAX_CHUNK_TOKENS = 2560
RANKS = (0, 1, 2, 3)


@dataclass(frozen=True)
class SchedulePolicy:
    chunk_tokens: int = MAX_CHUNK_TOKENS
    max_inflight: int = 2
    shared_policy: str = "tp_sharded"

    def __post_init__(self):
        if type(self.chunk_tokens) is not int or not MIN_CHUNK_TOKENS <= self.chunk_tokens <= MAX_CHUNK_TOKENS:
            raise ValueError("chunk_tokens must be in [128, 2560]")
        if type(self.max_inflight) is not int or self.max_inflight not in (1, 2):
            raise ValueError("max_inflight must be one or two")
        if self.shared_policy not in ("tp_sharded", "replicated", "none"):
            raise ValueError("unsupported shared expert placement")


@dataclass(frozen=True)
class SchedulePlan:
    policy: SchedulePolicy
    generation: int
    source_sha256: str
    arithmetic_sha256: str
    reference_sha256: str
    max_input_tokens: int = MAX_CHUNK_TOKENS
    tp_size: int = 4

    def __post_init__(self):
        if not isinstance(self.policy, SchedulePolicy):
            raise ValueError("schedule plan requires an immutable validated policy")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be nonnegative")
        if type(self.tp_size) is not int or self.tp_size != len(RANKS):
            raise ValueError("this schedule requires four ranks")
        if type(self.max_input_tokens) is not int or self.max_input_tokens <= 0:
            raise ValueError("maximum input token capacity must be positive")
        for value in (self.source_sha256, self.arithmetic_sha256, self.reference_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("schedule identity must contain lowercase SHA256 hashes")

    @property
    def sha256(self):
        return canonical_sha256(asdict(self))

    def chunks(self, tokens):
        if type(tokens) is not int or not 0 <= tokens <= self.max_input_tokens:
            raise ValueError("input token count exceeds agreed rank capacity")
        size = self.policy.chunk_tokens
        return tuple((start, min(start + size, tokens)) for start in range(0, tokens, size))


def validate_rank_plan(plan, receipts):
    """Verify saved rank/config/generation agreement before any collective.

    Receipts are gathered/verified by admission, not RPC on every forward. Exact
    tensor row equality across TP ranks remains the existing scheduler contract;
    a static receipt does not inspect or prove future runtime tensor values/shapes.
    """
    receipts = tuple(receipts)
    ranks = [receipt.get("rank") for receipt in receipts]
    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(RANKS):
        raise ValueError("schedule admission needs exactly ranks 0-3")
    namespaces = set()
    for receipt in receipts:
        if receipt.get("plan_sha256") != plan.sha256 or receipt.get("generation") != plan.generation:
            raise ValueError("rank chunk/config/generation plan mismatch")
        if type(receipt.get("pid")) is not int or receipt["pid"] <= 0:
            raise ValueError("rank worker identity missing")
        namespace = receipt.get("execution_namespace")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("rank execution namespace missing")
        namespaces.add(namespace)
    if len(namespaces) != 1:
        raise ValueError("rank execution namespaces differ")
    return receipts


class PoisonedSchedule(RuntimeError):
    """No more submissions permitted after partial/unresolved invocation failure."""

    def __init__(self, message, *, unresolved_handles, submission_uncertain):
        super().__init__(message)
        self.unresolved_handles = unresolved_handles
        self.submission_uncertain = submission_uncertain


def _fp32(value, shape):
    if tuple(value.shape) != tuple(shape) or str(value.dtype) not in ("float32", "torch.float32"):
        raise ValueError("schedule requires exact-shape FP32 local/reduced/output storage")


class ScheduledMoE:
    """One admitted plan; two scratch buffers exist only during each invocation.

    Callbacks: route(inputs)->(weights,ids); local(chunk,w,ids)->FP32 routed;
    shared(chunk)->FP32 shared; submit_reduce(local,slot)->(reduced,completion);
    wait(completion) establishes an ordered dependency or raises; store(dst,src)
    enqueues ordered main-stream copy; allocate_output(inputs) and
    allocate_slot(inputs,chunk_tokens) allocate FP32. combine_shared defaults to
    FP32 left-plus-right. Caller performs the final baseline dtype cast.

    The scheduler never stores a tensor buffer on the module across invocations.
    Allocator record_stream ownership must keep queued DMA sources alive after
    Python references drop. Capture/decode dispatch belongs to the outer candidate.
    """

    def __init__(
        self,
        plan,
        rank_receipts,
        *,
        route,
        local,
        shared,
        submit_reduce,
        wait,
        store,
        allocate_output,
        allocate_slot,
        combine_shared=None,
    ):
        self.plan = plan
        self.rank_receipts = validate_rank_plan(plan, rank_receipts)
        if plan.policy.shared_policy != "none" and not callable(shared):
            raise ValueError("configured shared policy requires its exact shared expert callback")
        self.route, self.local, self.shared = route, local, shared
        self.submit_reduce, self.wait, self.store = submit_reduce, wait, store
        self.allocate_output, self.allocate_slot = allocate_output, allocate_slot
        self.combine_shared = combine_shared or (lambda left, right: left + right)
        self.poisoned = False

    def run(self, inputs):
        if self.poisoned:
            raise PoisonedSchedule("schedule owner is poisoned", unresolved_handles=0, submission_uncertain=True)
        if len(inputs.shape) != 2:
            raise ValueError("MoE input must be [tokens, hidden]")
        tokens, hidden = inputs.shape
        chunks = self.plan.chunks(tokens)
        output = self.allocate_output(inputs)
        _fp32(output, (tokens, hidden))
        if not chunks:
            return output
        pending = deque()
        slots = [
            self.allocate_slot(inputs, self.plan.policy.chunk_tokens)
            for _ in range(min(self.plan.policy.max_inflight, len(chunks)))
        ]
        for slot in slots:
            _fp32(slot, (self.plan.policy.chunk_tokens, hidden))
        uncertain_submission = False
        waiting_failed = False

        def finish(task):
            nonlocal waiting_failed
            try:
                self.wait(task["completion"])
            except BaseException:
                waiting_failed = True
                raise
            task["waited"] = True
            result = task["reduced"]
            if self.plan.policy.shared_policy == "replicated":
                shared = self.shared(inputs[task["start"] : task["stop"]])
                _fp32(shared, result.shape)
                result = self.combine_shared(result, shared)
            # This copy must precede reuse of the local slot. The adapter records
            # result allocator ownership on main before dropping its reference.
            self.store(output[task["start"] : task["stop"]], result)

        try:
            weights, ids = self.route(inputs)
            if weights.shape[0] != tokens or tuple(weights.shape) != tuple(ids.shape):
                raise ValueError("route result shape differs from complete batch")
            for index, (start, stop) in enumerate(chunks):
                if len(pending) == len(slots):
                    finish(pending[0])
                    pending.popleft()
                slot_number = index % len(slots)
                local = self.local(inputs[start:stop], weights[start:stop], ids[start:stop])
                _fp32(local, (stop - start, hidden))
                if self.plan.policy.shared_policy == "tp_sharded":
                    shared = self.shared(inputs[start:stop])
                    _fp32(shared, local.shape)
                    local = self.combine_shared(local, shared)
                _fp32(local, (stop - start, hidden))
                active_slot = slots[slot_number][: stop - start]
                self.store(active_slot, local)
                uncertain_submission = True
                reduced, completion = self.submit_reduce(active_slot, slot_number)
                uncertain_submission = False
                # Track a known completion before validating returned shape. A
                # bad result still may own an in-flight collective's input slot.
                pending.append(
                    {"start": start, "stop": stop, "reduced": reduced, "completion": completion, "slot": slot_number}
                )
                _fp32(reduced, (stop - start, hidden))
            while pending:
                finish(pending[0])
                pending.popleft()
        except BaseException as error:
            self.poisoned = True
            # Never add work after an uncertain submit or failed completion. If
            # failure preceded any unresolved submit, drain known handles once.
            if not uncertain_submission and not waiting_failed:
                while pending:
                    task = pending[0]
                    try:
                        if not task.get("waited", False):
                            self.wait(task["completion"])
                    except BaseException:
                        waiting_failed = True
                        break
                    pending.popleft()
            raise PoisonedSchedule(
                "MoE invocation failed; dispatch must remain held",
                unresolved_handles=len(pending),
                submission_uncertain=uncertain_submission,
            ) from error
        return output


def streaming_prefill(inputs, plan, rank_receipts, **callbacks):
    """Create an explicit admitted schedule and execute one eager invocation."""
    return ScheduledMoE(plan, rank_receipts, **callbacks).run(inputs)


@dataclass(frozen=True)
class DeviceScheduleContext:
    deferred_stream: object
    ledger: object = None


def npu_streaming_prefill(module, inputs, plan, rank_receipts, *, route, local, context):
    """Explicit device adapter, only invoked after T8 hardware/evidence admission.

    context.deferred_stream is an explicitly owned DeferredReduceStream, possibly
    shared under outer worker ownership. No per-layer tensor slots are retained.
    main.wait_event orders work; it is not a host completion/error confirmation.
    Collective failure detection/thermal holds remain outer runtime responsibilities.
    """
    rank_receipts = validate_rank_plan(plan, rank_receipts)
    import torch

    from vllm_ascend.utils import current_stream, npu_stream_switch

    if inputs.device.type != "npu":
        raise ValueError("explicit device schedule requires NPU inputs")
    if module._tp_reduce is None or not module.grouped_routing or module.expert_tp_size != plan.tp_size:
        raise ValueError("device schedule requires matching grouped TP reduction")
    expected = (
        "none" if not module.has_shared_expert else "replicated" if module.shared_expert_replicated else "tp_sharded"
    )
    if expected != plan.policy.shared_policy:
        raise ValueError("device module shared placement differs from admitted plan")
    main, comm = current_stream(), context.deferred_stream.get()

    def submit(local_tensor, slot):
        ready = main.record_event()
        local_tensor.record_stream(comm)
        with npu_stream_switch(comm):
            comm.wait_event(ready)
            reduced = module._tp_reduce(local_tensor)
            done = comm.record_event()
        reduced.record_stream(main)
        if context.ledger is not None:
            context.ledger.record(
                "collective", "streaming_chunk_reduce", nbytes=local_tensor.numel() * local_tensor.element_size()
            )
        return reduced, done

    def wait(done):
        main.wait_event(done)

    def store(destination, source):
        source.record_stream(main)
        destination.copy_(source)

    return streaming_prefill(
        inputs,
        plan,
        rank_receipts,
        route=route,
        local=local,
        shared=module._forward_shared if module.has_shared_expert else None,
        submit_reduce=submit,
        wait=wait,
        store=store,
        allocate_output=lambda tensor: torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device),
        allocate_slot=lambda tensor, tokens: torch.empty(
            (tokens, tensor.shape[1]), dtype=torch.float32, device=tensor.device
        ),
    )
