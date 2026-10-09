# SPDX-License-Identifier: Apache-2.0
"""Explicit per-candidate layer composition at existing Qwen runtime seams.

This host-only module loads no device code. The hooks call unchanged serving
methods with owned resources, keep prefix completion drains, and reuse baseline
PLE staging. Device state-selection and graph QSA selection-copy fusion are not
implemented. StepOwnership is a dependency model, not a hardware event substitute.
"""

from contextlib import contextmanager
from dataclasses import dataclass

GDN_TARGET = "vllm_ascend.models.qwen4_exp.model:_GDNAttention._native_delta_rule"
PREFIX_UPDATE_TARGET = "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._update_states"
PREFIX_REMAP_TARGET = "vllm_ascend._310p.model_runner_310p:NPUModelRunner310._remap_compact_mamba_block_tables"
PLE_TARGET = "vllm_ascend.models.qwen4_exp.model:_PLEInjection._forward_eager"
MAX_STAGING_TOKENS = 65536


@dataclass(frozen=True)
class LayerPolicy:
    native_state_io: bool = True
    native_wy: bool = True
    prefix_phase_batching: bool = True
    ple_staging_tokens: int = 2560

    def __post_init__(self):
        for name in ("native_state_io", "native_wy", "prefix_phase_batching"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("layer policy switches must be booleans")
        if type(self.ple_staging_tokens) is not int or not 0 <= self.ple_staging_tokens <= MAX_STAGING_TOKENS:
            raise ValueError("PLE staging token capacity must be in [0, 65536]")


@dataclass(frozen=True)
class LayerResources:
    state_io: object = None
    wy: object = None


@contextmanager
def _attributes(instance, assignments, *, reject_competing=False):
    """Bind all resources only after conflict validation, then restore all attrs.

    State written by the serving call is intentionally not rolled back: kernel
    completion/recovery belongs to the runner. Resource references and configuration
    attributes are restored even on cancellation/error. No device barriers added.
    """
    old = [(name, hasattr(instance, name), getattr(instance, name, None)) for name, _ in assignments]
    if reject_competing:
        for (name, _, previous), (_, resource) in zip(old, assignments):
            if previous is not None and previous is not resource:
                raise ValueError(f"competing layer resource: {name}")
    try:
        for name, value in assignments:
            setattr(instance, name, value)
        yield
    finally:
        for name, existed, previous in reversed(old):
            if existed:
                setattr(instance, name, previous)
            elif hasattr(instance, name):
                delattr(instance, name)


def _require_original(original):
    # Silent unwrapping would hide a resident candidate from its controller.
    if not callable(original) or any(
        hasattr(original, name)
        for name in ("_qwen_delta_rule_base", "_qwen_prefix_base", "_qwen_streaming_layer_owner")
    ):
        raise ValueError("layer composition requires original methods, not stacked candidate wrappers")


@dataclass(frozen=True)
class LayerComposition:
    policy: LayerPolicy
    resources: LayerResources

    def __post_init__(self):
        if self.policy.native_state_io and not (
            callable(getattr(self.resources.state_io, "gather", None))
            and callable(getattr(self.resources.state_io, "scatter", None))
        ):
            raise ValueError("native state IO requires gather/scatter resources")
        if self.policy.native_wy and not callable(self.resources.wy):
            raise ValueError("native WY requires a callable resource")

    def gdn_call(self, original, instance, *args, **kwargs):
        assignments = []
        if self.policy.native_state_io:
            assignments.append(("_gdn_state_io", self.resources.state_io))
        if self.policy.native_wy:
            assignments.append(("_gdn_wy_prepare", self.resources.wy))
        with _attributes(instance, assignments, reject_competing=True):
            return original(instance, *args, **kwargs)

    def prefix_call(self, original, instance, *args, **kwargs):
        with _attributes(instance, [("_prefix_phase_batching", self.policy.prefix_phase_batching)]):
            # Actual runner invokes apply_prefix_mamba_updates/remap helpers. Their
            # fresh per-phase worker drain remains; no old completion is reused.
            return original(instance, *args, **kwargs)

    def ple_call(self, original, instance, *args, **kwargs):
        tokens = self.policy.ple_staging_tokens
        if not tokens:
            return original(instance, *args, **kwargs)
        layer = instance.ple
        previous = layer.host_staging_tokens
        if previous not in (0, tokens):
            raise ValueError("existing PLE staging capacity differs from candidate")
        stage = getattr(layer, "_row_host_stage", None)
        if stage is not None:
            expected = (tokens * layer.num_ngram_heads, layer.per_head_dim)
            if tuple(stage.host.shape) != expected:
                raise ValueError("existing PLE staging allocation has a different shape")
        with _attributes(layer, [("host_staging_tokens", tokens)]):
            # Baseline gather_embeddings owns the pinned buffer/completion event.
            # Keep any newly allocated stage on the layer: freeing on error while
            # DMA may still consume it is unsafe. Recovery must drain first.
            return original(instance, *args, **kwargs)

    def replacements(self, *, gdn_original, prefix_update_original, prefix_remap_original, ple_original):
        """One composition map for T8; no imports, global patches, or installation.

        Candidate admission supplies real originals after provenance verification.
        Policy stays immutable; resources are referenced, never cloned/reloaded.
        QSA and accepted-token runner methods remain unchanged deliberately.
        """
        methods = (
            (GDN_TARGET, gdn_original, self.gdn_call),
            (PREFIX_UPDATE_TARGET, prefix_update_original, self.prefix_call),
            (PREFIX_REMAP_TARGET, prefix_remap_original, self.prefix_call),
            (PLE_TARGET, ple_original, self.ple_call),
        )
        for _, original, _ in methods:
            _require_original(original)
        result = {}
        for target, original, invoke in methods:

            def call(instance, *args, _original=original, _invoke=invoke, **kwargs):
                return _invoke(_original, instance, *args, **kwargs)

            call._qwen_streaming_layer_owner = self
            result[target] = call
        return result


@dataclass(frozen=True)
class CacheOwner:
    layer_prefix: str
    key_cache: object
    value_cache: object
    index_cache: object


def validate_qsa_cache_owners(owners):
    """Layer caches cannot alias; shared step metadata does not change ownership."""
    owners = tuple(owners)
    prefixes = set()
    storage = []
    for owner in owners:
        if not owner.layer_prefix or owner.layer_prefix in prefixes:
            raise ValueError("QSA cache owners need unique layer identities")
        prefixes.add(owner.layer_prefix)
        caches = (owner.key_cache, owner.value_cache, owner.index_cache)
        keys = [
            (str(cache.device), cache.untyped_storage().data_ptr())
            if hasattr(cache, "untyped_storage")
            else ("object", id(cache))
            for cache in caches
        ]
        if any(cache is None for cache in caches) or any(key in storage for key in keys):
            raise ValueError("QSA caches cannot be missing or shared across layers")
        storage.extend(keys)
    return tuple(owners)


def consume_qsa_selection(selection, consumer, *args, **kwargs):
    """Pass selection directly to an eager consumer; baseline already does this.

    Graph selection uses persistent buffers with a required callback copy. This
    function does not replace that copy, share caches, or alter selection ties.
    """
    return consumer(*args, selection=selection, **kwargs)


def accepted_host_counts(snapshot, sampled, request_ids):
    """Reuse the existing sampled-token CPU snapshot only for the exact batch.

    No tensor value inspection, CPU copy, or count recomputation occurs. The
    baseline runner publishes this snapshot before filtering and clears it in
    finally; reordering/replacing sampled storage invalidates it.
    """
    if snapshot is None:
        return None
    sampled_owner, frozen_ids, counts = snapshot
    return counts if sampled_owner is sampled and frozen_ids == tuple(request_ids) else None


@dataclass(frozen=True)
class Lease:
    step: int
    name: str
    layer_prefix: str
    consumers: tuple[str, ...]


class StepOwnership:
    """Bounded host dependency checks for integration/shadow tests.

    Completion booleans must represent external established completion, not a
    kernel submission return. No polling or new barriers are introduced here.
    Baseline checkpoint drains and PLE events remain the actual runtime owners.
    """

    def __init__(self, step, *, max_leases=32):
        if type(step) is not int or step < 0 or type(max_leases) is not int or max_leases <= 0:
            raise ValueError("step and lease capacity must be valid integers")
        self.step = step
        self.max_leases = max_leases
        self._pending = {}
        self._leases = {}
        self._closed = False
        self._committed = False

    def acquire(self, name, layer_prefix, consumers):
        consumers = tuple(consumers)
        if self._closed or not name or not layer_prefix or not consumers or len(set(consumers)) != len(consumers):
            raise ValueError("invalid or closed ownership lease")
        if name in self._leases or len(self._leases) >= self.max_leases:
            raise ValueError("ownership lease reused or capacity exceeded")
        lease = Lease(self.step, name, layer_prefix, consumers)
        self._leases[name] = lease
        self._pending[name] = set(consumers)
        return lease

    def complete(self, lease, consumer, *, established_completion):
        if self._closed or self._leases.get(lease.name) is not lease or lease.step != self.step:
            raise ValueError("stale or foreign ownership lease")
        if established_completion is not True:
            raise ValueError("submission is not consumer completion")
        if consumer not in self._pending[lease.name]:
            raise ValueError("consumer absent or already completed")
        self._pending[lease.name].remove(consumer)

    def commit_accepted_state(self, *, snapshot, sampled, request_ids):
        if self._closed or self._committed or any(self._pending.values()):
            raise ValueError("accepted state commit requires all last consumers complete")
        counts = accepted_host_counts(snapshot, sampled, request_ids)
        if counts is None:
            raise ValueError("accepted snapshot identity missing or changed")
        self._committed = True
        self._closed = True
        return counts

    def cancel(self, *, established_completion):
        if self._closed or established_completion is not True:
            raise ValueError("cancellation requires existing completion before resource retirement")
        self._closed = True
        self._pending.clear()
