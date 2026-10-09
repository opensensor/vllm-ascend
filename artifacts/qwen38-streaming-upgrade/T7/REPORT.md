# T7: explicit layer composition and ownership

Implemented an immutable per-candidate `LayerPolicy`, `LayerResources`, and
`LayerComposition` in `tools/qwen4exp/streaming_layer.py`. One composition map binds
native state IO and WY together on the actual custom `_GDNAttention._native_delta_rule`
method. Conflicting resource owners and stacked candidate wrappers are rejected.
Configuration/resource attributes restore on success, error or cancellation; state
writes themselves are not claimed to be an atomic rollback transaction.

The composition also enables the existing prefix phase batching flag on runner
updates and compact-table admission. Existing helper functions still establish a
fresh worker completion drain for each required phase, then perform serialized
invalidation/CoW or admission. No completed phase is reused as proof for a later
phase, and no required completion barrier was deleted.

PLE composes at `_PLEInjection._forward_eager`, which is invoked by both eager
execution and the existing graph replay callback. This applies the same bounded
staging policy on every replay. It uses the baseline layer-owned `PinnedHostStaging`
buffer and its DMA completion guard, rejecting incompatible existing capacity or
shape. Newly allocated staging storage remains owned by the layer even after an
error, because DMA may still consume it. The controller must establish completion
before retirement. The host completion event protects host buffer rewriting; it
does not independently prove safety of reusing a device output.

QSA eager selection already goes directly to its consumer. The helper preserves
selection object identity and verifies distinct cache storage across layers. It
does not install another QSA forward path. The graph callback's fixed-buffer copy,
stable selection ties, causal masking and cache identity remain necessary baseline
boundaries. Logical-position sharing, indexer fusion and elimination of graph
selection copies are unsupported by this package.

The baseline runner already performs one sampled-token CPU readback, computes raw
acceptance counts before discard/vocabulary filtering, and exposes the same host
snapshot for Mamba alignment only with matching tensor and request identities.
This package retains that actual path and adds a data-only identity reuse helper.
Native accepted-state selection and removing host dispatch for W8A16 MTP remain
unsupported; no new readback or hidden W8A8 arithmetic substitution was added.

`StepOwnership` models bounded per-step leases, named last consumers, established
completion, cancellation and accepted-state commit. It rejects submission-as-
completion, foreign/stale leases, premature commit and unsafe retirement. This is
an offline integration/shadow dependency model; its boolean inputs do not prove
hardware completion and do not replace NPU events. T8 must preserve baseline
runtime owners and their real completion mechanisms.

Validation: 24 new CPU tests pass, including execution of the actual extracted
GDN method with cold/warm states and combined resources, decode/speculative bypass,
mixed attention views, actual runner snapshot publication/finally cleanup, CoW
phase drain, PLE DMA reuse and error restoration, QSA storage alias rejection,
MTP ownership and cancellation. Scoped Ruff passed. The combined prefix, GDN and
prior transfer suite passed 150 tests with 20 existing environment warnings in
11.42 seconds; results are recorded in `integration-tests.log`. That suite also
exercises previously implemented CPU-compiled kernel stubs. No NPU library was
loaded, no NPU runtime was queried, and no server was changed or started. CPU stubs do not establish
hardware accuracy, ordering, image quality, throughput or thermal improvement.

## T8 integration API

Construct `LayerComposition(LayerPolicy(...), LayerResources(state_io, wy))` only
after native source/binary/evidence admission. Call `replacements` with original
GDN, runner update/remap, and PLE eager methods. The returned four-target map is
one coherent owner; it performs no installation or global patch itself. Avoid
combining it with the older state IO/WY/prefix replacement factories.

Image handling and the existing multimodal pipeline remain unchanged. The policy
applies at text-layer/PLE seams, including image-plus-text requests; no image
processing success is inferred from these offline tests. Device validation of the
composed state/WY resources, graph replay staging, prefix/CoW and image/text
interleaving remains mandatory before live use.
