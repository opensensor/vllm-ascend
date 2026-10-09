# Packed compressor-state candidate

Historical offline-stage report. Hardware validation followed; see [hardware source record](HARDWARE_SOURCE.json).

US English summary. [中文说明](IMPLEMENTATION.zh-CN.md).

## Delivered behavior

The offline candidate implements eight four-token compressor pools per
32-token state page. It preserves FP32 state values, four-token compression,
the 640-token attention page and the existing 40 KiB padded small-page class.

Enable the candidate during model construction with the HF override:

```json
{"ascend_glm_kpool_state_block_size": 32}
```

The default remains four-token state pages. The explicit 32-token setting is
accepted only on 310P with four-token compression and 256 FP32 state elements.
Target and draft state layers both read this setting. Their MTP1 state window
becomes 33 tokens; no compression precision or speculative acceptance rule
changes. Unsupported values/geometry are rejected before cache allocation.

Both the ordinary writer and resident compact writer now divide slots by the
physical state block size, gather only the required four-row pool, and write
through the real padded page stride. They preserve gather-before-write order,
negative-slot masking, request isolation and speculative tail retention. The
legacy one-pool layout retains its original gather dispatch.

State metadata and cache backend already carry a per-group block size and
construct a padded view, so no model-runner or scheduler patch is introduced.
Metadata/cache geometry disagreement is rejected rather than silently using a
four-token slot table with a 32-token cache view.

## Native division bundle

The native integer kernel adds exact signed division by 32 and now exports
`glm_integer_divide_v2`. Its Python helper and build provenance identify that
entry. The admission gate rejects old entry metadata or missing divisor-32
coverage, and requires both integer widths, signed extremes, owned padding and
changed-input graph replay at counts **2, 8, 640, 1,280 and 2,560**.

The authoritative-writer rewrite recognizes the new reused state quotient while
retaining compatibility with the old writer expressions. New state addressing
uses the private native divide proxy rather than intentionally introducing a
generic host division fallback.

The native binary **must be rebuilt and hardware gated** before loading the new
resident helper. Existing v1 binaries are not qualified for this candidate.
The CANN kernel has not been compiled or run in this offline task.

## Memory result

For the archived TP4 + MTP1 geometry, at a 1,280-token scheduler budget and
131,072 context, state admission changes from 322 IDs to 42. Total required IDs
change from 536 to 256: **4.171142578125 → 1.9921875 GiB per rank**, a predicted
**2.178955 GiB** reduction in minimum cache admission.

This does not automatically shrink a fraction-based cache allocation. The
worker's actual allocation budget and graph/workspace reserve must be selected
from the completed per-rank memory ledger. The state geometry requires a cold
cache allocation and cannot be hot-swapped into an existing four-token pool.

The planner now accepts the implemented geometry explicitly:

```bash
python -m tools.glm_perf.prefill_memory_budget \
  --chunk-tokens 1280 --context-tokens 131072 \
  --state-block-tokens 32 --cache-gib 3.72
```

It still reports `offline_feasible: false` without qualified graph, transient,
external workspace, fragmentation and safety bounds. Envelopes must match the
state geometry as well as the candidate signature and configuration. Old
four-token envelopes cannot qualify the packed layout.

## Validation

- **302 targeted CPU tests pass in the delivery workspace.** They include the
  actual ordinary writer, compact writer, native-rewritten writer and bound
  wrapper against an independent token-history reference.
- Cases cover full 1,280-token chunks, large multi-request batches, empty
  requests, partial continuations, one-token decode, state page boundaries,
  aligned 640-token prefix reuse, request/page reuse, MTP1 rejection that
  overwrites an already-completed pool, FP16/BF16 key storage and masked slots.
- Poisoned 40 KiB pages, nonzero storage offsets and state/indexer views in one
  shared physical allocation are compared byte-for-byte, including padding.
- **1,764 CPU tests pass** in the isolated GLM worktree containing the earlier
  performance queue. Six warnings concern local Python/dependency deprecations
  and unavailable NVML. Ruff and Markdown checks pass for this change.
- The initial broad run exposed a missing test-only `Path` import; it was
  corrected before the passing run. No production failure is concealed.
- No NPU compilation, graph capture, real server requests, language-quality
  verdict or measured memory/speed gain is claimed. No server launch, restore
  or NPU probe was performed.

CPU tests validate writer-level state/compression behavior. Full scheduler
rollback, prefix-cache integration and target/draft graph replay remain hardware
qualification requirements. The complete offline memory envelope is still
unqualified, so this delivery does not authorize a guessed launch.

## Changed implementation

- `vllm_ascend/models/glm5next/kv_cache.py`: explicit state-page configuration and
  conservative window.
- `vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py`: ordinary addressing.
- `tools/glm_perf/resident_candidates/kpool_completed_prefill.py`: resident
  addressing and geometry checks.
- `tools/glm_perf/glm_integer_divide.cpp` and integer build/helper/probe/control:
  divisor 32, v2 binary contract and expanded admission gates.
- `tools/glm_perf/prefill_memory_budget.py`: packed geometry and matching
  envelope checks.
- `tests/ut/glm_perf/test_packed_kpool_state.py` and related parity/division/
  budget tests: reference history, shared backing and stale-bundle regressions.

Sources are present in the shared workspace, uncommitted. The unrelated Qwen
work was preserved. [Analysis](README.md), [historical inputs](INPUTS.json) and
[implementation validation](IMPLEMENTATION.json) distinguish predictions from
hardware measurements.
