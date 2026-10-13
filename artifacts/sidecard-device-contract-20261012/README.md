# DMA handoff contract: CUDA consumer qualification

**The CUDA consumer is released by direct DMA and validates the transferred
invocation record on device.** Both assigned Ascend chips passed 72 cases:
48 accepted records and 24 deliberately invalid records. Two independent
slots finish out of submission order. No socket command, host flush call,
or CUDA launch after the final flag releases the queued consumer.

This is a bounded transport prototype. DMA submission remains a blocking
host ioctl, and the Ascend producer is a host-initialized fixture. The live
GLM service retains its qualified transport; it does not use this contract
yet. No GLM throughput improvement is claimed from these probes.

The [design PNG](design.png) shows the qualified mechanism and remaining
integration boundary. The [runbook](../../tools/sidecard/pcie/DEVICE_CONTRACT.md)
describes the codec, ordering, ownership, and reproduction commands.

## Hardware results

The final v7 probe uses physical Ascend chips 2 and 3 on one 310P Duo card
and the RTX PRO 6000 Blackwell. The other Ascend card is reserved for the
user. Power, clocks, ECC, device reset, and driver unload were untouched.

| Check | Result |
| --- | --- |
| RTX 64-bit stream memory operations | Supported |
| RTX RDMA writes ordering | 100: native owner-context ordering |
| 310P `aclrtValueWait` / `aclrtValueWrite` | Unsupported, status 207000 on both chips |
| Accepted CUDA records | 48/48; exact transformed payloads |
| Rejected CUDA records | 24/24; fixed rejection marker |
| Largest tested payload | 524,288 bytes |
| Independent completion | Slot 1 finishes while slot 0 still waits, every wave |
| Lifetime guards | Intact; both probe processes closed normally |
| Bridge references afterward | 2, belonging to the live GLM workers |

Each slot has an owned CUDA payload and completion allocation, preallocated
stream/event, and its own bridge fd. The final consumer is preloaded before
any future wait. Payload DMA completes first; a second DMA writes the 64-byte
record beside an unready completion word; a third DMA writes the final 64-bit
sequence. CUDA validates epoch, sequence, expected layer/slot/rank, rows,
derived byte geometry, and reserved fields before consuming payload bytes.

Six rejection fixtures cover stale epoch, stale sequence, foreign rank,
foreign layer, changed geometry, and nonzero reserved data. Each rejected
record produces `0xEE` without loading the payload. The consumer XORs accepted
payload bytes with `0xA7`, establishing that queued device computation consumed
the new transfer, rather than merely reading back the producer allocation.

Native owner ordering qualifies this single-context probe. The live GLM
coordinator consumes worker-owned buffers through CUDA IPC; its existing
flush calls remain. Imported aliases are rejected by `CudaCompletionWait`.
The proof does not establish coherence of arbitrary running polling kernels.

## Failed first probe and recovery

The first probe stalled during lazy CUDA kernel loading behind an unsatisfied
future-value wait. It had submitted zero original DMA jobs. Recovery duplicated
only the authenticated completion DMA-BUF fd, transferred its expected 8-byte
value from the assigned Ascend chip, and established completion of the queued
CUDA consumer. Its intentional early-completion assertion then quarantined
the process. Only after both CUDA and recovery DMA quiescence were established
was ordinary process cleanup performed. The original GLM service remained
running throughout. No blind forced termination or reset was used.

[recovery.json](recovery.json) and
[quiescence-cleanup.json](quiescence-cleanup.json) retain the audit. v2 passed
with preload and host flush calls; v3–v5 passed independent completion words
without flush calls. v6 failed compilation before hardware execution; v7
adds the successfully compiled device descriptor validator and negative cases.
Earlier successful probes are retained separately from final v7 results.

## Prefill candidate rejected; service restored

The optional 32-row Ascend graph candidate passed 90 synthetic-input real-bank
expert comparisons, including local, mixed-peer, and zero-weight routes.
It failed the stronger complete-model prefill reference at the first boundary.
All 64 teacher-forced token IDs matched, but the full-vocabulary hashes did not.
It was rejected before API readiness and both workers closed with intact
guards and zero bridge references. The numerical cause remains unresolved.

The gate's reported zero `max_abs_logit_error` compares two runs of the same
graph-enabled candidate with different CUDA convolution choices. It does
not measure error against the prior qualified service. The authenticated
external reference caught the shared worker arithmetic change that this
within-candidate comparison could not detect.

The restored service disables NPU prefill graphs. Its 2K prefill plus 64 decode
steps match all 127 prior full-vocabulary boundary hashes and teacher tokens.
Six API cases passed, followed by a successful arithmetic request after 8K
prefill. The optional graph flag remains an experimental startup-gated
candidate, defaults off, and is not recommended for serving on this stack.

| Restored service measurement | Result |
| --- | --- |
| 2K time to first text, three runs | Median 22.9074 s |
| 8K time to first text, one run | 80.0047 s |
| 2K prompt / 64 output decode, three runs | Median 12.3189 tokens/s |
| Fresh pre-experiment control, two 2K/64 runs | TTFT 22.6469 s; decode 12.6419 tokens/s |

Serving math and architecture are unchanged from the prior qualified profile;
these small timing differences do not establish an optimization. All matched
fixtures repeat the exact profiler-test prompt from the preceding study.
No DMA probes ran concurrently with the restored measurements.

The live endpoint is `http://192.168.53.187:8000/v1`, model `glm-sidecard`,
with a 65,536-token admission limit. Completed 8K prompts do not establish
quality at the full limit. Frozen runtime output is
`/srv/ai/storage/artifacts/sidecard-prefill-graphs-20261012/restored-v1`.

## Delivery checks and next boundary

The sidecard suite passes 820 tests in both the research and fork worktrees.
Changed-file hooks and the CUDA format check pass. Full `bash format.sh ci`
on baseline and candidate reports the same seven existing failing hooks:
Ruff check/format, codespell, typos, clang-format, Markdown, and forbidden
imports. The isolated full-CI auto-edits were not copied into delivery.

All GLM execution files match the delivery base and scoped changes. The
restored snapshot contains an earlier, unused `mailbox_probe.py`; final
hardware probes use their own authenticated v7 source snapshot. All three
executed mailbox source files match delivery bytes. See
[source-parity.json](source-parity.json) and
[binary-provenance.json](binary-provenance.json).

Next, register coordinator-owned result/completion slots with the workers,
then queue GLM's ordered cold-result merge behind CUDA completion waits.
Ascend still needs a supported device dispatcher and asynchronous DMA queue.
The 310P stream wait/write APIs cannot supply that dispatcher on this hardware.
