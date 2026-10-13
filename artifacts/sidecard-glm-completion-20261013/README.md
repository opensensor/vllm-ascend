# GLM DMA completion integration: hardware measurements

The complete GLM service now uses coordinator-owned CUDA result buffers and
prequeued CUDA consumers released by Ascend DMA completion words. It passed
all reference checks and remains running. This integration establishes the
completion mechanism; it does not establish a throughput improvement.

The final input-record version measured **11.86 decode tokens/s** over three
matched runs, compared with **12.14 tokens/s** for the fresh qualified control.
That is a 2.3% lower median. Individual final runs span 11.79–12.29 tokens/s.
The [design PNG](design.png) shows the implemented architecture and its limits.

## Measurements

| Variant | Decode median, tokens/s | 2K time to first text, s | 8K time to first text, s |
| --- | --- | --- | --- |
| Fresh qualified control | 12.1389 | 22.9337 | 79.2827 |
| First packed-result contract | 11.7812 | 23.2627 | 81.8430 |
| Final GPU input-record contract | 11.8639 | 23.2798 | 79.4610 |

Decode uses three completed requests with 2,048 prompt tokens and 64 output
tokens. The 8K result is one completed request with 8,192 prompt tokens and
8 output tokens. All requests are uncached and force the output budget with
`ignore_eos`. The exact repeated prompt is:

```text
Inspect this module for profiler test 20261011 and explain the control flow.
```

The tokenizer seed includes a trailing space. Decode throughput excludes time
to first text: 63 subsequent output tokens divided by the remaining request
duration. The first control reuses the most recent pre-integration request; the other
two control requests were measured before the candidate launches. All baseline
and candidate generated continuations match exactly.
No transport probes ran concurrently with these performance measurements.
These small differences and overlapping run ranges warrant caution; neither
contract variant provides a demonstrated speedup over the fresh control.

For a public description, the current integrated result is approximately
**12 decode tokens/s**, with **23.28 s first-text latency at 2K** and
**79.46 s at 8K**, on one Ascend 310P Duo card and one RTX PRO 6000 Blackwell.
The older dual-Ascend result used a different output budget; it is historical
context rather than a matched comparison in this study.

## Implemented transport

CUDA runs attention, dense layers, and shared/hot experts. Both chips on the
assigned Ascend card compute cold routed experts. The separate Ascend card
remains reserved for the user.

The final variant generates the invocation record on CUDA and appends it to
the existing input packet. Ascend validates the record with its existing
routing metadata read. It transfers the native FP32 result directly into
coordinator-owned CUDA memory, then separately transfers the sequence from
the uploaded HBM record into the CUDA completion word. CUDA already has the
rank-ordered merge queued behind its completion wait.

This removes the first variant's HBM result packing and host-to-HBM header
upload. Explicit FP32 additions preserve the original rank 0, then rank 1
rounding order. Coordinator-owned controls qualify native owner-context RDMA
ordering; workers hold duplicate DMA-BUF fds rather than imported result views.
There are no host result flush calls on this path.

**Ascend initiates DMA in both directions. The CPU still submits blocking DMA
ioctls and schedules Ascend kernels.** Activation/result payloads bypass host
staging, while routing metadata and the input record are validated on the CPU.
This is not a fully autonomous device dispatcher. The current 310P stack
rejects stream value wait/write with status 207000, as documented in the
[earlier device-contract study](../sidecard-device-contract-20261012/README.md).

Each worker holds a main bridge fd for input/result mappings and a second fd
for the completion control. Four bridge references remain with the live
service. Shutdown closes foreign mappings before freeing the CUDA owner;
uncertain DMA or pending consumers retain their original owners.

The [runtime runbook](../../tools/sidecard/GLM_COMPLETION.md) describes the
configuration, compilation, qualification, and lifetime rules.

## Validation and provenance

- 842 sidecard unit tests pass in the research and clean fork worktrees.
- Both GLM variants pass the authenticated external prefill/decode reference:
  all 127 complete-vocabulary boundary hashes and teacher-forced tokens match.
- The final service passes transport, pipeline, decode-repeat, and DSA checks.
- Six API cases pass, followed by a successful arithmetic request after 8K.
- The final delivery probe passes 12 real-DMA cases on physical chips 2 and 3,
  including adversarial FP32 cancellation and prequeued consumers. Both
  producer processes close normally with intact guards.
- All final probe Python files match delivery. The live GLM snapshot differs
  only in unused `completion_probe.py`: delivery adds a docstring and corrects
  a report field. All executed GLM Python files and native merge source match.

`probe-v3-results.json` contains an obsolete constant sequence-offset report
field inherited from the packed-result probe. Its executed input-record path
uses `rows * 8256 + 16`; the final `probe-v4-results.json` records that formula
correctly. Both probes pass the same 12 cases. The first packed-result probe
failed with a missing import before any future CUDA wait; its separate
quiescence audit and failed result are retained rather than counted as passing.

The kernel bridge remains UAPI 5. Binary hashes, frozen source hashes,
per-request metrics, worker receipts, and gate results are retained alongside
this report. Private runtime environment contents are excluded.

The bounded worker profiles show median native computation of about 1.74–1.76 ms
per one-row cold invocation. Input DMA is about 0.10–0.12 ms; metadata validation
about 0.23 ms; result/sequence publication about 0.17 ms. These are worker
intervals, not additive full-token latency: both workers overlap and CUDA also
executes the model. See [worker profile summary](worker-profile-summary.json).
Legacy byte/mapping receipt counters cover payloads/main mappings; they exclude
the extra 64-byte input record and notification mapping.

Changed-file hooks and native CUDA formatting pass. Full `bash format.sh ci`
retains the baseline's seven failing hooks. The initial candidate exposed a
missing producer docstring; that was fixed before the final full-CI run.
See [full CI summary](full-ci-summary.json). Auto-edits from full CI were confined
to the clean integration worktree and discarded before delivery.

## Running service

- Endpoint: `http://192.168.53.187:8000/v1`
- Model name: `glm-sidecard`
- Configured admission limit: 65,536 tokens
- Benchmarked runtime: `/srv/ai/storage/artifacts/sidecard-glm-completion-20261013/candidate-v2`
- Current runtime: `/srv/ai/storage/artifacts/sidecard-glm-completion-20261013/kilo-v8`
- [Kilo configuration and API compatibility update](../sidecard-kilo-20261013/README.md)
- Parent PID at qualification: 1156406

Completed 8K measurements do not establish quality at the full admission limit.
NPU prefill graphs remain disabled; CUDA prefill convolution and 32-row chunks
retain their previously qualified settings.
