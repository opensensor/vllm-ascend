# Initial joined GLM service evidence (retired)

This records the retired 8,192-token deployment, not current process IDs or
performance. The shared endpoint is `http://192.168.53.187:8001/v1`, model
`glm-sidecard`. The [runbook](../../../tools/sidecard/SERVICE.md) describes
launching, validation and subsequent context/performance profiles.

The API parent is PID 864265, CUDA resident worker PID 839221, and the four
NPU workers are 865298, 865708, 866093 and 866427. The original Ascend-only
server, PID 276072, was stopped with SIGTERM after CUDA preparation passed.
Both physical Ascend cards participate. No source checkpoint, installed
package, driver or model-runner file was changed.

## Admission and source identity

[cuda-ready.json](cuda-ready.json) records all 3,024 assigned experts across
42 main routed layers. CUDA owns global IDs 0–71 in each layer, with the other
native EP partitions on Ascend. Actual CUDA allocations total
76,259,197,952 bytes, or 71.0219 GiB. Preparation took 592.45 seconds with
eight CPU threads. Each selected native shard was authenticated and a single
packed CPU layer was staged at a time. The 82 GiB budget and 4 GiB device
reserve passed before weight allocation.

[service-source-receipt.json](service-source-receipt.json) pins 27 sidecard
Python/CUDA files, the generated worker-extension shim, CUDA library,
checkpoint manifest, placement and configuration. All 27 corresponding local
source files matched. The frozen native checkpoint/kernel provenance is
retained. The CUDA implementation remains the scalar correctness kernel used
by the earlier selected-expert gate; it is not an optimized packed MoE kernel.

The original native rank-zero expert banks remain on NPU for shadow validation.
Attention/state/shared work still runs on Ascend. This profile differs from
the target CUDA-attention/four-54-expert-partition design. Eager execution is
required; MTP and graphs are disabled.

## Operational results

[service-validation.json](service-validation.json) records seven completed
cases. All returned generated text; streaming ended with `[DONE]`. The API
before/after snapshots showed at least 437 additional CUDA routes in every
routed layer. The chat arithmetic smoke completed with answer `4`.

| Case | Prompt tokens | Generated tokens | End-to-end elapsed |
| --- | --- | --- | --- |
| Completion | 5 | 12 | 7.58 s |
| Chat | 23 | 24 | 18.10 s |
| Repeated completion | 5 | 12 | 7.94 s |
| Chunked prefill | 243 | 4 | 51.19 s |
| Concurrent request 1 | 5 | 6 | 6.18 s |
| Concurrent request 2 | 4 | 6 | 6.18 s |
| Stream | 7 | 4 | Events and completion marker checked |

These are prototype API timings, including tokenizer, host tuple/packet
handling, synchronous copies and the eager Ascend backbone. A representative
single-request decode logged approximately 0.58 seconds per token, about
1.7 tokens/s. There is no matched speedup claim. Long prefill remains slow.

The independent workstation-to-host request in
[lan-chat-low.json](lan-chat-low.json) used `reasoning_effort: low` and
completed with answer `4`. [lan-chat.json](lan-chat.json) records the earlier
32-token default-effort request, which returned reasoning text and reached its
token budget. LAN access and `/v1/models` were also verified.

## Numerical limits

[model-boundary-metrics.json](model-boundary-metrics.json) records 126
real-input shadow comparisons, three at each routed layer. Maximum absolute
expert contribution error was 1.1920929e-7; maximum NRMSE was 6.8953902e-8.
Routing, shared addition, scaling and HCCL composition use the existing GLM
seams. These expert checks do not qualify the entire model.

**Full validation did not pass.** Greedy repeated completions varied, and the
initial completion differed from the earlier Ascend-only graph-enabled
baseline. The cause has not been isolated. The report deliberately retains
`passed: false` alongside `operational_passed: true`.
[repeat-investigation.json](repeat-investigation.json) records three further
repetitions and their top log probabilities. The original failed strict run is
preserved in [service-validation-failed-repeat.json](service-validation-failed-repeat.json).
Whole-model fidelity and repeat stability remain unresolved.

## Process and verification records

[service-process.json](service-process.json) preserves the exact launch command.
[hardware-inventory-after-service.json](hardware-inventory-after-service.json)
records healthy NPU chips and the resident CUDA process.
[service.log](service.log) contains startup and actual request timings.
[cuda-worker.log](cuda-worker.log) contains admission and complete preparation.

The first server launch failed because the worker extension attempted to
override `shutdown`, which vLLM rejects. The override was removed and a
regression check added; [service-attempt1.log](service-attempt1.log) preserves
that failure. The first, single-threaded CUDA preparation was stopped with
SIGINT and clean allocation unwinding before the eight-thread retry;
[cuda-worker-serial.log](cuda-worker-serial.log) preserves its interruption.
No device reset or SIGKILL was used by the sidecard tooling.

The scoped CPU suite passed 190 tests. Ruff, format, codespell, typos and
Markdown checks passed. The complete repository test suite was not run.
The server and its CUDA worker remain running; there is no automatic restart
policy or production quality qualification.
