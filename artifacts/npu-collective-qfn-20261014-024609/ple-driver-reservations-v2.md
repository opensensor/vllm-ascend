# Dynamic DMA reservations: v2 qualification

Status: **installed, enabled on all four NPUs, and qualified in live QFN serving**.

The v1 driver rejected fixed-instance host DMA targeting a leased queue. This
caused ACL initialization to fail with 507033 while leases were active. V2
requires an idle candidate under the existing CQ/SQ locks, publishes a stable
host successor, and then marks the queue reserved. Logical instance modulo
mapping remains immutable. Release does not reverse the redirect: the
successor can still contain outstanding work. Channel destruction/reset is
the reset boundary for this routing change.

## Validation

- 9 driver CPU tests, including 87,934 successful dispatches and the
  acquire-between-selection-and-submission race.
- 62 focused client/server tests; Ruff checks pass. These tests run with
  `--noconftest` because the unrelated repository-wide conftest requires an
  upstream `vllm.third_party.flash_linear_attention` module absent locally.
- 31 rebuilt modules passed kernel, symbol CRC, namespace, and isolated
  dependency validation. All 31 loaded versions match after the reboot.
- Boot ID: `9878eb44-bf13-4931-a2ba-de779ba5eb74`.
- 144 exact page-crossing transfers across four NPUs: 3 rounds each before,
  during, and after leases; sources include shared RW/RO, sealed memfd, and
  ACL host allocations. No ACL startup failure recurred.
- All four devices passed acquire/query, duplicate rejection, stale-release
  rejection, explicit release/reacquire, and last-close cleanup.
- Unprivileged API startup acquired all four leases (generations 17–20,
  channels 20/25 on each Duo); two host traffic channels remain per NPU.

- Six model suites passed 10 cases each, including 8K prompts and four
  concurrent requests; the final serving gate passed 12 cases covering token
  budgets, penalties, turnover, structured output, abort, and recovery.
- All retained controls were restored and inspected, including engine queue
  depth 2 and graph coverage `[3, 6, 9, 12]` on every rank. Existing PLE and
  general-selection eager segments remain; this is not full CPU removal.
- Slowest model-load rank: 116.85 seconds. No server error, traceback, or
  eager-fallback warning was logged during this qualification.
- The final API process (PID 23734) owns four live lease FDs, confirmed against
  four reserved sysfs records after both benchmarks. Hardware stayed healthy;
  highest sampled temperature was 84°C. Validation batches were spaced with
  cooling intervals; benchmark requests were not paused.

## Measured performance

| Fixed output workload | Prior v8, no reservations | v9, four reservations |
| --- | ---: | ---: |
| C1 median HTTP output, 3 × 256 tokens | 32.385 tok/s | 31.929 tok/s |
| C4 aggregate HTTP output, 4 × 256 tokens | 62.382 tok/s | 63.432 tok/s |
| C4 median per-request streaming decode estimate | 16.969 tok/s | 17.131 tok/s |

HTTP throughput includes prefill/API time; the streaming estimate uses the
interval between first and last output. These are short sequential checks,
not a controlled A/B or evidence of a speedup. C1 is 1.4% lower and C4 aggregate
1.7% higher than the prior run. Reservations coexist with retained performance.

Evidence: [DMA coexistence](ple-driver-dma-coexist-v2.json),
[lease lifecycle](ple-driver-lease-hardware-v2.json),
[live ownership](ple-driver-serving-reservations-v2.json),
[retained configuration](resident-live-v9/final-driver-restoration.json),
[final serving gate](resident-live-v9/requalification/scheduling-serving-gate-driver-restored-v1.json),
[C1](qsa-resident-v1/decode-benchmark-driver-reservations-v9-v1.json), and
[C4](qsa-resident-v1/concurrency-driver-reservations-v9-v1.json).

## Serving configuration

`qwen38-flash-next`, port 8001, context 262144. Run directory
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v9`.
The stable log remains `server-current.log` in the experiment root.

The new `--dma-reserve-device` CLI option can be repeated for arbitrary driver
IDs. It defaults to no reservations and requires capability bit 0 (isolated
host routing). No new per-token Python callback or lease registry lock was
added. The node is delegated to the existing `HwHiAiUser` group using udev,
mode 0660. The API owns leases before ACL initializes and through serving.

## Scope

`device_submit=0`: no device producer, queue mapping, or autonomous PLE DMA is
claimed. This resolves the reservation/ordinary-runtime conflict; it does not
by itself remove CPU lookup/submission or establish a performance gain.

## Rollback

The original installed modules and v1 source snapshot are saved under
`ple-driver-deployment-v2`. `ple-driver-deploy-v2.py rollback` restores those
files, rebuilds dependencies and initramfs; reboot afterward. V1 serving must
omit the new reservation options. Keep the existing model name and port.
