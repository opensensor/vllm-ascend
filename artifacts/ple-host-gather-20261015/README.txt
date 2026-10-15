Full shared-RAM PLE transport qualification — 2026-10-15

The private experiment passed on all four Ascend 310P NPUs, including four
simultaneous readers of the same sealed host table. The model table contains
102,400,491,520 bytes (95.3679 GiB). It was loaded once per run, and all workers
attached to the same memfd inode. The full-v5 table was retained for the
concurrent-v6 gate without reloading or making four physical copies.

The NPU computes sixteen row IDs per token using actual checkpoint n-gram
parameters. Native host code reads one request packet, validates all receipts
and indices, gathers rows from shared RAM, and returns one packed response.
CPU involvement remains on each lookup. This is an experimental transport;
the live vLLM lookup path has not been replaced.

Observed round-trip medians with 5-us driver DMA polling and 40-us native
producer-stream polling (microseconds, median of four rank medians):

  tokens per rank      sequential ranks      four ranks active
       1                    45.7                   53.9
       3                    55.2                   64.8
      12                    89.4                  127.7

These intervals include selector launch, producer completion, request readback,
host gather, response DMA, and Python call overhead. Input updates and exact
validation readbacks are outside the interval. They do not include downstream
PLE projection/convolution or model generation. The twelve-token sequential
per-rank medians varied substantially; consult the JSON ranges and p95 values.

The initial full-table Torch-graph comparison measured packed native transport
alone at 24.5 -> 14.0 us for one token when changing driver polling from 0 to
5 us. The full call still cost about 158 us because the graph/fence dominated.
For three/twelve tokens, responses exceed the driver's 8192-byte polling limit;
row gathering also becomes a larger fraction of time.

Moving the producer launch into native code and querying its stream addresses
another completion-wait cost. With the full table, the producer/fence phase
fell from approximately 86–97 us with blocking synchronization to 32–37 us
with a 40-us query budget. This uses CANN aclrtStreamQuery, not another driver
or firmware change. Poll expiry uses ordinary aclrtSynchronizeStream. In both
the sequential and concurrent gates, all 128 one-token and 128 three-token
measured producers completed inside the 40-us query window; 121/128 twelve-
token producers did so. The remaining cases exercised the normal fallback.

Cache qualification matters: each mode reuses the same changing request
sequence in fixed order. Later modes benefit from cache warming. For example,
the sequential one-token gather phase was about 4 us in the first native mode
and 0.7 us in the later polling mode. Therefore whole-call ratios are not
controlled cold-cache speedups. Producer timing is reported separately, and
the observed absolute round trips do not establish model throughput or
worst-case latency. Concurrent runs used caller CPUs 8/24/40/56; sequential
runs used CPU 8. Runtime helpers retained broad affinity.

Validation:
  - 75 standalone CPU tests passed; Python lint/format checks passed.
  - Native libraries compiled with -O2 -Wall -Wextra -Werror.
  - 120 full-table cases, 3840 measured requests, 4320 payload-checked requests
    including warmup, and 368,640 exact returned rows.
  - Device-selected hashes matched independent CPU math. Shared-table samples
    also matched admitted checkpoint file reads.
  - 48 invalid-device-receipt checks and 24 invalid-row checks preserved prior
    output and recovered on the next valid request.
  - All four simultaneous readers passed on shared inode 4199.
  - All four NPUs remained healthy; all 31 loaded module versions matched the
    installed driver bundle; device firmware hash was unchanged.
  - The existing boot-time missing event-config warning remains; no new Ascend
    kernel error was recorded during this qualification.

The ordinary repository pytest entry could not initialize its framework
conftest because vllm.third_party.flash_linear_attention is missing locally.
The standalone tests use --confcutdir=tests/ut/qwen38_1m; the native NPU paths
were separately exercised on hardware as recorded above.

State after tests:
  - Driver dma_query_spin_us=5 is still active; boot default remains zero.
  - Driver profiling is disabled.
  - Native stream polling exists only in this private experimental library;
    it is not a global runtime setting or a serving modification.
  - The temporary 95-GiB shared table was released.
  - rtsp-scene-model.service and rtsp-scene-log.service remain disabled/inactive.

Implementation:
  tools/qwen4exp/ple_host_gather.h
  tools/qwen4exp/native_ple_host_gather.cpp
  tools/qwen4exp/benchmark_ple_host_gather_310.py
  tools/qwen4exp/PLE_HOST_GATHER.txt
  tests/ut/qwen38_1m/test_ple_host_gather.py
  tests/ut/qwen38_1m/cpp/ple_producer_wait.cpp

Receipts:
  live-validation.json: aggregate correctness and per-stage metrics.
  health-validation.json: post-test driver/firmware/service checks.
  full-v2/: full-table Torch graph, packed/per-row transport, driver 0/5 matrix.
  full-v5/: full-table native producer with blocking/20-us/40-us waits.
  concurrent-v6/: four simultaneous native readers of the same full-v5 table.
  native-v5/ and native-v6/: matching staged sources and library provenance.
  summarize.py: verifies complete receipt counts and regenerates the summary.

Synthetic-v1 timings are discarded: whole-process CPU affinity crowded CANN
helpers onto one CPU. Later versions corrected affinity and qualified the
native interfaces before the full-table runs. Full-v1 failed before admission
due to a missing register() keyword argument; full-v2 corrected it.

The remaining integration work is a stream-ordered native owner in the actual
serving path, preserving these buffer lifetimes, invalid-request behavior,
shared-table ownership and completion ordering. It needs model-output and
throughput validation before replacing the existing lookup implementation.
