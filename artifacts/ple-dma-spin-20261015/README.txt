Ascend 310P host DMA polling qualification — 2026-10-15

The bounded polling driver is installed and loaded on 192.168.53.187.
All 31 loaded module source versions match installation.json. All four NPUs
report healthy. The device firmware archive hash is unchanged.

Current runtime setting: dma_query_spin_us=5. The boot default remains 0;
no persistent modprobe opt-in was written. Profiling is disabled after capture.
The two rtsp-scene services remain disabled and inactive.

Eligible synchronous query-mode physical DMA requests up to 8192 bytes check
for completion for up to the configured polling budget before entering the
existing query/sleep loop. Interrupts and preemption remain enabled. A scheduler
reschedule request exits polling. This does not change IRQ-mode transfers or
provide device-initiated access to host RAM. The budget is not a hard deadline:
preemption and completion-queue processing can extend elapsed time.

Untraced latency measurements (median of 16 case medians, microseconds):
                         first run       repeat
  existing query/sleep       8.496         8.581
  2-us polling              10.320        10.235
  5-us polling               4.373         4.414

Each phase covers four ranks, 320- and 5120-byte contiguous copies, aligned
and page-crossing host addresses, and 1000 measured submissions per case.
Descriptors are prepared once and reused. Sequential trials use CPU 8.
The 5-us setting reduced the aggregate median by approximately 49%.
Median caller CPU time was about 5.5–5.6 us at baseline and 4.9–5.0 us at 5 us;
this accounting excludes work charged to other threads or CPUs.

After source-reservation qualification, concurrent trials on CPUs 8/24/40/56
measured 8.807 us at baseline and 4.318 us at 5 us. Maximum per-case p99 was
19.336 us and 13.716 us respectively. Other concurrent trials also showed
occasional tail spikes; these short trials do not establish worst-case latency.

The trace contains 20,680 events, with no loss, malformed samples, or DMA
errors. Of 16,528 test-copy events at 5 us, 16,508 completed inside initial
polling; none slept. At 2 us, 3312 of 4132 events slept. Tracing perturbs timing,
so the performance comparisons above use separate untraced runs.

Validation includes 17 CPU tests, a clean 31-module build and module checks,
164 benchmark cases with exact final payloads (164,000 measured submissions),
12 successful source registrations including cleanup/error checks, and four
simultaneous source leases coexisting with 16 exact host DMA copies.
Readback is checked after each 1000-call case, not after each individual copy.
The existing boot-time missing event-config warning remains in dmesg; no new
Ascend kernel error was recorded during qualification.

These are host-submission microbenchmarks. They do not measure sixteen random
PLE row gathers, the full device-request/host-response round trip, or model
throughput. Full PLE workload qualification remains the next integration step.

Runtime controls on the host:
  echo 0 | sudo tee /sys/module/drv_pcie_host/parameters/dma_query_spin_us
  echo 5 | sudo tee /sys/module/drv_pcie_host/parameters/dma_query_spin_us

Driver source: /home/matteius/ascend-310p-host-driver (development machine).
Installed DKMS source: /srv/ai/src/ascend-310p-host-driver-spin-20261015.
Host artifacts: /srv/ai/storage/artifacts/ple-dma-spin-20261015.
The host artifact directory retains the source snapshot, module bundle and
rollback backup. Its rollback.py restores the prior profiling driver files,
DKMS source/cache and initramfs; reboot separately to activate restored files.
Setting the runtime control to 0 needs no reboot and restores query/sleep.

Machine-readable evidence: live-validation.json, installation.json,
parameter-validation.json, phase/rank JSONL files and live-trace/.
Implementation and profiling usage: dma-query-poll.txt and profile-dma.py.
