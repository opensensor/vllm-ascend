# Dynamic DMA driver deployment — 2026-10-14

The dynamic reservation driver is installed on the four-NPU host. It reserves
no channel at module load. The model loader/client can acquire a lease through
`/dev/ascend_device_dma`; serving does not acquire one automatically.

## Installed state

- Kernel: `7.0.0-38-generic`; 31 candidate modules installed and verified against
  the loaded modules' source versions.
- Candidate contains the installed `453e106` hrtimer fixes as well as the local
  `99b8459` base and dynamic reservation changes.
- Initramfs updated. `/usr/src/davinci_ascend-1.0` resolves to the updated
  `/srv/ai/src/ascend-310p-host-driver` DKMS source tree.
- Clean unload completed. The old vNIC teardown emitted a NAPI warning, and
  subsequent live reload timed out waiting for firmware boot state. One
  authorized host reboot recovered all four NPUs.
- Original modules, source archive, module hashes, and logs are retained in
  `ple-driver-deployment-v1` on the hardware host.

## Hardware results

| Check | Result |
| --- | --- |
| Normal startup | All four devices report `reserved=0` |
| Simultaneous acquisition | Four leases, channels 20/25 on each Duo |
| Queue geometry | 65,536 SQ/CQ entries; two host traffic channels remain |
| Duplicate owner | Rejected with `EBUSY` |
| Stale release | Rejected with `ESTALE`; valid lease remains queryable |
| Explicit release/reacquire | Pass; generations increase |
| Last-close cleanup/reacquire | Pass |
| Final state | All four devices report `reserved=0` |
| Ordinary DMA without leases | 16 exact page-crossing 320-byte copies pass |
| ACL initialization with leases | Fails: fixed-instance copy hits reserved channel |

The ordinary DMA test covered shared writable/read-only mappings, a sealed
memfd, and ACL host allocations on all four NPUs. The coexistence failure was
ACL error 507033 with driver `EBUSY` on a 1-MiB initialization copy. Every test
lease was closed. The tested guard prevents interference with the leased queue,
but transparent coexistence with vendor fixed-instance callers is not solved.

The driver advertises `capabilities=0` and `device_submit=0`. Queue mapping,
device producer/doorbell access, completion ordering, full-table registration,
and autonomous PLE fetch remain unimplemented or unqualified. This deployment
does not eliminate host PLE lookup and does not establish an inference speedup.

## Collective bootstrap correction

A fresh context produced identical exported strings for `inbox0`/`ready0` and
`inbox1`/`ready1`. Memory and Notify use separate SDK namespaces and import APIs.
The prior global-distinctness check rejected valid exports after reboot.

The N-rank and pair adapters now enforce distinct slots within each namespace.
Import and teardown still use the corresponding typed SDK APIs. All 180 focused
collective unit tests pass, including cross-namespace equality and same-namespace
rejection. The N-rank correction passed four-rank server startup and generation;
the pair adapter change has offline coverage only. Sixteen startup-client tests
also pass. Earlier driver build/lifecycle tests remain recorded separately.

## Service restoration

Public identity remains `qwen38-flash-next`, port `8001`, Kilo entry
`vllm-310p-w4/qwen38-flash-next`. The context limit remains 262,144. Startup
reports 1,068,936 cache tokens and 4.08 concurrency at that limit. The slowest
rank loaded the model in 114.11 seconds.

Current run directory: `resident-live-v8`. Stable hardware log:
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/server-current.log`.
The earlier failed bootstrap is preserved in `resident-live-v8-ipc-failure`.

`restore-retained-v8.py` restores the retained W4/HC/QSA/gather controls, graph
sizes, device transition/history/output observer, queue depth two, device drafts,
compact tables, GDN metadata, device preparation, host metadata removal, fused
checkpoint preparation, lazy readback, and device commit inputs. Each changed
execution path is qualified before enabling its normal mode.

All retained controls passed restoration and inspection on all four workers.
Graph inventory confirms shapes `[3, 6, 9, 12]`; the engine queue depth is two,
checkpoint backend is fused, lazy readback is enabled, and device commit inputs
are enabled with verification mode off after successful exact checks. The
engine control session is healthy, and the server is unpaused.

All 12 final serving cases passed: token budgets, penalties, log probabilities,
request turnover, structured JSON, stream cancellation, and generation after
cancellation. Cooling paused the test client during two turnover cases, so their
108-second wall times include that pause and are not performance measurements.
The hottest observed NPU reached 86 C; all devices reported OK, and load was
paused for cooling before completing the remaining work.

Separate short throughput measurements after restoration:

| Metric | Installed candidate | Saved v7 run |
| --- | --- | --- |
| C1 HTTP output tokens/s, median of three 256-token requests | 32.385 | 33.048 |
| C4 aggregate HTTP output tokens/s, four 256-token requests | 62.382 | 64.067 |
| C4 median per-request streaming decode tokens/s | 16.969 | 17.404 |

These are approximately 2–3% below the historical run. This is not a controlled
A/B experiment and provides no speedup claim. HTTP throughput includes prefill
and API overhead. The candidate is retained without automatic reservations.
No benchmark or profiler is left running.

Receipts: `resident-live-v8/final-driver-restoration.json`,
`resident-live-v8/requalification/scheduling-serving-gate-driver-restored-v1.json`,
`qsa-resident-v1/decode-benchmark-driver-dynamic-v8-v1.json`, and
`qsa-resident-v1/concurrency-driver-dynamic-v8-v1.json`. Final reservation status
is in `ple-driver-final-reservations-v1.json`; all four entries are unreserved.

## Rollback

On the hardware host, stop the Qwen worker group before rollback. Run as root:

```sh
python3 /srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/ple-driver-deploy.py rollback
systemctl reboot
```

This restores the backed-up module files, runs `depmod`, overlays the original
tracked source snapshot, and updates initramfs. Candidate-only source files
remain unreferenced by the restored Makefiles. Do not run this procedure over
later independent driver changes without reconciling them first. Relaunch the
same model identity and restore retained controls using a fresh run directory.
