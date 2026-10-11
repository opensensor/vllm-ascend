# Qwen long cached-prefix top-k memory repair

## Failure and change

A 227,420-token request resumed from 223,872 cached tokens and scheduled a
2,560-token prefill chunk. QSA scoring bounded its GEMM intermediates but
concatenated every score tile before calling top-k. The top-k backend requested
another 2.19 GiB on ranks 0, 2, and 4; EngineCore exited and three peer workers
remained orphaned. Reserved PyTorch memory was close to allocated memory, so
this report does not diagnose allocator fragmentation as the root cause.

The repair selects each GEMM query tile immediately after causal masking and
retains only compact group IDs. It also bounds backend selection calls for the
native scoring fallback. The selection score budget is 4,194,304 elements
(16 MiB FP32); the independent four-head GEMM budget remains in place. At 256K
context the selection tiles have at most 64 query rows. Each row continues to
score every compression group for its request. No group-axis approximation,
CPU offload, new environment variable, or device-to-host metadata read is added.
The existing stable and fast selection policies remain available. The fast
policy retains its preexisting unspecified set membership at cutoff ties.

The native scoring fallback still materializes its score output; its backend
selection is bounded. This repair is not a global memory ceiling for all model
operators, vision processing, recurrent archives, or six full context windows.

## Offline and operator validation

- 180 scoped CPU tests passed, including tile bounds, request/page mapping,
  scratch exclusion, causal masking, stable ties, fast-policy behavior, graph
  ownership, and empty-update guards.
- The direct 310P probe used 2,560 query rows, four 128-dimensional index heads,
  56,640 compression groups, and a 223,872-token cached-prefix geometry.
  Peak additional PyTorch allocated memory was 191,295,488 bytes (182.43 MiB).
  Twelve sampled rows matched the existing full-width selection routine.
- The weight-free probe elapsed time was 41.568 seconds including cold operator
  initialization. It is not a warmed serving throughput measurement.
- A repeatable NPU workspace benchmark is provided at
  `tests/e2e/nightly/310p/single_node/ops/benchmark_qsa_prefill_workspace_310.py`.
- Scoped manual pre-commit checks passed. Repository-wide `bash format.sh ci`
  failed on existing unrelated lint/spelling findings; 144 incidental formatter
  edits were restored. The whole repository is not reported as lint-clean.

## Real-weight recovery

The complete isolated runtime is
`/srv/ai/src/qwen-long-prefill-topk-runtime-20261011`. The selected profile is
`/srv/ai/src/qwen-long-prefill-topk-tp6-20261011`, with server and thermal logs
under that profile directory. It retains TP6/EP6, six request slots, MTP2,
256K context, graphs `[3,18]`, image processing, and the existing 94/85 C hold
policy. All six workers acknowledged the admitted fast candidate with clean
graphs and no native failure. Maintenance is released, the server is unpaused,
and a non-loopback request returned the exact `ASCEND_OK` marker.

| Real-weight request | Total prompt tokens | Cached tokens | Elapsed seconds | Exact answer |
| --- | ---: | ---: | ---: | --- |
| Cold background | 223,900 | 0 | 674.431 | 1763 |
| Extension through failing path | 227,448 | 223,872 | 28.639 | 1763 |
| Repeat | 227,448 | 227,328 | 15.646 | 1763 |

These are whole-request elapsed times, including five output tokens; they are
not streaming TTFT measurements. The extension used the production 2,560-token
prefill chunk setting with the same 223,872-token prefix as the failure. Fresh and
cached 1,024-pixel images, marker, arithmetic, and Python-generation probes
passed. Six simultaneous sessions returned all 18 expected arithmetic answers
across three rounds, including prefix-cache reuse. Six-chip health was verified
through actual requests and worker RPC acknowledgments.

After the long-context stress, six rotated 256-token coding benchmark requests
produced a median of 28.625 tok/s. This is about
-1.5% relative to the previously qualified
29.066 tok/s median. This single post-stress set does not establish a speedup
or a regression attributable to the repair. It is
48.6% above the historical two-card reference;
the fresh set does not independently meet the strict 50% threshold. No matched
old-code replay at the crashing context was attempted. Peak sampled temperature
was 88 C over 567 samples, with no thermal holds or sensor errors.

The cached repeat still spent 15.6 seconds processing a 120-token tail at this
context. Investigating the native-score versus GEMM crossover for short tails
is a separate performance candidate, not part of this correctness repair.
Six completely occupied 256K contexts have not been stress-tested.

The failed request cannot be recovered from the dead engine. Recovery used a
restart with fresh caches and the repaired code. The one-time large validation
bootstrap has been replaced on disk by the normal admitted-profile bootstrap.
Future starts use:

```bash
bash /srv/ai/src/qwen-long-prefill-topk-tp6-20261011/start-server.sh
```

Follow the current log with:

```bash
tail -f /srv/ai/src/qwen-long-prefill-topk-tp6-20261011/server-8001.log
```
