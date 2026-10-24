# M3.1 Flash: six-chip 310P serving and resident reload

The 219.16 GiB signed-W4 export is transferred and serving real weights on
`threadripper:8000` as `minimax-m31-flash`. The source checkpoint revision is
`54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`; upstream vLLM is
`3ab5dda29acabea01f6a63d0806bdbbb4a27bde5`. Conversion was delivered in
`704c3f8f30c5f306f4aa5f70385ecb632cc49a1f`.

## Qualified configuration

TP2, PP3 (21/20/19 layers), stage-local EP2, eager execution, one 8192-token
sequence, 128-token prefill chunks, one image capped at a 448-pixel long side.
The canonical checkpoint is packed into NZ256 expert buffers at load time;
the current grouped kernel supports the selected 512-route cap. The service
uses the `minimax_m31` reasoning parser to prevent older chat-template end
markers from leaking into streamed output.

Run `serve-resident.sh` from `/workspace`, with `runtime-env.sh` installed as
`/workspace/minimax-m31-env.sh`. The transient systemd unit is `minimax-m31`;
it is not installed as a boot-enabled service. Its log is
`/vllm-workspace/minimax-resident.log`. Existing Qwen services remain stopped.
The complete setup, limitations and control commands are in the
[model runbook](../../docs/source/tutorials/models/MiniMax-M3.1-Flash-310P.md).

The runtime is Python 3.12.13, Torch 2.13.0+cpu, torch_npu 2.13.0rc1,
transformers 5.17.0 (unchanged), and CANN 9.1.0. The two W4 operators were built
from the current source and combined with the existing qualified non-W4 OPP
operators; `merged-opp.json` records hashes. Model allocations per rank were
37.79, 38.26 and 37.55 GiB across the three stages, before cache/workspace.
All six loader ranks verified payload hashes and parameter completeness.

## Evidence

| Check | Result | Receipt |
| --- | --- | --- |
| Final coherence after all reloads | 9/9 passed, including red image and 3443-token retrieval | `final-coherence-report.json` |
| Near-8K sparse retrieval | Correct `amber cloud`, 7943 input tokens, 168.99 s | `resident-8k-retrieval/report.json` |
| Current NZ256 expert projection vs CPU | Finite, max normalized RMS error about 1.5e-5, including 512 routes on one expert | `expert-current-nz256512.json` |
| Six-rank Python/native switch and rollback | Passed; same PIDs and weight-storage digests | `resident-native-gate/report.json` and per-operation receipts |
| Direct CANN standalone reference | Passed, max normalized RMS error 0.000372 | `native-v4-standalone.log`, `native-v4-manifest.json` |
| CPU regression tests | 200 passed, 10 skipped | `final-cpu-tests.log` |
| Changed-file lint | Passed | `changed-files-lint-final.log` |
| Repository-wide `bash format.sh ci` | Failed on existing unrelated formatting/import violations | Local `format-ci-final.log` |

The ten CPU skips include six reasoning-parser tests requiring the pinned newer
vLLM; all six passed separately in the serving environment. The other four are
existing conditional skips. Hardware-specific behavior was also exercised on
all six actual 310P chips. Dummy text/image/8K-boundary tests preceded real-weight
loading, but dummy responses are not used as quality evidence.

At matched settings, greeting latency improved from 72.31 s to 8.75 s,
France-capital latency from 74.88 s to 15.26 s, and addition from 59.45 s to
11.01 s after current kernels, NZ256 packing and larger grouped batches.
These are individual requests, not sustained throughput measurements.
The real-expert microbenchmark improved from 1.81 s to 0.134 s at 128 tokens;
that ratio is not a full-model throughput claim.

## Resident reload contract

One startup installs `MiniMaxResidentExtension` and loopback-only administrative
controls. Subsequent compatible changes use existing workers and allocations.
With no active request, the tested Python switch took 0.147 s; native activation took 0.150 s and
rollback 0.135 s. These times exclude compilation/native qualification.

The controller drains requests, prepares identical source on all ranks, clears
request/KV state, patches functions and call-site aliases, and verifies rank,
PID, generation and storage identity before resuming. Packed expert buffers are
included in the fingerprint. Failed mutations remain paused. CLI controllers
are serialized, and the public control-route check returned HTTP 403.

The direct native build uses a unique Torch namespace and ACLRTC kernel entry,
owned binary handles, explicit workspace and immutable hashed artifacts. It
bypasses process-cached OPP discovery. Native code is append-only; rollback
restores dispatch without unloading handles. A native/device failure requires
worker recovery, potentially restart. Weight layout, topology, model allocation
and graph-mode changes still require startup. Eager Python/CANN changes do not.

The final post-reload nine-case suite passed again, and `final-resident-status.json`
confirms the original six worker PIDs and weight-storage identities.
All chips report healthy in `final-npu-smi.txt`.

The final baseline uses the qualified OPP kernel; the direct native kernel was
loaded, activated, checked and rolled back. It remains available for resident
experiments as `minimax_w4_v4`. Startup defaults remain independently usable.

## Rejected experiments and remaining limits

- The old OPP kernel uses 16x128 NZ tiles; interpreting NZ256 with it failed.
  Old large-batch concentrated routing also produced nonfinite output. These
  candidates were replaced by the current tested kernel, not enabled as-is.
- Loading an ordinary OPP binary directly initially failed because of missing
  argument metadata and hidden workspace state. Those failures were isolated
  to standalone processes. The qualified direct build recompiles an explicit
  workspace entry and validates before resident activation.
- The short-context attention fast path passed isolated comparisons and a
  real-activation shadow (390 calls, maximum normalized RMS error about 4e-5),
  but produced incoherent output when active. It failed again under resident
  reload and was rolled back. The discrepancy is unresolved; small local
  numerical error does not establish full-model correctness.
- Unformatted raw completion was repetitive. Use templated chat; no broad
  lm-eval, native FP4 activation-quantization parity, or long-context benchmark
  score is claimed. The two retrieval prompts are bounded checks.
- ACLGraph, sequence-parallel MoE/FlashComm1 and speculative decoding are
  explicitly rejected by this eager adapter. The separate DSpark/MTP checkpoint
  was not converted. Video is disabled in this profile.
- The offline 128K x 16 capacity estimate is 92.57 GiB on the busiest rank and
  does not fit these ~46 GiB banks. Current capacity is 8K x 1.

Large source archives, full traces and unsuccessful build logs remain on the
working hosts. This directory commits the compact receipts needed to assess
the delivered behavior; it does not include model weights or native binaries.
