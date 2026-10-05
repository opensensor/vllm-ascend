# QSA AI CPU metadata experiment

The qualified Qwen3.8 Flash-Next W4A8 serving path is unchanged. This
isolated candidate computes four device-resident INT32 arrays from causal
query positions: visible compressed groups, selected group count, causal-tail
start, and causal-tail count. It tests whether one AI CPU call can replace the
small tensor operations in `_qsa_position_geometry` and the subsequent group
count clamp. It does not score QSA groups, select top-k, gather KV, or change
attention arithmetic.

## Why this candidate

The prior exact AI CPU top-k gate took 114.616 ms at 2,048 queries against
26.660 ms for the current fast selection path. The AI CPU is not a useful
replacement for that wide selector. Device metadata is much smaller and
branch-heavy; one fused pass is the remaining narrow candidate. A decode
trace also recorded repeated AI CPU `Cast` tasks, but those tasks have not
been attributed to this exact geometry path. No speedup is claimed from their
count.

The code lives in `csrc/attention/qsa_position_metadata_aicpu_v310/`. The
kernel shares `position_metadata.h` with a host parity gate. The output is a
contiguous `[4, queries]` tensor; each row is one output field. INT32 output
matches the qualified QSA custom-op interfaces. INT32 overflow in the tail
start is rejected rather than wrapped. Serving positions stay well inside the
range.

The pure C++ loop took about 0.0007 ms for three rows and 0.0043 ms for 2,048
rows on the Threadripper (median of five host trials). These are host
feasibility timings, not 310P predictions. They suggest kernel arithmetic is
small compared with the expected device launch and synchronization cost.

## Gates

Host-only:

```bash
python3 -m pytest --noconftest -q tests/ut/qwen38_1m/test_qsa_position_metadata_aicpu.py
python3 tools/qwen4exp/benchmark_qsa_position_metadata_aicpu_310.py --dry-run
```

Build an isolated custom OPP from a matching source snapshot with the CANN
9.1 toolchain. Do not overlay its host API on the qualified serving vendor:

```bash
cd csrc
bash build.sh --pkg --soc=ascend310p \
  --ops=qsa_position_metadata_aicpu_v310 \
  --vendor_name=qsa_position_metadata_probe -j8 -O3
```

The corresponding extension must be rebuilt from that snapshot. With the
isolated OPP first in its operator path, run the gate on an idle 310P:

```bash
bash tools/qwen4exp/run_qsa_position_metadata_gate.sh \
  /path/to/new/qsa-position-metadata.jsonl
```

The gate compares exact outputs and full call latency against the existing
PyTorch operations at 3, 64, 256, and 2,048 queries. Include AICPU launch
cost. Test changing input values in a second call. A later integration gate
would also need a device trace to confirm execution and preserve dynamic
positions across requests and graph replay. End-to-end cold TTFT and decode
must be checked independently.

The 2026-10-04 host gate passed 8 tests. An isolated CANN 9.1 build on the
310P host produced `cann-ops-transformer-qsa_position_metadata_probe_linux-x86_64.run`
with SHA256
`4b55087d88d37b47297999779a6c71923ea3bb43cb0ac5031c35412337102884`.
The matching extension built with SHA256
`14083d6eeefe2695acb2dffb824e66bfef8033893c0f2de204384480a97762a9`.
Both are staged only under the isolated
`/srv/ai/src/qsa-position-metadata-probe-20261004` snapshot.

## Isolated 310P result

The one-card gate passed exact parity in all eight combinations of query count
and sequential/padded positions, including changed-input calls. The package
declares the custom kernel as `DNN_VM_AICPU`; a device profiler trace was not
captured. The benchmark times 20 repeated calls per trial with one device
synchronization after each batch, over five trials. Each number is the median
amortized call latency including launch, output views, and completion. The
PyTorch baseline returns the same four-tensor tuple as the candidate. An
earlier gate included an extra `torch.stack` in its baseline and is superseded
by the corrected result below. Both raw gates are retained under
`artifacts/qwen38-qsa-position-metadata-20261004/`.

| Queries | Candidate, sequential/padded (ms) | Torch, sequential/padded (ms) |
| --- | ---: | ---: |
| 3 | 0.0427 / 0.0424 | 0.1069 / 0.1038 |
| 64 | 0.0418 / 0.0418 | 0.1067 / 0.1047 |
| 256 | 0.0424 / 0.0444 | 0.1082 / 0.1008 |
| 2,048 | 0.0689 / 0.0703 | 0.0954 / 0.0925 |

This is a real operator-level gain, but it is too small to address cold
prefill. At 2,048 queries it saves 0.022-0.026 ms per call. If all twelve QSA
layers called it independently over roughly twenty chunks of a 40K prompt,
the arithmetic upper bound would be about 5-6 ms saved against the measured
roughly 120 s cold TTFT. The dominant QSA attention and selection work is
about 74 s. A three-row decode call saves about 0.06 ms, so a fully serial
twelve-layer decode could save at most about 0.7 ms per token; graph capture,
overlap, and actual call frequency could lower that. No serving path was
changed and no end-to-end TTFT or decode speedup was measured. Further NPU
testing is deferred at the user's request.

## Other AICPU proposals

- Cache/page planning: the current Qwen n-gram table is demand-paged from host
  checkpoint shards. The AI CPU cannot remove host disk and PCIe access. An AI
  CPU page planner would only be worth a prototype if a device trace shows
  repeated synchronous *device-resident* page-list work on the critical path.
- MoE route planning: the prior 128-expert route-lookup candidate improved an
  adverse synthetic projection by up to 8.5%, but the paired serving benchmark
  showed no throughput gain. A new AI CPU launch before both projections adds
  a dependency and is not justified without a trace showing duplicated route
  planning dominates that launch cost.
