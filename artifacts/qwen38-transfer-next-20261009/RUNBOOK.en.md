# Deferred transfer-candidate validation

[中文说明](RUNBOOK.zh.md) · [Findings and limits](REPORT.en.md).

Server startup and NPU work remain deferred. Commands below are preparation
and later validation instructions, not authorization to run device tests now.
Use a complete, committed runtime snapshot with the existing image-enabled
TP4/EP4 profile. Keep model, cache capacity, graph sizes and thermal policy
fixed when comparing candidates. Do not copy isolated files into a live runtime.

## Host-only preparation

From the matching serving Python environment and complete source snapshot:

```bash
python -m tools.qwen4exp.build_native_transfer \
  --source-root /path/to/complete/runtime \
  --cann-root /usr/local/Ascend/ascend-toolkit/latest \
  --build-dir /path/to/new/append-only/build --version 1
python -m tools.qwen4exp.prepare_native_transfer \
  --build /path/to/new/append-only/build \
  --runtime /path/to/complete/runtime --output /path/to/transfer-manifest.json
```

These commands compile/fingerprint only. A successful host build exists at
`/home/matteius/experiments/qwen-transfer-next-20261009/build-v1c` on the
serving host. Its `source` directory contains only build inputs and is not a
serving snapshot. The namespace is `qwen_transfer_v1`; use a new version and
update candidate bindings if a previously loaded binary changes. Append-only
build directories prevent overwriting compiled provenance.

## Gates after renewed NPU authorization

First run the component validator on an isolated authorized diagnostic device:

```bash
python -m tools.qwen4exp.benchmark_transfer_next_310 \
  --build /path/to/new/append-only/build --output /path/to/new/component-receipt.json
```

This executes native kernels and is deferred. It requires the coherent FP32
GDN and W4 OPP stack. It checks state gather/scatter, full GDN H/O output and
final-state parity at 4/12 and 16/48 heads, and W4 gate/up and down parity with
partial row tiles, empty experts and a peer tail. It does not measure speed,
images, capacity or sustained temperature.

On an authorized diagnostic server, use
`--worker-extension-cls tools.qwen4exp.resident_worker.QwenResidentExtension`.
The existing resident harness loads the manifest and applies one candidate
while drained. It may reset prefix caches and recapture graphs; do not run it
against a live demo or during a thermal hold. Keep all four worker receipts.
No operator search paths or model weights need to change.

| Candidate | File | Preconditions |
| --- | --- | --- |
| Bounded telemetry detail | `resident_candidates/transfer_audit.py` | Optional runner/W4 records; prefix totals already available |
| Prefix phases | `resident_candidates/prefix_phase_batching.py` | All tiers on one worker device; no interleaving forward |
| Native state IO | `resident_candidates/native_state_layout.py` | `qwen_transfer_v1`, FP32 cache and unique/in-range slots |
| W4 metadata cache | `resident_candidates/cached_w4_metadata.py` | `qwen_transfer_v1`, grouped native INT4, 128 local experts, eight metadata lanes |
| TP prefill pipeline | `resident_candidates/tp_prefill_pipeline.py` | Native grouped INT4, TP-sharded shared expert, identical chunk size on all ranks |

Paths in this table are relative to `tools/qwen4exp`. The corrected fused-WY
candidate uses `qwen_prefill_v2` from the earlier build and now targets the
actual Qwen model method. Regenerate its manifest against this runtime.
Switches unwrap an earlier candidate on the same method; they are not an
implicit way to combine candidates. Capture a baseline after installing the
telemetry candidate and before requests; comparing across a switch is rejected.

## Saved telemetry receipts

Read-only status requests on that later authorized server:

```bash
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 status > before.json
# Run the authorized, matched workload without configuration changes.
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 status > after.json
python -m tools.qwen4exp.transfer_snapshot \
  --before before.json --after after.json --output transfer-deltas.json
```

The last command is entirely offline. Require four ranks, stable worker PIDs,
configuration and storage identities. Counters remain cumulative over cache
resets; inspect drain reasons to exclude administrative operations from an
inference measurement. Recent events may be truncated, while totals remain.

Then compare unique cold and proven warm long prefixes, C1/C2/C3, mixed prefill
with active decode, fresh/cached images, thinking/tool calls, prefix CoW,
cancellation and rollback. Verify unchanged image/capacity behavior and MTP
acceptance. Profile all ranks' memcpy bytes/directions, event boundaries,
HCCL and native MTE behavior against the thermal clock.

Keep the staged 94°C request hold, resume only when every sensor is at most
85°C, and the independent 96°C cutoff. Hold ownership and missing-sensor
behavior remain unchanged. A sustained thermal gate is required before any
throughput recommendation or deployment. Combine individually qualified
candidates only with another full gate; do not start the server now.
