# Deferred Qwen candidate validation

## Offline preparation

Server startup and NPU tests remain deferred. The queued configuration is
`queued-profiles.json`; all five candidates are off by default. Preserve the
image-enabled base profile and existing capacity. Make one complete runtime
snapshot containing the previous memory fixes and this change; do not mix
individual model files into the old serving snapshot.

The final host build is on the development host at:

```text
/home/matteius/experiments/qwen-five-offline-20261008/build-v2
```

It contains three device binaries, `qwen_prefill_v2.so` and provenance.
They have not been loaded or executed. To rebuild from a complete source tree,
choose a new directory and a unique version; update `RESOURCE_NAME` in both
resident candidates to match that version before creating a manifest.

```bash
python -m tools.qwen4exp.build_native_prefill \
  --source-root /path/to/complete/snapshot \
  --cann-root /usr/local/Ascend/cann-9.1.0 \
  --build-dir /path/to/new/build-v2 --version 2
python -m tools.qwen4exp.prepare_native_prefill \
  --build /path/to/new/build-v2 --runtime /path/to/complete/snapshot \
  --output /path/to/new/native-manifest.json
python -m tools.qwen4exp.speculation_sweep > mtp-arms.json
```

These commands compile or write files only. Manifest creation verifies the
callback seam, resource version, frozen sources and binary fingerprints.
Manifest loading is a separate device operation and remains deferred.

## After explicit NPU access

First qualify the preceding P1 changes independently. Retain the 94-C hold,
85-C resume and 96-C hard stop. Confirm sensor coverage for every device;
thermal control cannot interrupt an already executing kernel. Standalone
component tests require an idle NPU and external thermal supervision.

From the complete snapshot and matching installed CANN/OPP environment:

```bash
python -m tools.qwen4exp.benchmark_prefill_next_310 \
  --case qsa --tokens 128 --context-tokens 8192 --output qsa-gate.json
python -m tools.qwen4exp.benchmark_prefill_next_310 \
  --case wy --native-build /path/to/build-v2 --tokens 128 --output wy-gate.json
python -m tools.qwen4exp.benchmark_prefill_next_310 \
  --case routes --native-build /path/to/build-v2 --tokens 128 --output routes-gate.json
```

These tests execute kernels. WY also compares the complete existing H/O kernels
and FP32 final state at both production head geometries. Route tests require
exact packed-prefix parity with the native quantizer. Increase token counts
only after the small gate passes. Hardware timings exclude D2H assertions.

For a separately authorized diagnostic server with the Qwen resident worker
extension, use the existing controller to load and validate the manifest,
then switch one candidate at a time. These are device and server mutations:

```bash
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 \
  load-native /path/to/native-manifest.json
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 \
  switch --mode graph --candidate tools/qwen4exp/resident_candidates/fused_wy.py
```

Use `local_routes.py` for the separate route candidate, against a baseline
already using native INT4 and `cann_swiglu_pack`. Do not attribute a change
from serving `cann_builtin_fp16` to route compaction alone. The controller
performs its established drain/reset/recapture transaction. Preserve manual or
thermal holds, and do not perform a switch while requests are thermally held.

The QSA candidate is an explicit HF override:

```json
{"text_config":{"ascend_qsa_prefill":{"backend":"paged_native"}}}
```

The pacing candidate requires startup with:

```text
--scheduler-cls vllm_ascend.core.qwen_prefill_scheduler.QwenPrefillPacedScheduler
--additional-config '{"qwen_prefill_pacing":{"target_step_ms":500}}'
```

Neither example authorizes startup now. Keep image limits and prefix retention
enabled. Test text, fresh images, cached images, ragged batches, warm prefix
reuse, cancellation and a long prefill arriving during decode. Record existing
decoder token gaps, queue time, cache/spill warnings and temperatures.

## MTP0/1/2 collection

Each depth uses its own cache plan and two capture sizes from `mtp-arms.json`.
For MTP0 omit speculative configuration entirely. Do not live-switch the depth
by editing a Python scalar. Verify the launch configuration matches the claimed
draft length and serving alias, and record its checksum with the quality gate.

For each already-running arm, provide one identical request JSON with
`max_tokens >=64` and a real quality receipt containing the matching `model`,
`draft_length`, `quality_pass` and `image_pass`. Run the collector on the
serving host with the correct API PID and the external controller's JSONL log:

```bash
python -m tools.qwen4exp.benchmark_speculation_sweep \
  --base-url http://127.0.0.1:8001 --model YOUR_SERVING_ALIAS --api-pid API_PID \
  --draft-length 0 --request matched-request.json \
  --quality-receipt mtp0-quality.json --thermal-log thermal-controller.jsonl \
  --output mtp0.jsonl
```

Repeat for depths 1 and 2 after their separate startup gates. The collector
runs at least 600 seconds per concurrency, with three repeated windows;
it cannot restart, pause or reconfigure a server. A changed API identity,
incomplete arm, missing sensors or early resume prevents qualification.
Client stream chunks approximate decode timing. Wall throughput includes
prefill, queueing and cooldown, so it is the preferred comparison when supplied.

```bash
cat mtp0.jsonl mtp1.jsonl mtp2.jsonl > matched-arms.jsonl
python -m tools.qwen4exp.speculation_sweep --records matched-arms.jsonl > comparison.json
```

Match the request bodies, output caps and cold/warm cache protocol. The collector
saves prompt token counts and output hashes; inspect server timing/cache logs
before asserting a cold-prefill gain. Quality and image receipts are independent
evidence, not assertions made by the performance collector. Add actual energy
measurements separately; absent energy remains unknown.
