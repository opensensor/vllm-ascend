# Qwen W4 grouped MoE finalizer experiment

The qualified four-card Qwen service remains unchanged. The model now accepts
`"grouped_finalize": "cann_v2"` inside `ascend_expert_quantization` as an
explicit, experimental opt-in for the native INT4 grouped path. The default
is `"torch"`. The new path calls the installed `torch_npu.npu_moe_finalize_routing`
with FP16 sorted expert rows, FP16-converted `[tokens, top_k]` route weights,
INT32 inverse row indices, and `drop_pad_mode=2`. It widens the result back to
FP32 before the existing TP reduction. No new custom OPP is needed. The
installed 310P CANN build rejected FP32 scales with FP16 rows: its tiler
requires them to have the same dtype. A direct FP16-scale probe succeeded.

The current grouped epilogue converts the FP16 routed result to FP32, gathers
route weights in expert order, multiplies, restores token order, and sums the
ten routes. At a 2,048-token chunk and hidden size 2,560, the routed tensor is
100 MiB in FP16, so removing FP32 intermediate passes is worth a direct gate.
The small decode path already has a fused custom down/reduce operator; this
experiment does not alter it.

The CANN finalizer rounds the route weights and weighted sum to FP16 before the
existing FP32 TP reduction, unlike the established FP32 epilogue. Peer-owned
routes must be zeroed by the grouped projection before finalization; the
finalizer's `drop_pad_mode=2` does not accept an invalid peer-row index.

## One-card gate

Run from an isolated runtime snapshot with this source file and the matching
benchmark script. The operator gate does not change the serving launcher or
qualified OPP.

```bash
python tools/qwen4exp/benchmark_w4_finalize_routing_310.py --dry-run
ASCEND_RT_VISIBLE_DEVICES=0 python tools/qwen4exp/benchmark_w4_finalize_routing_310.py \
  --graph-replay --output /path/to/new/w4-finalize-gate.jsonl
```

The script checks the current epilogue against the candidate at 3, 12, 512,
and 2,048 tokens with ten routes each, for mixed local/peer and all-peer
patterns. It checks changed inputs and, at 3 and 12 tokens, changed-input
graph replay. Timings include the INT64-to-INT32 index conversion, FP32-to-FP16
weight conversion, and FP16-to-FP32 output conversion. Each trial synchronizes
after five calls. It records numerical error and full epilogue latency.

## Results on 2026-10-04

The one-card gate passed parity at all tested shapes, including exact zero for
all-peer rows and changed-input graph replay at 3 and 12 tokens. At 2,048
tokens, the complete grouped epilogue fell from **7.16 to 1.00 ms**. A
real-weight layer partial with synthetic activations fell from **59.55 to
53.42 ms** at 2,048 tokens, with relative L2 error **0.000437**.

Three paired TP4 cold requests of 23,410 prompt tokens each reported zero
cached tokens. Median TTFT fell from **72.86 to 69.37 seconds**, a **4.8%**
reduction. Generated text differed by one short phrase in two of three pairs;
MTP acceptance also changed in those pairs. This is a useful speed result,
but the opt-in is **not qualified for production**. A broader quality and
logit comparison is needed before promotion. The paired service run also had
one arm order, baseline then candidate; a reverse-order repeat would help
separate clock or thermal drift from the measured gain.

Raw results and the service logs are in
`artifacts/qwen38-cann-finalize-20261004/`. An existing generic MoE path in
this repository records an accuracy issue with `npu_moe_finalize_routing`;
Qwen's quality gate must account for that history. Host tests:

```bash
python3 -m pytest --noconftest -q tests/ut/qwen38_1m/test_w4_grouped_finalize.py \
  tests/ut/qwen38_1m/test_w4_grouped_native.py
```

## Profile follow-up

The original October 4 A/B did not record a kernel trace. A later one-card
follow-up captured separate current and CANN traces for the epilogue and a
real-weight layer. See
`artifacts/qwen38-cann-finalize-20261004/profile-followup-20261004/README.md`.
The four-rank service capture remains open. Do not run these commands while
another test owns the devices. Use the same isolated source and CANN
environment as the A/B. Each `--output` file and `--trace-dir` directory must
be new.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.qwen4exp.benchmark_w4_finalize_routing_310 \
  --graph-replay --output /path/to/new/epilogue.jsonl \
  --trace-dir /path/to/new/epilogue-traces
ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.qwen4exp.benchmark_w4_finalize_layer_310 \
  --model /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --output /path/to/new/layer.jsonl --trace-dir /path/to/new/layer-traces
```

Both scripts time without the profiler first. They then capture three calls
per arm at 2,048 tokens, in separate `torch` and `cann_v2` directories. After
sourcing the same CANN environment, parse and summarize each raw trace:

```bash
python -m tools.qwen4exp.profile_runtime analyse /path/to/new/epilogue-traces/torch
python -m tools.qwen4exp.profile_runtime analyse /path/to/new/epilogue-traces/cann_v2
python -m tools.qwen4exp.summarize_trace /path/to/new/epilogue-traces \
  --output /path/to/new/epilogue-summary.json
```

Repeat those three offline commands for `layer-traces`. Compare the kernel
names, counts, and device task durations for the baseline's weight multiply,
inverse gather, and route reduction against the CANN finalizer. The traced
durations explain work distribution; the unprofiled JSONL timings remain the
latency measurements. The real-weight layer still uses synthetic activations
and one TP rank.

To repeat the one-card capture, use the commands above. For a service trace,
start each arm from its isolated launcher copy with a
different `--profiler-config` added to its `serve_cmd`. Generate the JSON for
each arm with `profile_runtime config --phase cold-prefill --steps 4
--trace-dir /path/to/new/service-arm-traces`. Once the server is ready, the
matching capture command is:

```bash
python -m tools.qwen4exp.profile_runtime capture \
  --phase cold-prefill --steps 4 --trace-dir /path/to/new/service-arm-traces \
  --output /path/to/new/service-arm-capture --execute -- \
  python -m tools.qwen4exp.benchmark_w4_finalize_service \
  --arm torch --cases 1 --skip-warmup --output /path/to/new/service-arm.jsonl
```

Use `--arm cann_v2` for the candidate. The profile starts before the single
unique long prompt, records up to four early prefill iterations, and stops
after the response. Confirm the recorded shapes are prefill chunks before
attributing kernels. Parse each arm's trace separately. This is a diagnostic
capture, so its TTFT must not be compared with the unprofiled A/B numbers.
