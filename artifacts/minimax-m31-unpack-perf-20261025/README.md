# M3.1 unpacking and attention experiments

The starting service used v7, TP2/PP3/EP2, 512-token prefill chunks, 2048 grouped
routes, and the 163840-token cache allocation. Full 160K testing remains with
the user; this iteration runs bounded operator and short serving checks.

## Native kernel comparisons

V9 enables the existing guarded biased-RINT field extractor. It computes the
same signed W4 values using fewer vector conversions. The packed-byte release
event after Cast is retained: an older experiment demonstrated corruption when
that event was absent. No checkpoint, packing, scale, clipping, routing or
attention arithmetic changes are involved.

Each candidate was compared against v7 on 15 projection/routing cases per rank,
using all six resident weight banks, two warmups and ten timed iterations.
All 90 v9 comparisons were bit-identical. An additional 24 varying-K one-hot
checks per rank exercised both nibbles, all byte values across output rows,
multiple 256-K tile boundaries and repeated buffer reuse: all 144 passed.
Across-rank/projection median speedups were 1.083–1.107x for four-route cases,
1.100x for balanced 512-route calls and 1.092x for balanced 2048-route calls.
Concentrated 2048-route calls were essentially unchanged. These are projection
microbenchmarks, not whole-model throughput measurements.

V10 adds four-core expert teams to the same unpacking change. It also passed
all output checks and slightly improved balanced prefill, but concentrated
2048-route calls took about 1.90x as long as v7. V10 is rejected. Binary hashes,
build options and complete per-rank timings accompany `kernel-summary.json`.
Both native comparisons preserved worker identities, weights and KV storage.

## Rejected attention experiment and recovery

The head-batched candidate preserves the original per-head top-k calls and
block order, then combines the gather/matmul/softmax batches. It passed the CPU
causality, padding, forced-block and mixed-request tests. Its first small device
case (one query, 125-token context) triggered a driver allocation failure while
synchronizing the candidate warmup. Private scratch caches were used; this
candidate never served user requests. The exact failing operator was not
isolated, so no device correctness or performance claim is made.

The driver left all workers blocked. The experiment was terminated and the
service restarted with v7 and the original attention implementation. This
restart lost resident request caches and required loading the shards again;
it was recovery from the failed device experiment, not a normal hot swap.
The recovered service returned a correct greeting and v9 passed a fresh full
projection/byte-reuse gate on the new workers. Raw driver diagnostics are
preserved as compressed logs. Attention remains unpromoted. Its experimental
benchmark now refuses to allocate when device free memory is below 2 GiB;
that admission check is not a general proof that an experiment cannot exhaust
memory. It prevents rerunning this experiment on the current tight profile.

The direct FP16-scale prototype (v11) was also rejected. Its small native-load
check raised an AI-core exception on all workers; the driver reported an
incorrect MTE burst count. At the 256-K validation shape a scale row shrinks
from 32 to 16 bytes, below the DMA transfer alignment; this is the suspected
cause, not a hardware-validated fix. No user request used v11. The failed
prototype is retained as a patch/manifest rather than enabled in the builder.
Recovering these poisoned contexts required a second restart. The final service
loads v9 directly through its post-start activation script.

Future unqualified native experiments should first run a small standalone probe
in a separate process before loading into resident workers. The INT8 probe below
uses that approach and leaves serving dispatch untouched.

## Serving qualification

`run-qualification.py` records the fresh baseline, native activation, candidate
decode measurements and optional broader checks. The broader checks were not
run in this iteration. The
counting probe uses 64 output tokens, three repetitions and greedy decoding.
It measures tokens per second after first output to exclude queue/prefill time.
Requests share the live service; this is a bounded measurement, not an isolated
throughput sweep or a large-context decode claim. The median increased from 1.746 to 1.830 tokens/s, a 4.8% gain. All three
64-token messages matched the baseline exactly. V9 is the active service kernel.
At the user's explicit request, the pending full coherence suite and tool-call
rechecks were skipped; only short output sanity checks were retained. The
existing parser, sparse attention and 160K configuration remain unchanged.

`activate_equivalent` requires successful bitwise projection and byte-reuse
receipts from the same resident generation, plus matching loaded binary hashes.
It drains active requests while new requests queue, changes only native dispatch,
and preserves caches. Failed activation restores the previous generation when
workers remain healthy. A native failure leaves the server paused for inspection.

## Reproduction

```bash
python -m tools.minimax_m3.build_resident_native /vllm-workspace/minimax-native-v9 \
  --namespace minimax_w4_v9 --wide-l1 --rint-unpack
python -m tools.minimax_m3.resident_benchmark \
  --manifest /vllm-workspace/minimax-native-v9/manifest.json --iterations 10 \
  --output /workspace/m31-v9-fresh-benchmark.json
python -m tools.minimax_m3.activate_equivalent \
  --manifest /vllm-workspace/minimax-native-v9/manifest.json \
  --evidence /workspace/m31-v9-fresh-benchmark.json \
  --output /workspace/m31-v9-activation.json
```

Build into a new directory/namespace each time. Do not reuse a sealed binary
name. CPU tests, changed-file hooks and repository-wide baseline checks are
recorded separately. The CPU suite ran with
`python -m pytest -q --confcutdir=tests/ut/minimax_m3 tests/ut/minimax_m3`:
185 passed and 24 device-dependent cases skipped. The parent UT conftest cannot
load in this local environment because its installed vLLM lacks
`vllm.third_party.flash_linear_attention`; the focused suite has its own fixtures.
The repository-wide `format.sh ci` baseline has existing
failures; its summary comes from an isolated checkout of `b6943a6db`.

## INT8 compute direction

`int8_projection_probe` ran in a separate process on chip 5, capped at 2% of
that chip's memory, with a 1 GiB free-memory admission check. It used one real
M3 expert with gate/up shape 3072×6144 and down shape 6144×3072, at one and
64 input rows. It compared pre-expanded FP16 with per-output-channel INT8
requantization, including prepared and dynamic-activation variants.

Including activation quantization, INT8 was 1.17–1.19x as fast as pre-expanded
FP16 in these four cases. Normalized RMS error against the CPU W4-dequantized
reference was approximately 1.07–1.13%. This is not a model-quality result or
an end-to-end speedup. Serving was available during the separate-process probe,
so these are shared-device timings, not peak-throughput specifications.

The prototype uses the public
[npu_quant_matmul API](https://ascend.github.io/docs/sources/pytorch/api_doc.html#torch_npu.npu_quant_matmul)
with INT8 operands and applies activation row scales afterward. Expanded INT8
weights cannot all reside alongside this model; a serving implementation must
retain packed INT4 storage and fuse its block-scale conversion into INT8 tiles.
The current measurements exclude that conversion, routing and collectives.
M3's 32×32 block scales require explicit handling; reinterpreting packed W4 or
using one whole-K scale directly would change the operation. INT8 is not enabled
in the service.
