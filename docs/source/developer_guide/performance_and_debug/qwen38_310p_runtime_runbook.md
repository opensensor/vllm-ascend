# Qwen3.8 Flash-Next W4 on 310P: Runtime Runbook

This runbook records the operating rules for the experimental Qwen3.8
Flash-Next W4A8 runtime on four Ascend 310P devices. Read it before changing
the launcher, checkpoint loader, cache accounting, custom operators, or ACL
graph settings. The paths and measured values below describe the qualified
development host; they are evidence for this runtime, not general vLLM Ascend
defaults.

## Sources of truth

- Develop changes on the repository's current main branch, then deploy them as
  one complete runtime snapshot. Confirm the runtime path printed by the server
  before interpreting a result.
- Keep `examples/start_qwen38_flash_next_w4_310p.sh` and
  `~/start_qwen38_flashnext_mtp_graph.sh` byte-for-byte equivalent after a
  launcher change.
- Do not stage the whole working tree. Several agents can share it, so add only
  the files owned by the current task and use signed commits.
- Stop NPU processes immediately when device use is deferred. Host-only review,
  builds, and documentation can continue without importing NPU packages.

## Read the shard progress correctly

The target checkpoint has 1,610 safetensors files. With TP4/EP4 and
`--enable-ep-weight-filter`, every rank reads the shared dense tensors in the
first approximately 48 files. Those files dominate cold disk time. After that
boundary, each rank rejects non-local expert tensors before disk I/O, so the
progress bar can jump from about 3% to 50% and then to 100%.

The MTP draft model reuses target weights and crosses its first 48 entries in
roughly 5-10 seconds. Do not compare that draft progress with the target's cold
first 48 files. A healthy target load may appear stalled for several minutes
and then jump at shard 49.

Keep the default safetensors iterator. The generic multi-threaded iterator does
not preserve the local-expert filter and can make every rank read the whole
expert bank. Auto-prefetch is intentionally disabled on this host: the
169.19-GiB checkpoint is larger than 90% of the approximately 136-GiB available
RAM, and EXT4 is not treated as a network filesystem. Forcing a full prefetch
would add memory pressure rather than fix the dense-shard bottleneck.

Native INT4 conversion temporarily raises PyTorch's host packing threads from
the inference setting of one to at most six CPUs in that worker's affinity
set, then restores the original setting. This accelerates the dense first 48
files without changing shard iteration or numerical packing. The launcher's
`--check-runtime` gate rejects a snapshot that has lost this packer.

Measure startup by the slowest rank's `Model runner load_model total time`.
Rank 0 finishing early does not make the service ready; the remaining ranks and
collective barriers still determine wall-clock startup.

## Keep custom operators coherent

The host API and kernels for an operator must come from the same custom OPP
package. Path order alone cannot safely combine two packages that both define
the same operator. We previously loaded an FP16 recurrent host API with an FP32
recurrent kernel, which failed during graph capture even though both packages
looked valid in isolation.

The qualified launcher places a coherent package containing both the retained
W4 operators and the FP32 recurrent-state operator first in
`ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`. The embedded package follows it
because its QSA host API matches the current `logical_kv_heads` ABI. The older
retained package is last and may provide only operators absent from both newer
packages. Putting the retained package before the embedded package selects its
older QSA signature; the shifted arguments then make graph capture report a
null output tensor.

`vllm_ascend` bootstraps its embedded `_cann_ops_custom` vendor by prepending it
when that path is absent. The launcher must therefore include the embedded
vendor explicitly after the coherent vendor and before the retained fallback.
Merely exporting the two external paths lets bootstrap move the embedded host
library to the front and silently restore its FP16-only recurrent schema.

The W4 host tilers must also cover the routed rows produced by one prefill
chunk. The target checkpoint routes each token to ten experts, so the qualified
2,048-token chunk requires capacity for 20,480 routed rows. A stale custom OPP
capped both the matmul and activation-pack tilers at 5,120 rows: decode graph
capture and short prompts passed, but the first chunk of a cold 23K prompt
failed with `Failed to execute tiling function`. The r2 coherent package raises both
bounds to 20,480. Rebuild the host tilers whenever
`MAX_NUM_BATCHED_TOKENS * top_k` exceeds the packaged bound, and validate with
a prompt longer than one chunk. A decode-only startup test cannot establish
prefill support.

Test chunked GDN with the model's production head geometry before accepting a
rebuilt operator package. Qwen uses 16 gate/key heads and 48 value heads with
128-dimensional heads. The small one- and two-head ACLNN cases did not expose
a bad gate-vector copy: they passed while the production case produced about
0.95 recurrent-state cosine and incoherent text. Compare both the chunk output
and final state with the PyTorch reference for
`(B, Hg, Hv, T, K, V) = (1, 16, 48, 128, 128, 128)`. A tiling-offset warning
at that shape is a kernel failure even when decode startup succeeds.

Before starting a server, verify:

1. `nm -D` shows every required W4 and recurrent host symbol in the first host
   library.
2. The recurrent operator configuration contains both FP16 and FP32 state
   variants.
3. Library resolution after sourcing the hardware environment points to that
   same package.
4. `bash -n` passes for the launcher and its `--show` output contains the
   intended model, cache, graph, and TP/EP settings.

Do not diagnose an asynchronous failure from the top Python frame alone. An
error reported at an all-reduce can originate in the preceding attention or
recurrent operator. `ASCEND_LAUNCH_BLOCKING=1` is useful for one diagnostic run,
but unset it for performance measurements.

## Cache accounting and memory profiling

Memory profiling remains necessary when the model, operators, dtype, graph
mode, cache layout, or workspace changes. The qualified fixed-cache path may
skip the maximum-token profile forward only when the user supplied an explicit
`--kv-cache-memory` value and that exact configuration has already passed a
real profile run.

The fixed value is expressed in logical planner bytes. It is not the raw NPU
allocation. For the qualified compact-state layout:

```text
8,400 planner blocks x 10,556,416 logical bytes/block = 88,673,894,400 bytes
```

Those blocks consume about 15.4 GiB of physical attention pages per rank plus
an approximately 1.87-GB, 64-slot recurrent-state pool. The graph-visible
recurrent-state tensor shape is accuracy-qualified and must not be enlarged.
After graph capture and HCCL initialization complete, the runner fills otherwise
unused NPU memory with a separate LRU checkpoint archive while retaining a
4-GiB prefill/runtime reserve. Deferring this allocation is required because
HCCL may create its communicator during graph warmup. The archive does not
change any kernel-visible slot or block-table shape. Startup logs report its
capacity. The first NPU-to-CPU spill after both device tiers fill and the first
later CPU-to-NPU restore each emit an explicit warning. Cache transforms such
as offload and KV parallelism must still run even when the maximum-token
forward is skipped.

The retained launcher defaults to the complete
`qwen38-head-unified-runtime-20261001` snapshot. Do not assemble a serving
runtime by copying individual Qwen files from different source trees. The
launcher's `--check-runtime` gate catches known API mismatches, while a
thinking-enabled generation gate remains required for semantic coherency.

The 4 x 256K claim means the engine reports at least 1,048,576 cache tokens and
maximum concurrency of at least 4.00 for a 262,144-token request. Setting
`--max-model-len 262144` alone does not establish capacity.

The Mamba page-size messages are arithmetic and normally complete immediately.
A long delay before them is usually a lagging rank at model load. A long delay
after them and before the cache result was the maximum-token memory profile
forward. The profile metadata previously omitted GDN attention, so that
forward did not warm the cold-prefill route.

## Graph-capture constraints

GDN supports decode-only full graph capture in this runtime. Use
`FULL_DECODE_ONLY`. MTP verifies `K + 1` tokens for every live request, so each
capture size must be a multiple of `K + 1`. TP graph capture on 310P has a
two-size event-id budget: a third graph exhausted HCCL capture events in the
qualified experiments. The general MTP2 service profile therefore uses
`[3, 6]`, keeping the interactive C1 and C2 shapes exact. C3 and C4 use eager
decode. Launch with `--c3-c4-graphs` for the qualified `[9, 12]` profile during
four-request GPQA runs. It captures C3 and C4; C1 and C2 pad to the 9-token
graph, so return to the default profile for interactive service. Mixed
prefill/decode full capture generated requests whose token count
exceeded the decode graph size and failed the GDN assertion.

The 2026-10-02 [GPQA Diamond end-to-end run](../../../../artifacts/qwen38-w4-offline/GPQA_DIAMOND_20261002.md)
used this C3/C4 profile for its final 106 cases. All 106 completed, with no
eager decode fallback or zero-acceptance interval in the server log. This
qualifies the profile for that four-request workload; it does not replace the
default interactive C1/C2 profile.

When a uniform decode batch cannot use any configured graph key, the model
runner emits a one-time warning for that batch shape with its token count,
request count, query length, and capture sizes. Treat that warning as a
performance failure: the affected shape pays host dispatch cost on every step.
The launcher also enables `VLLM_ASCEND_LOG_REQUEST_TIMINGS`; every completed
request reports prompt tokens as computed plus cached, making lost hot-prefix
reuse visible without inferring it from aggregate cache metrics.

Decode graph capture must also exercise the FP32 recurrent state successfully.
An error saying that `params.state` supports only FP16 indicates an incoherent
host API/kernel package, not a reason to cast the model state down to FP16.

## Validation sequence

Run the following gates in order and retain the logs:

1. Host checks: launcher syntax and `--show`, Python compilation, focused unit
   tests, custom OPP symbols/configuration, and library resolution.
2. Stop the existing service and verify that no process owns an NPU.
3. Start the full TP4/EP4 server. Record target and draft load times for every
   rank; confirm the expert filter and the post-shard-48 jump.
4. Confirm the fixed-cache skip message, at least 1,048,576 cache tokens,
   concurrency at least 4.00, successful decode graph capture, and `/v1/models`
   readiness.
5. Run generation and tool-call smoke tests.
6. Measure a genuinely cold unique long prefix, then the same warm prefix.
   Prefix caching speeds reuse; it does not turn the first request into a warm
   request.
7. Measure serial 512-token decode and a four-request concurrent throughput
   run. Report TTFT independently from decode and end-to-end throughput.

Useful clients are:

- `tools/qwen38_decode_study/smoke.py`
- `tools/qwen38_decode_study/long_prefix.py`
- `tools/qwen38_decode_study/benchmark.py`
- `tools/qwen4exp/benchmark_capacity.py`

For streaming results, compute serial decode as
`(completion_tokens - 1) / (last_token_time - first_token_time)`. Compute total
throughput as the sum of completion tokens divided by the concurrent wall time.
Save speculative draft and accepted-token counters with the result. Never
claim a cold-prefill improvement from a request that reused a live prefix cache.

On the qualified r2 runtime, an exact uncached 23,000-token prompt took
68.24 seconds to first token with 2,048-token chunks and 68.51 seconds with
2,560-token chunks. Both runs preserved 4.08 concurrent 262,144-token requests,
but the larger prompt microbatch produced no cold-prefill gain. That test
did not change the model's grouped expert chunk. A later isolated native-W4
candidate moved the kernel's large-tile switch to 128 rows per expert and
reduced the grouped chunk to 1,536 tokens while leaving the scheduler at
2,048. Three matched 23,410-token cold prompts had median TTFT 68.950 s,
versus 72.864 s in saved baseline records, with identical generated-text
hashes and zero cached tokens. See the
[batching experiment](../../../../artifacts/qwen38-prefill-batching-20261004/README.md).
The shared source now carries those two changes; rebuild and gate its combined
pending operator changes before treating a new package as qualified. The
`--warm-prefixes` capacity probe also reported `cached_tokens: 0` in this hybrid
Mamba configuration. Treat a warmup as cached only when the response usage or
server prefix-cache metrics prove a hit.

A native-W4 candidate raises the source route caps to 25,600 and adds an
explicit `grouped_prefill_chunk_tokens` model override up to 2,560. Its
default remains 1,536. The candidate keeps the 32-row projection schedule
across the previously measured 1,638/1,639-token cliff. An isolated 310P
one-layer gate found bitwise-identical output and an 8.0% improvement for a
23,410-token partial with 2,560-token chunks versus 1,536-token chunks. A
threshold-128 control confirmed that the new projection schedule is material
at larger chunks. A coherent five-operator package and isolated launcher
passed a TP4/EP4 service gate: three matched 23,410-token cold prompts fell
from 64.292 to 62.032 seconds mean TTFT against the earlier 1,536-token
CANN-finalizer service. All reported zero cached tokens. The service reported
4.08 concurrent 262,144-token requests and captured both decode graphs; a
four-request workload and sustained thermal gate remain. The older
20,480-route OPP cannot run a 2,560-token top-10 chunk. See the
[experiment record](../../../../artifacts/qwen38-prefill-batch256-20261005/README.md)
before changing the serving default.

Native INT4 grouped prefill now defaults to `cann_builtin_fp16`, which uses
`torch_npu.npu_swiglu` before the existing down-projection pack. Other W4
backends keep their torch activation default, and an explicit
`grouped_activation=torch` selects the native INT4 reference path. On three
matched 23,410-token cold prompts, mean TTFT improved from 68.920 to 67.225
seconds and effective prompt rate from 339.7 to 348.2 tok/s. One of the three
32-token outputs changed an opening phrase. The user accepted that difference
for this experimental backend; broader quality and thermal comparisons remain
open. See the [SwiGLU experiment](../../../../artifacts/qwen38-prefill-swiglu-pack-20261004/README.md).

The opt-in CANN grouped finalizer was then tested with built-in SwiGLU still
active and the 1,536-token expert chunk unchanged. In one real-weight layer,
the local grouped MoE call fell from 34.249 to 29.671 ms. Three matching
23,410-token requests had mean cold TTFT 67.225 to 64.292 seconds, or 348.2
to 364.1 effective prompt tok/s, with zero cached tokens. One opening phrase
changed; the other two short outputs matched. The finalizer remains opt-in
because it changes FP16 rounding and broad quality has not been checked.
The [combined gate](../../../../artifacts/qwen38-builtin-finalize-20261005/README.md)
records the configuration and response checks. That candidate previously ran
on port 8001 as `qwen38-w4-builtin-finalize-candidate`.

## Known bad turns

- Replacing the default loader with a generic parallel iterator lost the EP
  skip and increased I/O and memory use.
- Treating shard progress as linear made the expected dense-shard phase look
  stalled and the expert-filter jump look accidental.
- Mixing overlapping custom OPP packages caused dtype and stream failures at
  graph capture.
- Omitting the embedded vendor from the explicit path let plugin bootstrap
  prepend it and shadow the coherent recurrent host API.
- Putting the retained package before the embedded vendor selected the older
  QSA host ABI, shifting `logical_kv_heads` into the output-tensor position.
- Mixed prefill/decode full graphs violated GDN's decode-only constraint.
- Treating logical `--kv-cache-memory` bytes as physical allocation produced
  false capacity conclusions.
- Looking only at rank 0 hid the slowest-rank startup bottleneck.
- Assuming a larger prompt microbatch would improve prefill added no measured
  gain at 2,560 tokens; benchmark the physical batch instead of inferring it.
- Calling a request warm without checking `cached_tokens` mislabeled a second
  full prefill as a prefix-cache measurement.
- Validating chunked GDN only at one or two heads missed a production-shape
  gate-copy regression and allowed a numerically bad device object to ship.

## Six-chip compact GDN qualification

For the three-card TP6 candidate, read the
[six-chip hardware report](../../../../artifacts/qwen38-six-chip-hardware-20261009/README.md).
The image encoder uses data mode, MTP is disabled, and the baseline routed
projection remains faster for short batches than streaming v2.

Nine-head GDN shards require both corrected chunk schedulers and aligned
causal-mask writes. A short decode smoke test does not exercise these paths;
validate every prefill row and final state against reference math, then check
coherent output on a cold prompt longer than one chunk. Use the matching
stride-aware convolution binding and a coherent package containing the four
convolution, recurrent, state-prefill, and output-prefill operators. An
output-only package can shadow metadata for a convolution binary it does not
contain. Preserve the qualified TP4 packages and launcher for rollback.

Six slots do not imply six full 256K windows. Report planner token capacity,
measured concurrent request overlap, and full-window validation separately.
Logical copy accounting is not a measurement of physical bus traffic or heat.
