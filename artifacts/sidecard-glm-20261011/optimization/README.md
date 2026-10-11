# GLM sidecard mHC optimization

The retained service is `sidecard-glm-service-20261011-one-card-v12` at
`http://192.168.53.187:8001/v1`, model `glm-sidecard`. It uses only Ascend
physical devices 2/3 and the RTX. The other Ascend card's Qwen processes were
preserved. [service-process.json](service-process.json) records the launch.
CUDA daemon 1300754 retains the scalar packed expert banks prepared under v8;
the CUDA projection experiments below are disabled in serving.

## Ascend finding and change

[profile-summary.json](profile-summary.json) records two 20-second, 25 Hz
nonblocking Python profiles of the preceding v10 worker. In decode, 128 of
436 main-thread samples landed in the two lines of the unfused 20-round mHC
Sinkhorn loop. In prefill, most main-thread samples landed in the NPU-to-host
input copy or the CUDA reply wait. Python samples identify host call sites
and pending device work; they do not measure individual NPU kernel durations.
The raw traces remain on the host at the paths pinned in the summary.

The earlier GLM research already implemented the fused 310P Sinkhorn kernel.
The sidecard service had not selected its `ascend_glm_fused_sinkhorn` override.
The launcher now exposes `--fused-sinkhorn`, preserving the other model
overrides. All 20 normalization rounds remain present. No expert code,
scale, ownership, or attention implementation changed.

[mhc-kernel-gates-qualified.log](mhc-kernel-gates-qualified.log) records
24 passing Sinkhorn/native-mHC-post hardware tests on the assigned Ascend
card. They imported `sidecard_worker` first to load the actual serving
bindings. The stock extension alone omits the auxiliary mHC post binding.

The retained profile has 512-token prefill, one live sequence, prefix caching,
3.5 GiB KV per NPU, and 260,352 admitted context tokens. Kilo retains 228,352
input / 32,000 output limits. Full context prefill and model quality remain
unqualified. [service-validation.json](service-validation.json) records all
seven operational API cases and 168 serialized/overlap comparisons with zero
error. Greedy repeat variation still prevents a full validation pass.

## Completed serving measurements

| Check | Earlier v10 | Retained v12 |
| --- | ---: | ---: |
| Decode, 2K prompt / 64 output tokens | 2.55 tok/s | 3.46 tok/s |
| Cold 8K prefill, zero cached tokens | 57.20 tok/s | 57.73 tok/s |
| Cold 8K first text | 143.23 s | 141.90 s |
| Cached 8K first text, 8,064 cached tokens | 2.73 s | 2.64 s |
| Admitted context | 260,352 | 260,352 |

[decode-64.json](decode-64.json) and [prefill-8192.json](prefill-8192.json)
record completed requests. The 64-token rate excludes prefill and the first
output token. The earlier decode used the identical prompt/output budget
with nonblocking profiling enabled; this is not a perfectly isolated timing
comparison. The approximately 35% decode gain is consistent with the previous
2.6–2.8 tok/s decode observations. Cold prefill is essentially unchanged.

[npu-inventory.log](npu-inventory.log) records healthy devices and 443 MiB free
on the smaller GLM chip after the mixed 2K/64-output and 8K workload. The
current cache budget was retained. [cuda-inventory.log](cuda-inventory.log)
records the retained daemon, with no experimental GPU worker left running.

## Retired 640-token candidate

The 640-token budget activated the existing native mHC post dispatcher and
passed the short operational checks. It decoded a completed 64-token output
at 3.44 tokens/s, but later exhausted memory on physical NPU 3 during cold
8K prefill after the 2K/64-token request. The asynchronous error reported
`SelfAttentionOperation`; the Python stack alone does not establish which
allocation caused it. The candidate was retired through the receipt-checked
launcher, keeping the CUDA banks and the other Ascend workload.

Files prefixed `candidate-640-` retain its successful short checks and failed
8K result. They do not qualify that candidate for long prompts. Its smaller
255,744-token admission limit was also retired.

## CUDA experiments

[cuda-projection-gate.json](cuda-projection-gate.json) records 27 real-weight
prefill cases plus 42 short-batch cases across W4/W3/W2 layers, activation
bits 4/8, and the 31/32/33-route tile boundary. All outputs matched bitwise.
Grouped INT8 Tensor Core projections retain each integer K32 dot product and
ordered FP32 scale/add. The split-K decode projection writes independent
K32 products, then adds them in the original order using bounded scratch.

Paired medians used the same resident banks, packets, transfers, and compiler
flags. The Tensor Core path improved some largest batches by only a few
percent and regressed others. Decode projection already took about 0.1 ms
for four routes, so its isolated speedup could not explain overall decode
latency. These paths remain explicit experiments, disabled by default.
Reproduction now requires `batch_projection_gate --experimental-projection`;
the captured gate predates that explicit switch and selected library symbols
implicitly. [cuda-experiment-receipt.json](cuda-experiment-receipt.json)
records the compiler flags and library identity.

CPU verification: 381 focused sidecard/pipeline tests passed. Changed-file
lint hooks and the CUDA formatter passed. Repository-wide formatting retains
unrelated baseline failures; those are recorded separately from scoped checks.
