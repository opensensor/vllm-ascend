# Qwen 310P whole-K INT8 math experiments

The direct INT8 routed MoE prototype is substantially faster than the deployed
compact INT4 projection in isolated real-weight tests. It is **not admitted to
serving**: it changes quantization and requires a larger resident weight bank.
The live TP6 1M server remains on its prior qualified configuration. No serving
worker, checkpoint, prefix cache, decode graph, or thermal threshold was changed.

## Measured component result

These are MoE rank partials with fixed router outputs and synthetic FP16 inputs.
Timing includes device dispatch, quantization, routed copies, both projections,
FP16 SwiGLU, and route finalization. It excludes the router matrix product,
shared expert, TP collective, attention, GDN, and the rest of the model. Five
alternating timing repeats contain two complete partial executions each.

| Layer / rank | Input tokens | Compact W4 ms | Direct INT8 ms | Local INT8 ms | Local speedup | Output relative L2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 / 0 | 128 | 4.56 | 2.91 | 2.75 | 1.66x | 2.57% |
| 0 / 0 | 640 | 12.82 | 6.70 | 6.06 | 2.12x | 2.55% |
| 0 / 0 | 2560 | 39.10 | 18.78 | 16.22 | 2.41x | 2.56% |
| 47 / 5 | 128 | 5.08 | 3.74 | 3.64 | 1.39x | 2.76% |
| 47 / 5 | 640 | 10.89 | 6.62 | 5.98 | 1.82x | 2.88% |
| 47 / 5 | 2560 | 32.12 | 17.78 | 14.96 | 2.15x | 2.93% |

Expanded INT8 weight banks occupy 424,028,160 bytes on layer 0/rank 0 and
419,097,600 bytes on layer 47/rank 5, in addition to the original diagnostic W4
banks. This does not demonstrate that full-model INT8 fits the live 1M cache.
The rank-5 128-token trial has visible timing variability; the larger trials
are more stable. Receipts contain every sample rather than only the median.

## Why the recipe helps

The stock projection experiment folds the original G128 weight scales into a
new per-output-channel INT8 weight and dynamically quantizes each activation
row. The native implementation keeps exact integer partials in L0C over the
whole K axis and applies weight and activation scales once. It avoids the
per-G128 readback, conversion, and scaling loop and the second activation limb.
Direct FP16-to-INT8 quantization runs on core in bounded UB storage. Input tokens
are quantized once before routed copies; SwiGLU output is quantized directly.
The local variant quantizes only the device-counted local prefix of the down
projection, skips peer padding, and produces the same output bytes as the full
direct variant. That reduces 18.78 ms to 16.22 ms on layer 0/rank 0 and 17.78 ms
to 14.96 ms on layer 47/rank 5 at 2,560 tokens.

A reconstructed-packed-activation variant takes about 39 ms at 2,560 tokens,
showing why the large gain requires removing the surrounding conversion chain.
The INT8 peak rating alone does not predict this serving performance.

## Correctness and rejected alternatives

- Real checkpoint layer-0 banks on all six chips passed independent CPU integer
  recipe and FP16 output checks, including changing-input NPU graph replay,
  odd group starts, 127/128/129-row boundaries, one-row tails, and empty experts.
  This qualifies the implemented arithmetic, not original-model equivalence.
- A missing MTE1-to-MTE3 ownership dependency let the next activation copy
  overwrite L1 before operand loading finished. The corrected path protects L1
  explicitly. Earlier failing whole-K prototype timings are invalid evidence.
- The lossless packed INT8 G128 hybrid, cached metadata, and paired dense-limb
  variants passed six-chip exact W4 checks. The hybrid and metadata variants
  were slower; pairing produced only a small isolated gain and was not deployed.
- On-chip W4-to-channel-INT8 expansion preserves the original resident W4 bank
  and adds only channel scale metadata. It passed rank-0 arithmetic checks, but
  took **54.12 ms versus 39.16 ms** at 2,560 tokens. Expansion cost outweighs the
  saved readbacks. This variant is rejected for serving.
- The wider M64/N64 tile was also slower than compact W4 and was not promoted.

The new recipe differs by approximately 2.6–2.9% relative L2 in these full routed
partials. Coding, image, logits/perplexity, long-context quality, decode latency,
thermal behavior, and whole-server throughput have not been qualified for it.
There is no measured 50% whole-server tokens/s improvement from this phase.

## Next admission work

1. Preserve the measured local-row quantizer and full-K integer accumulation
   in any storage design; they avoid host count reads and unnecessary work.
2. Resolve resident storage before proposing a server cutover. Full shadow
   INT8 banks do not fit current spare memory. Replacement banks also need an
   honest KV/workspace/graph budget and a decode benchmark: some stock INT8
   single-expert decode cases are slower than W4.
3. Qualify the new quantization across the full model with coding and image
   requests and explicit long-context checks; independent integer parity is
   insufficient. Changing quantization also requires separating cached states
   from the prior model recipe.
4. Measure matched cold prefill, inter-token latency under overlapping prefill,
   and complete request throughput before admitting any serving configuration.

## Reproduction

Use the pinned serving Python and CANN environment on the NPU host. Builders
create an append-only namespace and freeze C++ source/binary hashes without
opening devices. The historical source stages and build manifests are retained
under `/srv/ai/src/qwen-hybrid-prefill-*20261011-*`.

```bash
python -m tools.qwen4exp.build_hybrid_prefill   --build-dir NEW_BUILD --cann-root CANN_ROOT --source-root SOURCE_ROOT   --version NEW_NAMESPACE_NUMBER
python -m tools.qwen4exp.validate_hybrid_prefill_310   --model CHECKPOINT --build NEW_BUILD --output NEW_RECEIPT --rank 0
python -m tools.qwen4exp.benchmark_channel_grouped_310   --model CHECKPOINT --build NEW_BUILD --output NEW_TIMING --rank 0   --layer 0 --direct-activation --tokens 128 640 2560   --iterations 2 --repeats 5
```

Add `--packed-weight` only for the rejected storage experiment; add
`--local-activation` for the local-row quantizer. Both validation and timing
check live request activity before and after, reject overlapping traffic, and
never install an implementation into serving workers.

The repository-wide `bash format.sh ci` was run and failed on pre-existing
files outside this change. Its automatic unrelated edits were restored only in
the owned worktree. Scoped checks passed. The final focused CPU suite passed 74 tests. Results and
source/binary hash manifests accompany this report. The latest build-v22
manifest matches the delivered kernel source; historical measurements retain
their own build versions.
