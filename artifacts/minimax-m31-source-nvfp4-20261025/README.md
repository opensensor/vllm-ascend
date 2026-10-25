# Source-preserving M3.1 experts and loop comparison

## Result and scope

**The source-preserving weight change has not fixed the tool loop.** A fresh
Kilo trial repeated the same directory read on three consecutive turns, despite
stating an intent to read the README. Each tool call succeeded and returned the
directory listing. The run was stopped after the repetition appeared; this does
not establish that the session could never recover. See
[source-fresh-kilo.json](source-fresh-kilo.json).

A versioned CANN kernel on 310P now consumes the original nonlinear NVFP4
expert codes and per-row K16 scale bytes without uniform INT4 requantization.
CPU packing reproduced the decoded source FP16 weights exactly. Real-weight
device measurements reduce the sampled complete-expert errors from
14.77–20.26% to 0.00776–0.01380% normalized RMS error. This removes the observed
conversion loss for this sample; it does **not** establish a reasoning-loop fix.

The isolated kernel comparison ran in a separate device-5 process. All 342
source-preserving banks have since been exported and transferred, and the full
TP2/PP3 model has started on all six chips. Full-model startup is not a quality
qualification; the controlled loop trial below is the decisive gate.

## Format and implementation

The design follows the source-preserving approach inspected in
[Unsloth's converter](https://github.com/unslothai/llama.cpp/blob/822be707b23345b291f3042defa7508aa813cc4b/conversion/minimax.py).
We use the original MiniMax checkpoint already on disk; no second GGUF download
is needed. Codes are losslessly reordered into 16x256 NZ tiles, scales into
matching K16/N16 tiles. One 8 KiB lookup per expert projection contains every
valid scale/code combination, including FP32 division by the original global
scale and final FP16 rounding. The device expands only the current tile and
uses the existing FP16 Cube/L1 schedule.

`tools/minimax_m3/nvfp4.py` supplies packing, the independent layout inverse,
and byte accounting. `nvfp4_native.py` validates geometry and dispatches compact
or grouped routes. `build_resident_native --source-nvfp4 --wide-l1` creates an
append-only binary/bridge/manifest using the existing resident-native mechanism.
The optional build flag leaves ordinary W4 compilation unchanged. A separate source-bank loader now connects this format to the full model
while retaining the existing audited FP16 dense export.

An initial graph attempt failed because `torch.zeros` invoked uncapturable
`ZerosLike`. The retained kernel writes peer rows to zero itself, allowing
`torch.empty` in the wrapper. Graph replay with changed inputs and routes then
matched eager execution exactly. Kernels and tiling metadata remain resident;
the projection performs no host copies or per-token shard loading.

The comparison uses FP16 activations and decoded FP16 weights with FP32
accumulation. It does **not** reproduce original FP4 activation quantization or
the checkpoint's trained input scales. Exact decoded weights are a controlled
reference, not a claim of bit-identical original-model inference.

## Real-weight comparison

The fixture uses the same layer-3 inputs and experts as the preceding
[loop investigation](../minimax-m31-loop-isolation-20261025/README.md): one token
from a synthetic 1025-token cache probe, not the looping conversation. Each
complete expert independently evaluates gate, up, clipped SwiGLU and down.

| Expert | Prior stored W4 vs source, CPU | New source kernel vs source, NPU |
| --- | ---: | ---: |
| 52 | 14.7745% | 0.00776% |
| 47 | 16.5680% | 0.01380% |
| 15 | 20.2593% | 0.01042% |
| 46 | 18.9501% | 0.00784% |

The aggregate complete-expert NRMSE is 0.01001%. A matched device measurement
of the current W4/INT8 v15 path gives 17.13% aggregate NRMSE against the same
source reference. This comparison includes v15's additional compute error.
Neither result is a model accuracy percentage or a representative quality score.

The new kernel takes about 12 ms for each four-expert projection; v15 takes
1.28–1.33 ms. Capturing the source projection in a graph changes little because
this prototype is dominated by device work. These are five-iteration isolated
projection measurements, not serving tokens/s. The candidate is not a speed
improvement and is deliberately not promoted.

See [device-v3.json](device-v3.json) for per-projection/per-expert errors,
compact/grouped equivalence, graph checks and timings. Synthetic tests include
K=1024/3072/6144, concentrated and mixed groups, the 128-row L1/fallback boundary,
reordered/repeated routes, peer-first and all-peer routes. The fixture receipt
is [fixture.json](fixture.json); large weight payloads remain outside Git.

## Memory constraint and next implementation target

Source scale bytes plus lookup tables increase total expert storage by
23.4624 GiB relative to the current W4 export. The full payload would be
242.6214 GiB. A conservative TP2/PP3 budget with dense FP16 weights, replicated
vision, 160K context and 3 GiB runtime reserve peaks at **49.7132 GiB per chip**;
it does not fit 45.6 GiB. This is an offline estimate, not a measured allocation;
it does not credit the current selected dense INT8 savings.
See [memory-budget.json](memory-budget.json).

That budget replaces each expert record's scale shape with `[N,K/16]` U8 and
adds a `[4096]` F16 lookup, preserving all dense records in the existing manifest,
then calls `serving_budget.make_budget` with context 163840, utilization 1.0,
device capacity 45.6 GiB and runtime reserve 3 GiB.

After the full-model loop gate, implementation work must address both costs: faster tile decode/INT8
compute and lossless scale storage or other measured memory savings. Doubling
the E2M1 values permits exact integer codes, but applying scales that vary every
16 input values is not equivalent to ordinary single-scale INT8 GEMM. A fresh
multi-turn source-preserving model comparison has reproduced repetitive tool use. This operator
result alone cannot justify calling the reasoning loop solved.

## Reproduction and checks

On the source host, using the existing trusted activation capture:

```bash
python -m tools.minimax_m3.nvfp4_probe prepare \
  --source /path/to/MiniMax-M3.1-Flash \
  --inputs /path/to/m31-real-projection-inputs.pt \
  --fixture /tmp/m31-nvfp4-fixture.pt --output /tmp/fixture.json
```

On the 310P host, from the installed checkout with the existing CANN environment:

```bash
python -m tools.minimax_m3.build_resident_native /workspace/m31-nvfp4-new \
  --namespace minimax_w4_nvfp4_new --wide-l1 --source-nvfp4
python -m tools.minimax_m3.nvfp4_probe measure \
  --fixture /workspace/m31-nvfp4-fixture.pt \
  --manifest /workspace/m31-nvfp4-new/manifest.json \
  --output /workspace/m31-nvfp4-result.json --device 5
```

Check free memory before launching alongside a service. The isolated operator is validated
on device 5, and text-only TP2/PP3 startup is hardware-validated. The source
trial uses 32K context; images and 160K context are not qualified for it. The
existing W4 profile remains separately documented.
The build receipt is [native-manifest.json](native-manifest.json).

CPU suite: `python -m pytest -q --confcutdir=tests/ut/minimax_m3
tests/ut/minimax_m3` passed 293 tests with 24 skips. The repository-wide UT
conftest cannot import the locally mismatched vLLM FLA module; the dedicated
CPU suite avoids that unrelated dependency. Scoped formatting and repository
checks are recorded separately; whole-tree failures are preexisting.

## Full-model loop gate

The source-preserving operator result is insufficient to claim a loop fix.
The full trial replaces all 57 layers of routed experts, retains FP16 dense
weights, disables expert/dense INT8 compute and uses the same eager attention,
activation, routing, tokenizer and tool handling as the matched W4 baseline.
The temporary context is 32768 tokens, enough for the 29577-token recorded
history plus the bounded completion. It does not establish 160K feasibility.

The matched W4 baseline used temperature 0, seed 71 and a 1024-token completion
cap. Both the 5549-token late history and the complete 29577-token history
continued the repeated promise to summarize followed by more repository tools.
The complete history ended after 221 completion tokens with `tool_calls`,
not a final project description. Thus disabling INT8 compute does not eliminate
the symptom. Private prompts remain outside Git.

The source run must be compared with these same inputs. A separate fresh Kilo
session uses the original pre-mitigation agent prompt and normal tool results,
with a read-only filesystem view. Its external request budget stops the
experiment without injecting an instruction to finish. Natural task completion,
coherent final output, and absence of repeated equivalent discovery are the
behavioral criteria; a numerical tolerance or HTTP 200 alone is not a pass.

Export source expert banks without modifying the original checkpoint:

```bash
python -m tools.minimax_m3.export_nvfp4 /path/to/MiniMax-M3.1-Flash \
  /path/to/MiniMax-M3.1-Flash-NVFP4-experts --threads 8
```

The existing W4 loader accepts `source_nvfp4` (the bank directory) and
`source_nvfp4_kernel` (the sealed native manifest) in its extra configuration,
with `expert_layout=nz256`. Both must be supplied. The bank manifest must cover
the same source config/index and complete expert set as the dense export.
Each bank is checksum-verified before transfer to the device. Source activation
scales are retained in the export for provenance but are not applied to FP16
activations in this controlled reference.

For a subsequently built and validated source kernel, resident `load-native`
followed by `switch --source-native-name <versioned_namespace>` changes the
projection callable while preserving code, scale and lookup storage. Standard
resident pause/drain/cache-reset rules apply. W4 and source-native dispatch
cannot be selected together.

### Full-model startup

The complete source bundle contains 342 banks (216.5892 GiB). See
[export-receipt.json](export-receipt.json) for the source and manifest identity.
The successful text-only trial uses 32,768 context tokens, eager execution,
FP16 dense weights/activations, a 512-token prefill batch and a 900 MiB explicit
KV budget. The engine allocated 35,072 cache slots. Reported model allocations
were 40.6823 GiB on stage 0, 41.5671 GiB on stage 1 and 40.6523 GiB on stage 2.
These allocations leave little headroom; they are not a larger-context claim.

Two startup issues were fixed before inference:

- Source file checksums changed in cached reads during direct mapped-tensor
  loading, although direct disk reads retained the expected checksum. Evicting
  the affected file's clean cache pages restored the ordinary read checksum.
  The loader now clones one tensor into owned host storage and synchronizes its
  device transfer before releasing staging. The successful load no longer
  passes file-backed, potentially unaligned storage to device copies. The
  exact underlying driver interaction was not isolated.
- Lazy HCCL initialization failed after the larger source banks occupied device
  memory. Releasing PyTorch's allocator cache alone did not fix it. A small TP
  collective before expanding source scales reserves communication memory;
  the complete six-rank startup then succeeded. Warmup also releases unused
  allocator blocks before profiling.

The independent [bank audit](remote-bank-audit.json) checked all transferred
files before the mapped-load issue was isolated. It is not a post-inference
checksum audit. No original checkpoint or W4 export was overwritten.

The fresh W4 control reached the external 12-turn budget after 24 tool calls
and 2,574 completion tokens without delivering a project description. It
repeated promises to summarize and fork/upstream discovery. Two bash tools
reported no exit status, which limits interpretation of this fresh run;
the recorded-history controls independently reproduced the same symptom.
See [w4-fresh-kilo.json](w4-fresh-kilo.json) and
[w4-matched-replays.json](w4-matched-replays.json). Private conversations and
complete tool outputs are intentionally excluded from these receipts.

### Source-weight behavioral result

The fresh source run used the original pre-mitigation agent prompt, temperature
1.0, top-p 0.95 and seed 71. Its first call listed the checkout. Turns 2, 3 and 4
then called `read` on the same directory, with equivalent offsets, while the
reasoning claimed it would read the README. The tool returned a valid directory
listing every time. No prompt told the model to finish and no loop detector
rewrote its history. The external harness was stopped during turn 5, before
another tool call completed. This is an observed repetition failure, not a
measurement of eventual completion rate or proof of permanent looping.

The isolated Kilo runs necessarily have different timestamps and configuration
paths; repository state and natural tool results can also differ. The fresh
result is not an exact-token paired comparison with W4. Fixed-history replays
hold the messages and tool outputs constant separately.

The source late-history replay used the identical 5,549-token input, temperature
0, seed 71, eager FP16 computation and cleared caches. It returned 58 completion
tokens in 259.42 seconds, ending with `tool_calls`: “Let me now describe the
project.Let me check the current state of the file.” It then requested `ls -la`
instead of delivering the description. The complete-history replay also failed: all 29,577 prompt tokens were
processed, then it returned 132 completion tokens with `tool_calls` in
1,040.61 seconds. Its reasoning stated it had enough context and would summarize,
then proposed more git/AGENTS/structure inspection and called `ls -la` on the
parent directory. The supplied history already contained the gathered evidence.
See [source-late-replay.json](source-late-replay.json) and
[source-full-replay.json](source-full-replay.json).

These results show that the old uniform-W4 conversion loss is not necessary
for repetitive tool behavior in this runtime. They do not establish intrinsic
source-model behavior, exonerate the remaining runtime, or reproduce the source
release's quantized activation arithmetic. The numerical repair remains useful,
but promoting it as a coherence fix or optimizing its speed would be premature.

A follow-up [dense-file audit](dense-file-audit.json) found **zero checksum
mismatches across all 1,540 dense-export files** after the source-trial model
loaded. It read the ordinary file cache and completed in 4.19 seconds. This
checks host payload integrity; it is not a device-resident weight comparison.
No evidence from that audit links the earlier mapped-source loading problem
to the observed repetition in the successfully loaded trial.

| Controlled history | W4 expert weights, FP16 compute | Source NVFP4 expert weights, FP16 compute |
| --- | --- | --- |
| Late, 5,549 prompt tokens | 155 completion tokens; more discovery | 58 completion tokens; more discovery |
| Full, 29,577 prompt tokens | 221 completion tokens; more discovery | 132 completion tokens; more discovery |

Both fixed-history source trials ran with empty scheduler and physical KV
caches. Neither ended with a final project description. The full-history result
reproduces the original promise-to-summarize/discovery pattern, independently
of the distinct repeated-directory behavior in the fresh Kilo run. Source
weights change the generated continuation but do not eliminate the symptom.

### Device-resident source identity

After both failed fixed-history continuations, every source expert bank was
read back from its actual NPU allocation and compared with the verified export.
**All 342 banks / 1,026 code, scale and lookup tensors matched exactly**, across
all six ranks. The longest rank audit took 192.75 seconds. Worker PIDs and all
registered weight-storage identities remained unchanged. See
[source-resident-readback.json](source-resident-readback.json).

This is stronger than checking transferred files: the full-model trials used
the intended source code/scale/lookup payloads on device. It rules out a bad
expert transfer in this loaded trial, while leaving numerical execution,
remaining runtime behavior and model/instruction interaction unresolved.

### Source-kernel hot reload

The full-model resident controller loaded `minimax_w4_nvfp4_v4`, compared it
with v3 on real resident banks (three projections on each of six ranks, including
a peer-zero route), switched source dispatch to v4, then restored v3. All 18
direct-native comparisons and all 18 comparisons through the installed v4
resident dispatch were finite and exactly equal. All worker PIDs and
weight-storage identities stayed unchanged; no checkpoint shards were reloaded.
See [source-hot-reload.json](source-hot-reload.json).

The first switch exposed a namespace error: `SourceProjector` lived under
`tools`, outside the controller's allowed runtime replacement roots. The class
now lives in `vllm_ascend._310p.ops.minimax_m3`, and the regression test uses the
actual `PatchSession` prepare/apply/restore path. Already-running workers were
upgraded by adding a runtime alias to their existing class, preserving every
instance and allocation; fresh startup imports the runtime class directly.
This reload qualification does not change the failed coherence result.

### Retained service and next diagnostic boundary

The temporary source service remains available on port 8000 in its **text-only,
32K, eager v3** configuration, with deterministic operators. The original W4
160K launcher remains unchanged and its service is stopped. The source kernel
is slower and has not been promoted as the standard performance profile.

The next causal comparison needs an independent implementation or captures of
the actual looping request through production normalization, rotary positions,
sparse selection/attention and routing. Prior seeded-input layer probes do not
substitute for that comparison. Public hosted-model reports remain contextual
evidence, not a matched reference proving that this runtime is correct. No
loop-solving claim is justified by this work.
