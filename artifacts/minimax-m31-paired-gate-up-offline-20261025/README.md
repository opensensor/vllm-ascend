# Offline paired gate/up candidate

Status: **offline candidate only; not CANN-compiled, device-tested or deployed**.
The active six-chip server was not contacted or changed during this work.
Startup settings are unchanged. The most recent hardware throughput remains
3.958 tokens/s; no performance improvement is claimed for this candidate.

## Mechanism

The compact route path has one input token shared by up to four experts.
`w4a8_gate_up.cpp` uses one launch and one per-core activation preparation across
both projections and all local routes. It reuses the existing W4 bank layout and
INT8 Cube implementation. Gate/up outputs stay separate FP16 tensors, retaining
the current rounding boundary before clipped SwiGLU. Down projection is unchanged.

For L positive local routes, the candidate reduces activation preparations from
2L to one and gate/up launches from two to one. It also avoids expanding the input
into route rows. Both weight decodes and matrix multiplies remain. The actual
speed benefit must be measured; these operation counts are not timing estimates.

`INT8_FULL_A_OFFSET` is beyond the weight and staged-activation L1 buffers. The
paired entry changes only weight/output bindings between projections. Its
activation reuse flag is local to one launch and resets before the next token.
Peer-only groups skip both projections; the existing caller masks unwritten rows.

## Offline evidence

- CPU runtime tests compare separate and paired dispatch with mixed and all-peer
  routing and retain prefill fallback behavior.
- Metadata/ABI tests reject invalid input before launch, keep separate bank and
  output addresses, and reuse prepared metadata without copying route IDs to CPU.
- The build packaging test stubs unavailable compilation, exercising source and
  symbol selection, optional binary assets and tamper detection. It is not a
  successful CANN compilation receipt.
- The host C++ harness executes the actual entry's pointer selection, skips peer
  IDs, and asserts activation preparation is requested once per launch. Its CANN
  and tile stubs do not model device math, synchronization or memory retention.

## Deferred hardware steps

Run these only when NPU work is available. No commands below were run during
this offline iteration. Use a fresh namespace/output directory if already used.
The compiler command does not open an NPU; the probe does. Copy/update source
into an isolated development checkout before building, not into the live runtime.

```bash
python -m tools.minimax_m3.build_resident_native \
  /vllm-workspace/minimax-native-int8-gate-up-v16 \
  --namespace minimax_w4_int8_gate_up_v16 \
  --wide-l1 --rint-unpack --int8-compute --scale-cache --paired-gate-up

python -m tools.minimax_m3.paired_gate_up_probe \
  --candidate /vllm-workspace/minimax-native-int8-gate-up-v16/manifest.json \
  --baseline /vllm-workspace/minimax-native-int8-v15/manifest.json \
  --bundle /srv/ai/storage/artifacts/minimax-m31-20261024/expert0 \
  --device 5 --iterations 10 --output /workspace/m31-paired-gate-up-probe.json
```

The probe compares separate and paired real gate/up matrices, including repeated,
unsorted and peer-only routes, in eager and captured execution. Native admission
also checks K1024/K3072/K6144 and changing inputs. Require exact masked outputs
and a useful timing gain before attempting a brief resident serving benchmark.
Do not infer end-to-end performance from this isolated projection benchmark.
The ordinary binary is bundled alongside the paired binary for unchanged down
projection and prefill dispatch. Both binaries and the Python helper are sealed
by the existing append-only native loader. Nothing changes live unless a later
resident transaction explicitly loads and selects this candidate.

## Local check results

The complete M3 CPU suite passed **258 tests**, with 24 NPU-specific skips.
All changed-file hooks passed. Required `bash format.sh ci` was also run in a
separate worktree; it failed on existing unrelated files (formatting, spelling,
and forbidden imports). Its hooks modified 225 unrelated files and none of this
candidate's files. Those fixes were discarded with the temporary worktree.
See `whole-tree-format.json` for representative diagnostics and `scoped-hooks.txt`
for the candidate checks. No whole-repository lint pass is claimed.
