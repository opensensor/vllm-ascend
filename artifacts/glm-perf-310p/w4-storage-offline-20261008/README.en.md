# Lossless W4 storage candidate — October 8, 2026

[Chinese report](README.md)

The new CPU exporter stores selected W2/W3 expert codes in native W4 packing on
disk. Signed integer values and weight scales stay unchanged. The prepared W4
branch in `glm_fused_moe.cpp::Decode` copies nibbles directly into the Cube weight
buffer; it skips the W2/W3 byte reconstruction, sign extension and casts. The
current W3 path already uses INT4 Cube products after reconstruction, so this
candidate does **not** reduce the number of Cube operations or change arithmetic
precision. More packed-weight traffic may offset the reconstruction savings.

No server operations, NPU selection, kernel loading, graph capture or model
conversion occurred. No latency or tokens-per-second improvement is claimed.
The prior pre-rounded-scale candidate remains a separate offline candidate; this
exporter preserves its scale markers and kernel contract when used together.

## Concrete memory cost

The permanent checkpoint at
`/srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p` contains 288 experts
across four ranks, 72 experts per rank. Its 43 routed layers, including the draft
layer, have 22 W4, 8 W3 and 13 W2 storage layouts. The eligible W3 layers are
8, 9, 10, 12, 13, 14, 16 and 17. Many layers already take the W4 direct-copy path.

| W3 bank promoted | Extra resident code bytes per rank | New disk payload across all ranks |
| --- | ---: | ---: |
| One gate/up bank | 144 MiB | 2.25 GiB |
| One down bank | 72 MiB | 1.125 GiB |
| One complete layer | 216 MiB | 3.375 GiB |
| Four complete layers | 864 MiB | 13.5 GiB |
| All eight W3 layers | 1.6875 GiB | 27 GiB |

These are exact shape-derived code costs, **not measured free HBM**. The planner
requires an explicit allowance for additional resident codes after reserving KV
cache, graph pools and workspaces. It does not infer an allowance or change
context settings. The 1 GiB and 2 GiB allowances in
[memory-budget-examples.json](memory-budget-examples.json) are illustrative.
All eight W3 layers exceed the 1 GiB example and fit the 2 GiB example. No bank
selection is justified by a measured speedup yet. W2 widening costs twice the
original code bytes; W3 widening costs one third more.

## Delivered implementation

- `tools/glm_perf/w4_storage_checkpoint.py`: header-only inventory and budgeted
  export. Select whole gate/up or down banks across every expert/rank. Widen
  signed code fields directly on CPU; verify inverse packing. Hard-link
  unchanged indexed shards, preserve scales/dense/draft/indexer data, copy the
  verified coupled native kernel assets, and publish `complete=true` last.
- `tools/glm_perf/w4_storage_probe.py`: deferred complete native routed MoE
  comparison using one frozen build. Compare gate/up-only, down-only and both
  against the original packing, with separate native scratch allocations.
  Input quantization, routing, gate/up, SwiGLU/hidden quantization, down and route
  reduction stay inside graph replay. CPU preparation stays outside timing.
- `vllm_ascend/model_loader/glm_native_int4.py`: verify new promoted code-shard
  checksums once during loading. No conversion is added to serving or capture.

The probe covers A4/A8 and 2/8/17/640 tokens. It requires finite, bitwise-equal
FP32 outputs; changes activations, codes, scales and routes without recapturing;
and checks all-peer and zero-weight routes for stale output. It alternates
baseline/candidate replay timings. An optional real-weight fixture reads full
GLM matrix dimensions for a small contiguous local expert subset. This is an
operator gate; it does not qualify full-rank inference, communication, long
context attention or model quality. `real_model_evaluated` stays false.

## Offline evidence

- [Real checkpoint inventory](real-checkpoint-inventory.json): safetensors
  headers and index only, without mapping tensor payloads or querying devices.
- [Real code CPU gate](real-code-cpu-gate.json): independent signed-code decoding
  checked **150,994,944 values** from gate/up and down banks of layers 8 and 17,
  experts 0–2. Original W3 and widened W4 values match exactly. Scales have
  unchanged source digests. No checkpoint was written and no kernel was loaded.
- [Compatible GLM suite](cpu-tests.txt): **1,576 passed**. The same three files
  requiring unavailable upstream/NPU dependencies remain excluded:
  `test_glm5next_w2_assembly.py`, `test_grouped_gate_up.py`, `test_kpool_ops.py`.
- [Final regressions](final-regression-tests.txt): **64 passed** after the
  incremental export integrity checks. Cover both byte dtypes, every signed
  W2/W3 code, multiple tiles/experts, exact budgets, loader/bank width admission,
  interrupted exports, checksum rejection and pre-rounded-scale composition.
- [CPU gate source snapshot](real-codes-cpu-source.tar.gz): the exact CPU probe
  and helpers staged for the real code check. Later exporter lineage fixes are
  covered by the final regressions. See `evidence-manifest.json` for hashes.

The remote CPU commands used PyTorch's existing
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` switch to prevent backend imports. No new
Ascend environment variable was added. ACLGraph, real model decode, cold prefill,
EP/communication, MTP inference, multimodal inference and capacity testing are
**not run** in this delivery, following the user's device deferral.

## Offline commands

Run from the repository with CPU dependencies installed. On the Ascend host,
prefix CPU commands with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.

```bash
python -m tools.glm_perf.w4_storage_checkpoint plan \
  --source /srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p \
  --bank layers.8.gate_up --bank layers.8.down \
  --extra-bytes-per-rank 226492416 --output /tmp/w4-layer8-plan.json
```

The byte allowance above is exactly 216 MiB for illustration; it is not a
statement that the running allocation has that much space. After establishing a
real memory allowance and selecting the candidate, an export command is:

```bash
python -m tools.glm_perf.w4_storage_checkpoint export \
  --source /srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p \
  --bank layers.8.gate_up --bank layers.8.down \
  --extra-bytes-per-rank 226492416 \
  --output /srv/ai/models/GLM-5.3-Flash-W4-storage-layer8-candidate
```

The destination must be new and on the source filesystem. This command has
**not been run on the real checkpoint**. Disk payload is larger than the
resident delta because unchanged original files remain hard-linked. To compare
variants of the same layer, export each from the same base checkpoint into its
own destination. Existing destinations and conflicting promotion shards are
rejected. Incremental exports of different layers retain prior integrity records;
the allowance is additional code memory relative to the supplied source.

## Deferred hardware and model gates

When device testing resumes with exclusive device access, run the complete MoE
probe against the intended frozen build. Example, not executed:

```bash
python -m tools.glm_perf.w4_storage_probe \
  --build /srv/ai/artifacts/glm-prefill-weight-reuse-20261007/build957 \
  --checkpoint /srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p \
  --layer 8 --experts 3 --tokens 17 640 --activations 4 8 \
  --allow-device-gate --output /tmp/w4-layer8-prefill-gate.json
```

Use decode build956 with tokens 2 and 8 for the decode comparison. The probe
verifies all coupled binary/helper checksums before device selection and uses the
same build options for every packing. W3 specialization, if present, remains the
original dispatcher behavior; W4 uses the build's generic direct-copy branch.

After operator replay parity and a favorable measured tradeoff, validate an
exported candidate with the existing `glm_native_int4` loader, full target/draft
weights, model inference and identical context/KV/graph settings. Measure cold
long-prompt TTFT and decode tokens/s end to end. Do not promote this candidate
based on synthetic timing, CPU code equality or successful startup alone.
