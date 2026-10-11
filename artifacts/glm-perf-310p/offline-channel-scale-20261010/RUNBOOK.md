# Offline candidate artifacts and qualification

## Artifacts

Local original source:

```text
/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8
```

Permanent full candidate overlay:

```text
/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-channel-W4-G256-20261011-offline
```

This is an expert overlay, not a standalone published model. It supplements the
qualified base GLM checkpoint's tokenizer, config, dense/shared/attention weights,
and indexer bundle. It contains all 12,096 routed experts in layers 3 through 44.
The exporter must finish with a complete manifest before admission;
[storage verification](full-overlay-storage-verification.json) records each
shard SHA and native tensor header. No original W2/W3 files are overwritten. It has not been uploaded to Hugging Face
or staged onto the NPU host.

The eight-expert subset is available for a small operator gate:

```text
/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-channel-native-sample-20261011
```

It covers layers 3 and 14, expert IDs zero through three. Its `complete` marker
means this selected subset is complete, not a full-model checkpoint.

CPU-compiled bundles on the host's other 4 TB drive:

```text
/srv/ai/storage/artifacts/glm-offline-channel-scale-20261010/v11-integer-products
/srv/ai/storage/artifacts/glm-offline-channel-scale-20261010/v12-integer-reciprocal
/srv/ai/storage/artifacts/glm-offline-channel-scale-20261010/v13-route-address-cache
```

Build logs are `v11-build.log`, `v12-build.log`, and `v13-build.log` in the parent directory. Each
bundle snapshots sources and Python helpers, owns a versioned native namespace,
and verifies ABI, compiler macros, and binary/helper SHA hashes. Device operator
profiling names distinguish quantization, fused gate/up, and fused down stages.
Neither compilation nor verification opens an NPU device.

## CPU gates

From the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. pytest \
  --confcutdir=tests/ut/glm_perf -q \
  tests/ut/glm_perf/test_channel_scale*.py \
  tests/ut/glm_perf/test_channel_broadcast_cpp.py \
  tests/ut/glm_perf/test_channel_row_cache_cpp.py \
  tests/ut/glm_perf/test_channel_prefill_protocol_cpp.py \
  tests/ut/glm_perf/test_packed_pipeline_all_gather.py \
  tests/ut/glm_perf/test_pipeline_all_gather.py
```

The confined collection bypasses an unrelated local upstream vLLM import issue
in the parent unit-test conftest. This gate uses CPU tensor math, safetensors,
production C++ fragments with CPU stubs, and simulated collectives.

Use `probe_channel_scale --plan-only` to check real native-bank shapes, hashes,
and calculated scratch before any device work. Copy the small kernel bundle to
an offline machine if its Torch backend would otherwise autoload; set
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` during CPU-only build/admission.

Export resumes from `completed.jsonl` using `--resume`, with identical source,
index SHA, layer/expert selections, groups, and clipping. Completed shards are
SHA-verified before being reused; no completed expert is requantized. The final
manifest is replaced atomically and stays incomplete until all selected experts
are saved. Export reads one original FP8 expert at a time on CPU.

## Required device qualification, not yet run

1. Reserve a chip explicitly and run the real-bank operator gate with changed
   inputs, routes, and weights during graph replay. Exercise row counts spanning
   physical M16/M32/M48/M64 (two- and three-token cases deliberately concentrate
   routes in one expert for M32 and M48), repeated experts, empty/local/peer routes, decode, and
   640-token prefill. Validate direct-division v11 first, then compare v13
   address caching with the same numerical contract. Compare reciprocal v12
   separately because its activation rounding changes.
2. Profile each named stage using absolute Cube, vector, scalar, and memory time.
   Measure operand DMA, barriers, padded Cube rows, quantization, routing, and
   readback. Source instruction counts and percentages alone cannot establish a
   speedup. Use the same real weights and matched inputs for schedule comparisons.
3. Complete a cold TP2/PP3 load using the existing composed GLM adapter. The
   candidate loader requires complete coverage of every local MoE expert. It
   excludes all legacy/original expert tensors before materialization and leaves
   the shared expert, scaling, and TP reduction with their existing owners.
4. Validate actual per-rank peak memory before allocating KV/graph pools. The
   loader's explicit device reservation covers only new experts and their runtime
   allowance. Dense/KV/state/HCCL/PP/graph resources require separate headroom.
5. Run full-model coding, tool, reasoning, cold-prefill, and decode checks. Confirm
   output quality with the user before promoting this different quantization
   contract. Check changed-input graph replay and independent returned tensors.

Example operator command after hardware has been released:

```bash
python -m tools.glm_perf.probe_channel_scale \
  --overlay /path/to/native-sample \
  --bundle /srv/ai/storage/artifacts/glm-offline-channel-scale-20261010/v11-integer-products \
  --tokens 1 2 3 4 17 37 640 --device 0 \
  --output /srv/ai/storage/artifacts/glm-channel-real-bank-gate.json
```

The nightly wrapper accepts `--glm-channel-overlay`, `--glm-channel-bundle`, and
`--glm-channel-device`. Ordinary collection skips it without backend initialization.
The direct operator tool accepts `--decode-bits 4` and `--prefill-bits 8` for
a separate activation-precision trial; defaults remain A8/A8. Compare its
latency and full-model quality before changing serving defaults.
Its implementation NRMSE threshold is 0.02 against the same quantization contract;
this checks kernel implementation, not error against the original model.

## Cold-load integration

[Candidate options](candidate-loader-options.json) specify the explicit
`glm_channel_scale_candidate` load format, A8 in both modes, 8 GiB CPU staging,
and a 31 GiB expert/runtime device reservation. The listed overlay host path is
planned: the full overlay currently remains local. The fragment is not an
automatic launcher and must not be used to bypass the separate memory plan.

Keep the qualified local base model's architecture and compatible attention/KDA
configuration. Supply `model_loader_extra_config` to the named loader, use
TP2/PP3 with expert parallelism, and initially retain context 8192, batch 640,
four sequences, and decode graph sizes one/four. No MTP is admitted for this new
contract until target-only qualification passes. No old MoE monkeypatch should
overwrite the new method after loading.

The packed pipeline transport is a separate opt-in component. All pipeline ranks
must switch together before requests or capture, using `take_ownership` on an
already initialized transport. Do not create additional HCCL groups. Compare its
added copies and temporary allocations against the existing four-collective
transport before promotion. It is not activated by the new expert loader.
