# TP6 fusion candidates prepared offline

These three candidates are opt-in experiments for the six 310P devices. They do
not modify the launcher, installed OPP package, model defaults, or resident
server. No NPU execution was performed during preparation. CPU tests and CANN
compilation do not establish numerical equivalence or throughput on hardware.

## Implemented candidates

| Candidate | Implementation | Qualification risk |
| --- | --- | --- |
| HC projection operand reuse | Convert normalized activations to FP16/NZ once, then carry that operand from down projection to injection projection. Preserve FP32 gated mean and checkpoint weight storage. | CANN may choose different GEMM behavior for an explicitly formatted operand. Test complete mix/combine at 3, 6, 12, and 18 rows. |
| W4 gate/up, SwiGLU, and packing | Private native kernel pairs gate/up tiles in local memory and writes the four packed operands directly. Reuses pinned INT4 and SwiGLU arithmetic with the FP16 boundaries intact. | N=128 tiling differs from retained schedules. Only five of eight cores own quantization groups; saved transfers may not offset utilization and repeated weight loads. |
| Grouped prefill finalization | Fuse FP16 cast, original route weight, and inverse permutation for 85/86 local experts. Skip unreadable peer tails. The ordered variant retains the backend sum. | Sequential and pairwise native sums are separate experiments; neither is presumed equivalent to the backend reduction. |

The gate/pack resource supports routed INT32 IDs up to 128 rows, including
ten-route broadcast, and grouped INT64 ends up to 20,480 rows. It initializes
peer outputs to the retained zero-input packing result. It borrows packed
expert banks directly; it does not duplicate or repack weights.

All native resources require eager inference and reject graph capture. They
retain arguments after uncertain submission failures and refuse subsequent
reuse. Descriptor tensors are append-only and bounded. The bridge must be the
producer-owned launch implementation, not an older callback-release bridge.

## Offline validation

The focused suite covers native finalizer arithmetic, exact FP32 bit patterns,
nonfinite peer weights, invalid inverse indices, 85/86 expert tails, maximum
route capacity, descriptor geometry, failure ownership, operand lifetimes,
and private source hashes. The CPU native harness checks bounds, complete
output writes, and balanced events; it does not simulate asynchronous hardware.
The gate/pack CPU test checks its NZ indexing, not Cube/vector numerics.

```bash
python3 -m pytest -q --confcutdir=tests/ut/qwen38_1m \
  tests/ut/qwen38_1m/test_tp6_fusions.py \
  tests/ut/qwen38_1m/test_tp6_fusion_guards.py \
  tests/ut/qwen38_1m/test_local_route_epilogue.py \
  tests/ut/qwen38_1m/test_native_int4_route_groups.py
```

The integrated preparation run passed 163 tests, including the existing TP6
admission guards. Full repository collection was blocked
by missing `modelscope` and `vllm.third_party.flash_attn` dependencies.
Build identities and validation details are stored in
`artifacts/qfn-tp6-offline-fusions-20261016/`.

## Prepared build and CPU preflight

The compiled artifacts are on the existing host under:

```text
/srv/ai/storage/artifacts/qfn-tp6-fusions-offline-stage-v1/final-build
```

The candidate vendors the exact INT4 schedule from commit `139f20832` as
`tools/qwen4exp/tp6_frozen/native_int4_schedule.h.txt`. The filename preserves
its bytes under formatting checks; the builder emits it as a normal header.
Its original SHA256 remains mandatory. New optional schedules merged into main
are deliberately independent of this candidate. Other source hashes are also
checked. The prepared binary sources match the integrated builder exactly.

They were built using CANN's CPU compiler without opening a device. Source and
binary hashes are checked before loading. Importing the preflight entry point
does not import Torch, load a native library, or contact a server.

```bash
python3 -m tools.qwen4exp.benchmark_tp6_fusions_310 \
  --build /srv/ai/storage/artifacts/qfn-tp6-fusions-offline-stage-v1/final-build
```

To rebuild elsewhere, use a fresh destination, the Ascend environment, and the
existing ACLRTC compiler executable:

```bash
python3 -m tools.qwen4exp.build_tp6_fusions \
  --output /absolute/new/build-directory \
  --compiler /absolute/path/to/compile-hc-residual \
  --cann-root /usr/local/Ascend/ascend-toolkit/latest
```

## First hardware test when the server is available

Use a maintenance window and the same Python/CANN/custom-OPP environment as
serving. This isolated process loads one checkpoint layer and then one HC
module; allow memory for both the layer and temporary operator outputs. It
does not drain, stop, restart, patch, or automatically enable anything on a
server. Avoid running it alongside a server that has filled device memory.

Set `qfn_model` to the local QFN checkpoint directory, `qfn_bridge_sha` to the
bridge digest in the preparation receipt, and `qfn_result` to a new result path.
From a checkout containing this work:

```bash
python3 -m tools.qwen4exp.benchmark_tp6_fusions_310 \
  --build /srv/ai/storage/artifacts/qfn-tp6-fusions-offline-stage-v1/final-build \
  --bridge /srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/hc-producer-v3/qwen_hc_residual_producer_v1.so \
  --bridge-sha256 "$qfn_bridge_sha" \
  --model "$qfn_model" \
  --hc-prefix model.language_model.layers.0.attn_hyper_connection \
  --rank 0 --device 0 --quick --iterations 1 --repeats 1 \
  --output "$qfn_result" --run-npu
```

The runner checks random, zero, and changed inputs; all-local, mixed, and
all-peer ownership; exact packed bytes and metadata; downstream native
projection; finalization against the retained backend sum; and complete HC
mix/combine. It records results incrementally and times only cases that pass.
A failing optional sum makes the aggregate result fail even if the ordered
variant passes: inspect each mode separately. No result enables a candidate.

Remove `--quick` for larger prefill shapes, and repeat across ranks 0 through 5
to cover both expert counts and different checkpoint weights. Repeat with
representative later layers and HC prefixes, including MTP. The quick matrix
alone is insufficient for qualification.

Before retention, still required: exact tests against the frozen serving
stack, changed-input ACL graph capture/replay qualification and admission,
all-rank integration, paired end-to-end C1/C2/C4/C6 measurements, mixed long
prefill/decode, cache retrieval, and idle recovery. The standalone benchmark
does not measure these and deliberately leaves `serving_qualified` false.

## Integration seams

`tp6_operand_reuse.make_mix(original_mix)` returns a replacement to bind only
through the existing drained resident experiment controller. The original
combine consumes the saved operand; unsupported geometry uses the original mix.

`tp6_fusions.load_resources(build, bridge, digest, owner=owner)` constructs
private eager resources. Keep the caller-owned dictionary alive for the entire
experiment, including failures. `gate_pack(bank, prepared, ends)` replaces only
native gate/up projection plus SwiGLU packing and returns the same four-tensor
contract. `ordered(routed, inverse, original_weights, ends)` replaces the
finalization and backend sum together; weights are in original route order.
The other two named resources have that same contract with different sums.
There is no automatic model patch, fallback after launch, or serving enablement.
