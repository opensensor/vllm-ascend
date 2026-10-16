# TP6 fusion candidates prepared offline

These three candidates are opt-in experiments for the six 310P devices. They do
not modify the launcher, installed OPP package, model defaults, or resident
server. No NPU execution was performed during the initial preparation. CPU tests and CANN
compilation do not establish numerical equivalence or throughput on hardware.
Subsequent resident hardware results are recorded below; none qualified for
serving enablement.

## Implemented candidates

| Candidate | Implementation | Qualification risk |
| --- | --- | --- |
| HC projection operand reuse | Convert normalized activations to FP16/NZ once, then carry that operand from down projection to injection projection. Preserve FP32 gated mean and checkpoint weight storage. | CANN may choose different GEMM behavior for an explicitly formatted operand. Test complete mix/combine at 3, 6, 12, and 18 rows. |
| W4 gate/up, SwiGLU, and packing | Private native kernel pairs gate/up tiles in local memory and writes the four packed operands directly. Reuses pinned INT4 and SwiGLU arithmetic with the FP16 boundaries intact. | N=128 tiling differs from retained schedules. The revised schedule uses all eight cores and caches paired weights, but complete MoE calls still regress. |
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

## October 16 resident gates with contexts preserved

The tests used the existing six workers in `resident-v8`, API PID 605287, on
port 8001. Each gate drained requests using
`/pause?mode=wait&clear_cache=false`, borrowed loaded weights, exercised private
tensors, and resumed serving. No model method was installed, no graphs were
recaptured, and no cache reset or restart occurred.

Receipts are in `artifacts/qfn-tp6-preserve-fusions-20261016/`. Before/after
checks cover worker and weight identities, resident patch generation, graph
entry ownership, KV storage addresses, Mamba checkpoint mappings, and scheduler
block hashes/reference counts. Later gates also check the primary prefix-hash
map explicitly: the older `cached_blocks` receipt field counted the separate
alias index, which was empty. These are metadata/ownership checks, not a
bytewise copy of all cached tensor contents. Each worker retained its 12 graph
entries. The packed-zero gate retained all 2,606 primary hash mappings.

| Candidate | Hardware result | Decision |
| --- | --- | --- |
| Ordered finalizer | Exact against the FP32 reference on all six ranks, including poisoned peer tails and three real-weight projections. The live `cann_v2` finalizer uses different FP16 rounding. At 256 tokens, medians were approximately 0.59 ms candidate versus 0.36 ms retained. | Keep the live finalizer. |
| HC FP16/NZ reuse | 430/432 complete mix/combine comparisons were exact. Mismatches occurred at 128 rows on ranks 1 and 2. No consistent timing improvement. | Do not enable. |
| Original fused W4 gate/pack | 162/162 packed and downstream comparisons passed. Mixed 650-route case approximately 4.0 ms versus 2.7 ms retained. | Revise scheduling. |
| Revised fused W4 gate/pack | 324/324 comparisons passed, including grouped and routed inputs. Mixed 650-route case approximately 2.39 ms versus 2.74 ms retained; mixed 2,560-route case approximately 4.63 ms versus 4.83 ms. Dense local ownership remains slower. | Continue as an opt-in diagnostic. |
| Complete MoE chunk with revised gate/pack | 864/864 comparisons passed across all 48 loaded MoE modules, all six ranks, and 65/128/256-token inputs. Complete calls generally regressed 2–5% at 128/256 tokens. | Do not bind to serving; isolated operator gains did not survive integration. |

Timing uses counterbalanced, synchronized wall-clock samples. It includes
submission overhead and is a component measurement, not serving throughput.
Real checkpoint weights were used with generated inputs; these results do not
constitute full-model accuracy or concurrent serving qualification.

The revised experimental kernel distributes expert/group pairs across eight
cores, retains both projection halves in separate L1 banks across row tiles,
and writes the exact zero-input packing representation for peer rows. It keeps
the frozen K accumulation and FP16 boundaries. The zero writer is also covered
by a CPU harness that executes its generated implementation with bounds and
event checks, including an untouched active prefix and a 20,480-row allocation.

`tp6_finalize_gate.py` and `tp6_resident_fusion_gate.py` generate diagnostic-only
engine candidates; their `apply` method deliberately rejects installation.
`tp6_eager_gate_pack.py` provides a source-qualified eager replacement for
private complete-chunk comparisons. It preserves the loaded finalizer and
falls back before launch for capture, training, unsupported geometry/policy,
or exhausted descriptor admission. It is not installed automatically.

The HC reporter initially failed after completing its comparisons because
`inspect.getsource` cannot recover dynamically compiled resident methods.
Results were recovered from retained diagnostic owners; all six ranks passed
synchronization and unchanged-cache checks before resume. The reporter now
accepts unavailable source metadata and retains records incrementally.
