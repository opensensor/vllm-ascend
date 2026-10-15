# Qualified W4 prefill candidates

Status: CPU tests and host-only ACLRTC compilation passed. No device kernel
execution, model binding, or serving performance is qualified by this note.

Subsequent controller-run component gates passed on all four ranks, but the
local-count serving trial regressed. See the
[offline regression audit](w4-prefill-schedule-regression.md) for receipts and
limitations. The implementation handoff below does not recommend promotion.

## Profile exposure and priorities

The cold 6,849-token prefill trace contains 4.827 seconds of W4 matmul work.
The pinned kernel selects M=128, N=64 whenever **allocated** route rows exceed
128 experts × 64 rows. A 2,048-token top-k-10 chunk allocates 20,480 route rows,
but a balanced EP4 rank owns about 5,120. The small schedule uses M=32, N=160
for both model projections. This is the largest candidate exposure.

Both schedules iterate actual device INT64 expert boundaries and zero the
peer tail through the allocated row capacity. Tile selection changes work
partitioning, not route coverage. Their arithmetic remains the frozen signed
INT4 kernel with the same ordered FP32 epilogue. Exact parity remains a
hardware gate because different tile shapes change the executed native code.

The second candidate fuses only the last route cast, multiplication, and
inverse permutation. At 20,480 × 2,560, the trace contains 233.743 ms of
FP16-to-FP32 cast, 296.315 ms of FP32 multiplication, and 308.385 ms of FP32
gather: 838.443 ms total exposure. This is not a predicted saving.

The fused kernel reads FP16 sorted rows, casts to FP32, multiplies by the
original-order FP32 weight, and writes FP32 original-order rows. The existing
`reshape(tokens, top_k, hidden).sum(1)` remains unchanged. No activation
quantization, weight layout, FP16 rounding, FMA, or reduction-order change is
included. It removes two large intermediate passes but still traverses every
route, including peer-zero rows.

## Frozen sources and build

`build_route_finalize.py` requires the exact qualified schedule and seven-field
descriptor headers by SHA256; it rejects the newer local headers. It compiles
the finalizer plus baseline, forced-small, and device-local-count schedule
entries into versioned private binaries and a generic bridge. Device-local
selection reads the last group boundary inside the kernel; there is no host
readback. The local-count threshold remains 64 rows per expert, matching the
pinned threshold while correcting which rows it counts.

Successful host-only build:

```text
/srv/ai/storage/artifacts/qfn-route-finalize-compile-20261014-v1/build2/provenance.json
namespace: qwen_route_finalize_v1
```

The first attempt encountered torch backend autoload, which failed on a missing
HCCL library before device initialization. The builder now requires existing
upstream `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before importing torch. A second
attempt needed the SDK shell environment for transitive compiler libraries.
The successful build used the SDK environment and disabled backend autoload.
No NPU device was opened by the build or compiler.

## Resident gate contract

Stage the new Python files and existing `native_prefill.py` unchanged, then add
only `qfn_w4_prefill_probe.replacements()` to the paused resident payload.
It temporarily uses `qfn_cpu_profile_stop` with a JSON string argument.
Ordinary non-JSON profiler-stop arguments retain their prior behavior.

Example request, repeated with the desired action:

```json
{
  "action": "preflight",
  "build": "/srv/ai/storage/artifacts/qfn-route-finalize-compile-20261014-v1/build2",
  "rows": [650, 20480],
  "repeats": 1,
  "projection": "down_proj"
}
```

1. Drain and pause inference; stop both profilers.
2. Call `preflight` on every rank and require every receipt to pass before load.
   At 20,480 rows, required free memory is 1,107,296,256 bytes. This includes
   conservative intermediate allocation and a 256 MiB reserve. The gate does
   not empty allocator caches or change model/KV ownership.
3. Call `load`; every resource is instance-owned before use. Partial failures
   retain ownership. Bridge, binaries, and frozen sources are hash checked.
4. Call `schedule` first at 650 rows as a small-schedule control, then at 20,480
   rows for each projection. It borrows an actual resident bank read-only,
   packs private activations, and compares every output bit against the
   installed operator. Cases cover all-peer, balanced EP4, all-local, and
   concentrated expert-71 routing. Separate wall timings are component-only.
5. Call `validate` for finalization, starting small. Changing permutations,
   weights, local/peer boundaries, signed zero, subnormals, and large finite
   values must preserve exact weighted-row bits and exact FP32 route sums.
   At least three cases always cover all-peer, mixed, and all-local routes,
   including when the requested `repeats` is one; zeros alone cannot qualify.
6. `timing` requires prior successful finalization validation for each exact
   requested row count. It compares the three-operation baseline and fused
   kernel with synchronization; it does not measure serving throughput.

A numerical or native failure leaves the owned probe state failed and prevents
further native submissions through it. It never resets a device or unloads
native resources. The controller decides recovery. Do not mix unreviewed
operator bindings into this component gate.

Native launch and synchronization exceptions also retain the first exception
and traceback on the worker, preserving tensor-bearing Python frames. Rejected
retries cannot overwrite that retained failure. Warm-up results remain owned
through synchronization. Successful schedule receipts are retained in
`schedule_receipts` only after the final device drain, so a later binding can
require exact projection, row-capacity, and route-pattern evidence. Both CPU
and native profilers must be stopped before this private gate.

The separate `qfn_route_finalize` model candidate requires a successfully
qualified resource and the SHA256 of the pinned `_forward_grouped` function.
It substitutes one epilogue block and preserves original behavior outside
native INT4, FP32 compute, FP16 params, hidden 2,560, top-k 10, and supported
chunk sizes. This candidate has not been installed by the implementing agent.

## Validation still required

CPU simulation proves indexing, operation order, fallback, source guards,
bounded allocation admission, integrity checks, and failure ownership. Host
compilation proves the CANN/native ABI can compile. Neither proves device
numerics, DMA ordering, native latency, thermal behavior, or model quality.

After component parity and timing pass, gate one serving binding at a time on
cold prompts and changing routing. Keep the QSA/draft/collective baseline
constant. Record cold TTFT, prefill throughput, decode throughput, output
coherence, temperatures, and fresh trace coverage before claiming a gain.

## Separate schedule binding

`resident_candidates/qfn_prefill_schedule.py` provides the controlled
`make_replacements(worker, mode="local_count")` factory; `mode="small"` selects
the forced-small schedule. Its no-argument factory intentionally rejects
activation because it cannot establish ownership or qualification.

Preparation requires the worker's existing loaded, healthy probe state and
synchronized successful `schedule_receipts`. Both projections at 650 and
20,480 allocated routes must pass all four routing cases against the installed
operator for baseline, small, and local-count kernels. The preparation factory
returns only `PackedExpertBank.native_linear`, after checking the exact pinned
function SHA256. It loads no binaries and mutates no resource dictionaries.

The resulting closure strongly owns the selected native resource. It performs
metadata-only admission for INT64 grouped boundaries, 128 local experts,
audited projection shapes, dtypes, layout, device, and 129–20,480 allocated
rows. Other inputs call the original installed operator. INT32 routed decode
takes an early exit before bank geometry inspection. No device count is copied
to the host; local-count schedule selection still happens inside the kernel.

This wrapper adds a small Python dispatch check to the installed method, so
unchanged decode mathematics does not establish unchanged decode latency.
Check both serving prefill and decode after any trial binding. Its 33 CPU tests
cover the positive geometries, decode and unsupported fallbacks, exact source
guard, all qualification requirements, and retained resource identity. Device
activation remains solely the controller's next qualified experiment.

Both model bindings now use `native_descriptor_admission.py` before native
submission. Existing descriptors remain usable when the 256-entry dictionary
is full or capture is active. A new descriptor falls back to the original
installed operations when it cannot be allocated. Finalization checks all
distinct full-chunk and tail keys together before processing any chunk, so a
255-entry dictionary cannot admit two missing shapes and fail halfway through
the request. Admission reads only Python metadata; it does not allocate,
evict, synchronize, or catch broad native exceptions. All 60 focused binding
and admission tests pass, including capacity and capture regressions.
