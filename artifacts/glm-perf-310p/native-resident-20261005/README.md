# Resident GLM: append-only native loading

The user authorized extending the resident harness and then reloading GLM.
Implementation and validation use shared main; build packages remain isolated.

## Implemented

- `load-native manifest.json`: drain and pause, verify every worker's library
  and binary hashes, reject operator collisions, load and probe, verify unchanged
  worker PIDs / weight-storage digests / dispatch / graph state, then resume.
- Native libraries and device code remain resident after dispatch rollback.
  Repeating an identical manifest does not register or allocate again.
- Native failure after mutation leaves the server paused and blocks resume;
  Python rollback cannot repair a failed device or undo native registration.
- `prepare()`/`validate(resource)` retains per-worker resources on `NativeSession`.
  Python candidates receive those resources explicitly, without device loading
  during the patch preparation phase.
- Public inference on 8001 with local-only administrative routes, using
  `ResidentControlMiddleware` and launcher mode `public-local-control`.

## Direct Ascend kernel bridge

`bridge.cpp` registers the versioned `glm_native_v1::launch` operator and a
`Kernel` custom class. Runtime binary registration uses
`aclrtBinaryLoadFromData` plus `aclrtBinaryGetFunction`, then
`aclrtLaunchKernelWithArgsArray` through the torch-npu command queue. It neither
changes `ASCEND_CUSTOM_OPP_PATH` nor depends on late OPP discovery.

Tensor arguments remain owned until command submission and are then released,
so completed queue entries do not retain large prefill intermediates. Binary
handles are deliberately not unloaded while captured graphs may reference them.
Use fresh names for subsequent library versions, not in-place replacement.

`compile-kernel.cpp` invokes ACLRTC for `dav-2002` without creating an NPU device
context. `prefill-direct.cpp` is the separately compiled four-query prefill
kernel with an explicit direct-launch entry. The ordinary OPP candidate is
recorded separately in `../kpool-prefill-tiled-20261005/`.

## Validation so far

- **76 CPU tests passed**, covering native manifest validation, changed artifacts,
  repeated loading, retained resources, operator collisions, partial failures,
  worker agreement, weight identity, pause/resume, Python rollback, graph cleanup,
  public/local routing, and prefill wrapper geometry.
- CANN compiler and supplemental C++ bridge build succeeded.
- Initial `aclrtBinaryLoadFromFile` attempt rejected the explicit binary magic
  option. Switching to the previously demonstrated in-memory load API fixed it.
- Hardware loaded the library **after** creating an NPU context and a live
  tensor. That tensor's address and contents stayed unchanged.
- Direct-kernel gate: **12 of 13 cases passed**. Native score comparisons at
  tile/page boundaries and through 311K equivalent context passed tolerance.
  Partial wrappers, invalid geometry, poisoned padding, and exact ties passed.
- The 640-row random full-selector case changed 1572 of 1312640 sorted token
  index entries (about 0.12%). This is not a count of independently changed
  selected pools: one cutoff change shifts several sorted entries. The candidate
  is **not enabled for serving**. No speedup is claimed.

The public GLM launch is `serve-public311k-native-harness.log`, API PID 1192244,
with the validated selector, native mHC, MTP1, full graphs `[2,8]`, 640-token
prefill chunks and 311040 configured context. The native manifest registers
and validates the candidate only; loading it does not switch model dispatch.
The resident full-model check completed at 17:20:38 UTC:

- Library and device kernel were loaded **after model loading and graph capture**
  through the harness, with no further worker restart.
- All four workers passed the exact small native probe. The load/validation
  transaction took **0.585 seconds** on an idle server; this is not inference
  throughput and excludes the initial model restart.
- PIDs **1194443, 1194895, 1195395, 1195969** and every recorded weight-storage
  digest were identical before/after. Dispatch and graph state also matched.
- Deterministic inference returned `56` before and after the native load.
- Repeating the same manifest succeeded without re-registering the library.
- The scheduler resumed. The active candidate remains `baseline` on every rank,
  meaning the validated live selector/native-mHC configuration, not an older
  unoptimized server.
- A client outside loopback received HTTP 200 for `/v1/models` and HTTP 403 for
  `/is_paused`. Administrative access is local; public inference remains available.

Receipts, before/after identity records, request outputs, and the public-route
check are alongside this document. Subsequent compatible versioned native
loads now have a demonstrated path that retains GLM's resident weights.

## Commands

After the instrumented server is ready, on the host with the runtime environment:

```bash
python -m tools.glm_perf.resident_harness load-native \
  /home/matteius/experiments/glm-native-resident-20261005/prefill-v1.json
python -m tools.glm_perf.resident_harness status
```

The manifest contains the exact bridge, kernel and Python-wrapper hashes plus
the small boundary validation source. No request-serving Python patch is
included in that command. `kpool_prefill_direct.py` is staged for future
selection qualification, not installed in the serving model.
