# Add native operators with GLM weights resident

The resident worker extension supports loading versioned native libraries after
model initialization. It does not replace an existing operator registration or
refresh CANN's cached OPP search paths. Libraries must handle kernel discovery
themselves or use dependencies already registered in the process.

```bash
python -m tools.glm_perf.resident_harness load-native /absolute/path/native-v1.json
python -m tools.glm_perf.resident_harness status
```

## Manifest

Supply a JSON object with these fields:

| Field | Value |
| --- | --- |
| `name` | Unique version identifier, such as `prefill_v1` |
| `libraries` | Nonempty list of objects containing absolute `path` and `sha256` |
| `assets` | Same format for kernel binaries and other required files |
| `operators` | Nonempty list of new `namespace::operator` names |
| `validation_source` | Python source defining `validate()` |

`validate()` must return a JSON object with `passed: true`. An optional
`prepare()` can return a native resource; in that case `validate(resource)`
checks it. Resources stay on the worker session. A Python patch factory can
declare `replacements(native_resources)` to receive the resources by manifest
name, so preparation of a dispatch change does not itself load device code.

## Transaction and recovery

The client drains requests and pauses all workers. Every worker verifies file
hashes and rejects name collisions before mutation. Workers then load the library,
verify operator registration, prepare resources, run validation, and synchronize.
The client requires matching manifest receipts and unchanged worker PIDs,
weight-storage digests, active dispatch, and graph state before resuming.

Loading a library does **not** activate it in the model. Use the existing
`switch --candidate ...` transaction to change dispatch and recapture graphs
after the candidate passes its hardware gates. Rollback restores dispatch;
native libraries/device code remain loaded until worker exit.

An identical manifest can be loaded repeatedly without registering it again.
Changed files or reusing a name with different content are rejected. Failure
after native mutation leaves the server paused. `native_failed` blocks subsequent
switch/resume commands and requires a worker restart; Python rollback cannot
undo partial native registration or a device fault.

## Access

Use this only with trusted artifacts. Both native loading and the pre-existing
Python patch RPC can execute arbitrary code. Bind development APIs to loopback,
or configure `tools.glm_perf.resident_middleware.ResidentControlMiddleware` when
public inference is needed. It allows inference/health routes publicly and
requires loopback for administrative routes, without trusting forwarded headers.

## Validation

The first four-rank GLM load on Ascend 310P retained all worker PIDs and weight
storage, returned identical deterministic inference before/after, and completed
the native load plus small validation transaction in 0.585 seconds. An identical
repeat succeeded. This does not qualify the loaded experimental prefill scorer:
its full 640-row selection gate differed from the baseline, so serving dispatch
remained on the validated implementation.

Detailed receipts and the direct runtime bridge are in
`artifacts/glm-perf-310p/native-resident-20261005/`.
