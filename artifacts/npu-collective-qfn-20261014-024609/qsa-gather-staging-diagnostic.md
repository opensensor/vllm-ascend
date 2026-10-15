# QSA gather staging: separate kernel cost from submission overhead

The private native4 control uses the qualified processing header byte-for-byte,
including its four-block staging. Reported T64 dense V event intervals are about
0.43 ms installed, 0.48 ms native4 and 0.49 ms native8. Sparse V intervals are
about 0.55 ms installed and 0.59 ms native4. These parent-reported measurements
show a common private-path cost; they do not isolate an eight-block staging
regression. Retain the K-only serving candidate while investigating.

## Evidence from source and the existing device trace

- Both installed and private timed paths keep cache/output in FP16 NZ format29.
  Neither converts NZ to ND or FP16 to FP32. Metadata casts to INT32 match the
  installed Python helper; the timing input metadata is already INT32.
- The timing helper records its start event before Python admission, descriptor
  lookup, metadata preparation and host command submission. A queued event
  interval can therefore include device idle time while the host prepares work.
  It is not equivalent to the native kernel's execution interval.
- The private bridge calls `OpCommand.Run()` followed by producer-side
  `currentStream.stream(true)`. This drains the framework's host submission
  queue before the end event is enqueued. It must remain in this qualified
  bridge; removing it is not a safe diagnostic shortcut.
- Native4/native8 share that bridge, append-only descriptor ownership and
  descriptor event bookkeeping. They also share eleven INT64 descriptor loads
  and scalar geometry validation in the ACLRTC entry. The installed entry
  deserializes its own eleven-field GM tiling structure. Instruction-level
  differences have not been measured.
- The installed source declares `KERNEL_TYPE_AIV_ONLY`, while ACLRTC required
  `KERNEL_TYPE_AICORE` for this device. The existing cold-prefill
  `native-timeline-v1/cold-prefill/kernel_details.csv` reports all 975 installed
  T64 K calls and 975 installed T64 V calls as **AI_CORE, Block Num8**. Private
  T64 H1 launches also use eight blocks. The source annotation alone therefore
  does not establish a runtime core-classification or launch-count mismatch.
- Installed T64 kernel-duration medians in that earlier serving trace were
  564.922 us K and 934.609 us V. The V calls overlap other work and cover many
  selections; these old medians are not matched comparisons with the new
  private component test.

## Bounded diagnostic

Use the existing paused/drained resident private RPC with loaded resources and
completed shape qualification:

```json
{
  "action": "profile",
  "build": "/absolute/existing/build-r2",
  "rows": [64],
  "repeats": 2,
  "output": "/absolute/new/artifact/profile-directory"
}
```

`qsa_gather_diagnostic.py` uses the established TorchNPUProfilerWrapper settings
and labels matched installed/native4/native8 calls by shape, density, K/V and
repeat. It includes the existing warm-up pass, output-buffer reuse and the
qualified bridge drain. Each rank writes below `output/rankN`. The session is
one-shot and retained on the worker. Successful stop releases the worker's
active-profiler slots; an uncertain start/launch/stop retains ownership and
tracebacks, with no automatic retry or resume. It allocates no model KV cache
and changes no serving binding.

For each labeled call, extract the actual gather kernel start/end, preceding
host submission/end-event gaps, block count and core classification. Compare
native8 versus native4 first, then native4 versus installed. Preserve original
wall/event measurements as component costs, but do not report instrumented
profile timings as throughput improvements. If actual native4 and installed
kernel durations agree, focus the next candidate on submission/bookkeeping.
If they differ, inspect ACLRTC versus OPP code generation and entry descriptor
loads before changing staging policy.

This diagnostic has CPU lifecycle tests only until the parent runs it on the
resident service. It does not establish graph safety or a V performance win.
