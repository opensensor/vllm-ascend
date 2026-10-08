# GLM experiments with resident weights

The experimental MTP audit server can drain requests, switch target and draft
execution, replace selected Python functions, reset request/cache state, and
recapture graphs while keeping model weights allocated. A mode switch reuses
the existing captures. A Python change recaptures both target and draft graphs
for the configured shapes before requests resume.

## Start once

Use the existing packed-W3 checkpoint, matching OPP/binding, and staged source
tree with the updated `tools/glm_perf` files. At the next planned startup:

```bash
bash artifacts/glm-perf-310p/mtp-packed-candidate-20261004/serve-mtp-audit.sh \
  /path/to/staged/vllm-ascend /path/to/selective-W3-checkpoint
```

The launcher selects TP4, MTP1, synchronous scheduling and full decode graphs.
It loads the audit extension from the launcher's directory and enables the
existing upstream `VLLM_SERVER_DEV_MODE` APIs. The service binds to loopback
because these development APIs permit Python execution in the workers. Run the
client on that host, or use an SSH tunnel. Installation of this instrumentation
requires one initial startup; it cannot attach to an uninstrumented worker.

## Compare using the loaded model

From the staged source directory:

```bash
python -m tools.glm_perf.resident_harness status
python -m tools.glm_perf.resident_harness switch --mode direct-draft
python -m tools.glm_perf.resident_harness compare \
  --workloads fault fault4 --output /tmp/glm-resident-modes.jsonl
```

| Mode | Target | Draft |
| --- | --- | --- |
| `graph` | Graph replay | Graph replay |
| `direct-target` | Python forward | Graph replay |
| `direct-draft` | Graph replay | Python forward |
| `direct-both` | Python forward | Python forward |

Direct execution bypasses the graph wrapper while retaining the server's
graph-oriented metadata and MTP configuration. It is a replay diagnostic;
it does not establish equivalence to a separately launched eager server.

`compare` sends identical prompts and seeds in each mode. It records request
outputs, timing, control generation, source digest, rank, PID, and a digest of
weight storage addresses/shapes/dtypes. It rejects missing rank acknowledgments
or changed worker/weight identities. The default short workloads exercise one
and four requests; select `quality` for scored answers or `short` for the
existing 256-token workload. The final selected mode remains active.

## Reapply a Python candidate

First qualify the included reference candidate, which preserves the existing
MTP normalization math:

```bash
python -m tools.glm_perf.resident_harness switch --mode graph \
  --candidate tools/glm_perf/resident_candidates/mtp_norm_reference.py
```

A candidate defines functions plus a side-effect-free `replacements()` returning
a mapping from `module:attribute` to a new Python function. Ordinary class
methods use `module:Class.method`. Patch the alias that the existing caller
uses; editing a defining module alone does not replace earlier `from ... import`
references. The included candidate demonstrates both aliases.

The client sends the actual source text to every worker and verifies matching
digests and target lists before applying anything. Rapid file edits are compiled
fresh, without Python bytecode cache reuse. Each worker saves the original
functions and restores them before applying a different candidate. Candidate
module initialization and `replacements()` must not mutate model state, install
patches, load weights, or execute distributed operations.

```bash
# Edit the candidate, then reapply it with the same command.
python -m tools.glm_perf.resident_harness switch --mode graph \
  --candidate /path/to/candidate.py

# Force recapture of unchanged Python, or restore the original functions.
python -m tools.glm_perf.resident_harness switch --mode graph --recapture
python -m tools.glm_perf.resident_harness switch --mode graph
```

Omitting `--candidate` selects baseline Python. To retain a candidate while
switching modes or forcing recapture, supply its file again. Compatible native
experiments can select separately named operators already registered in the
process through a Python candidate. Weight layout, model construction and cache
allocation still require a restart. New versioned native libraries can now be
added with the transaction below; replacing an already loaded library or the
ordinary cached OPP search configuration is not supported.

## Add a native operator while weights stay resident

```bash
python -m tools.glm_perf.resident_harness load-native /absolute/path/native-v1.json
```

The manifest contains `name`, `libraries`, `assets`, `operators`, and
`validation_source`. Every library/asset has an absolute `path` and `sha256`.
Operators use `namespace::name`; use a fresh namespace for a new library version.
Validation source defines `validate()` returning a JSON object with
`passed: true`. It may also define `prepare()` returning a resource object;
then `validate(resource)` checks that object. Successful resources are retained
on the worker session. Candidate factories can accept `native_resources` to
obtain them without loading device code during Python-patch preparation.

The client drains requests and pauses the scheduler, then every worker:

1. Verifies hashes, manifest identity, and absence of operator-name collisions.
2. Loads the library using `torch.ops.load_library` and verifies registration.
3. Creates its native resources, runs the small validation probe, and synchronizes.
4. Returns a receipt with the same manifest digest. Worker PIDs, weight storage,
   active dispatch and graph state must remain unchanged before resume.

Loading and dispatch are separate commands. `load-native` leaves model execution
on its existing path. Use `switch --candidate ...` to enable a qualified native
candidate and recapture affected graphs. Rollback restores Python dispatch;
libraries and device code remain loaded for the lifetime of the workers.

Preparation failure can resume the unchanged server. A failure after native
mutation keeps it paused. If a worker reports `native_failed`, restart is
required: native registration/device faults cannot be undone by restoring Python.
Repeated loading of an identical successful manifest is a no-op. Altered files
or reusing a name for different content are rejected.

The direct 310P bridge in `artifacts/glm-perf-310p/native-resident-20261005/`
uses runtime binary registration and explicit kernel handles. It bypasses OPP
discovery, retains queued tensor ownership until submission, and does not unload
code that may still be referenced by captured graphs. This interface is for
trusted experiment artifacts, just like the existing Python patch facility.

### Public serving with local controls

The GLM comparison launcher supports `--resident public-local-control`.
Inference stays available on port 8001; `ResidentControlMiddleware` permits
administrative routes only from loopback clients. The middleware does not use
forwarded headers to grant access. Keep the control interface behind this gate;
the upstream development RPC API permits arbitrary code execution.

## Request boundaries and recovery

The client calls `/pause?mode=wait&clear_cache=true`, prepares every rank, clears
worker request records and physical cache contents, applies the candidate,
recaptures when required, clears cache contents again, and resumes. Weight
tensors are excluded from the reset. Target/draft attention handles, events,
workspace references, and bounded audit snapshots are cleared before recapture.
Target and draft receive a fresh shared graph-pool handle: torch-npu retires
the old allocator pool when its final capture is destroyed. Keeping that handle
caused an allocator `use_count` assertion in the first hardware recapture.
The implementation conservatively recaptures both models after a Python change.

Use one experiment controller at a time on a dedicated diagnostic server;
queued application traffic would mix with the comparison. A server already
paused stays paused after `switch`. Preparation failure resumes an originally
running server. Mutation, transport failure during mutation, or capture failure
leaves it paused, with the error and partial worker state available for diagnosis.
Capture exceptions are returned as per-rank error receipts so the client consumes
all replies before reporting failure; an uncaught collective error can leave
stale replies in the executor queues. Other transport/native failures may still
require a restart.

```bash
python -m tools.glm_perf.resident_harness status
python -m tools.glm_perf.resident_harness switch --mode graph
python -m tools.glm_perf.resident_harness resume
```

`resume` rejects dirty graphs or disagreeing worker generations. If a worker
has exited or native execution failed, restart the diagnostic server.

If older executor errors left positive acknowledgments queued on some ranks,
the client confirms the prepared/applied generation with bounded reply reads.
Only side-effect-free preparation is retried; mutations are issued once, then
confirmed through status. Worker preparation and application errors are returned
as per-rank receipts, and reapplying the same generation is idempotent.

## Verification

CPU regressions cover staged application/restoration, rapid edits, rank
agreement, pause ordering, direct mode selection, capture failure, attention
handle cleanup, and unchanged weight/cache allocations. The opt-in real-weight
gate uses an already loaded server and never starts one:

```bash
python -m pytest --noconftest -q tests/ut/glm_perf/test_resident_control.py \
  tests/ut/glm_perf/test_resident_harness.py tests/ut/glm_perf/test_resident_worker.py
python -m pytest -sv tests/e2e/nightly/310p/single_node/resident \
  --resident-glm-url http://127.0.0.1:8001
```

The real-weight gate switches all four modes, sends requests, applies the
reference candidate, recaptures, restores baseline, and checks rank/PID/weight
storage identities throughout, including GLM's unregistered packed expert banks
and per-expert views. Fingerprints cover addresses, layouts and devices, not
weight contents. It qualifies resident control, not GLM answer
quality or performance. Existing MTP repetition remains a separate diagnostic.
Hardware qualification subsequently passed on TP4, including an injected
capture failure and recovery with unchanged workers and packed weight storage.
Recapture took about 4.3 seconds. See the
[resident hardware study](../../artifacts/glm-perf-310p/mtp-packed-candidate-20261004/resident-13/README.md).
The [fresh live recheck](../../artifacts/glm-perf-310p/resident-recheck-20261005/README.md)
also passed all four modes, a reference Python patch, restoration, and four
concurrent requests: 10/10 bounded arithmetic answers were correct, with unchanged
workers and weight storage. It exposed and verified the acknowledgment fix above.
MTP answer repetition remains unresolved; passing the control gate does not
qualify model quality or serving performance.
The host command excludes the repository's global startup hooks, whose import
currently fails because the installed CPU vLLM lacks
`vllm.third_party.flash_linear_attention`. The resident tests explicitly provide
their own worker/graph stand-ins and run the shipped control and extension code.
