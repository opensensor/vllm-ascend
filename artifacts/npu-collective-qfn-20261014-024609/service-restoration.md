# Stable QFN service and resident optimization

The candidate API wrapper keeps the public identity
`vllm-310p-w4/qwen38-flash-next` in Kilo, API model `qwen38-flash-next`, and
`http://192.168.53.187:8001/v1`. No Kilo configuration change is required.
The service is running from `resident-live-v3`, with corrected cold QSA variants
active after metadata-isolation validation. The failed request-time replacement
remains disabled. See the [failure, correction and validation record](qsa-resident-v1/README.md).
The [earlier hardware handoff](resident-live-v2/README.md) records the initial
successful resident reloads; its process identity is historical.
Do not stop this server for further optimization: use resident transactions.

## Inspected qualified sources

- Remote `~/start_qwen38_flashnext_mtp_graph.sh` and local
  `examples/start_qwen38_flash_next_w4_310p.sh` retain the same qualified model,
  operator package precedence, TP4/EP4 and cache geometry. Their default served
  alias is a research name, so the candidate explicitly uses the stable alias.
- `qfn-pinned-position-runtime` is the selected custom research baseline.
  `qfn-custom-unobserved-config-v2.json` contains the admitted library hashes,
  device groups, fixed collective geometries and unobserved graph policy.
- Existing resident APIs implement pause, prepare, apply, capture, reset and
  resume without unloading model parameters. The parent supplies the combined
  `tools.sidecard.qfn_resident_worker.QfnResidentWorkerExtension` to manage
  native graph leases correctly during those operations.
- No active Qwen/vLLM systemd service definition was discovered during the
  audit. The retained launcher uses a persistent shell context and explicit
  worker affinity rather than an installed Qwen unit.

## Launch once

Stage a complete tools overlay that extends the qualified runtime's `tools`
package path, including `qfn_resident_server.py`, the parent's resident worker,
all existing custom collective tools, and the requested candidate modules.
The overlay's `tools/__init__.py` must use `pkgutil.extend_path`; do not mix
new local model files into the qualified runtime. The launcher changes into
the overlay root so an older runtime `tools` package cannot shadow it.

The [candidate launcher](start-qfn-resident-candidate.sh) accepts:

```bash
bash start-qfn-resident-candidate.sh \
  --tools-root /srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-source-v1 \
  --run-root /srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-service-v1 \
  --show
```

Omit `--show` only for the coordinated one-time launch. The run directory must
be new and absolute; the script creates it with mode 0700 and assigns a fresh
collective owner directory underneath it. It does not reuse or delete old
owners. It defaults to the pinned-position runtime; `--runtime` selects a
separately qualified complete snapshot. Optimization afterward uses resident
changes, rather than stopping the endpoint and loading 169 GiB again.

Preserved service settings:

| Setting | Value |
| --- | --- |
| Model | Qwen3.8-Flash-Next-W4A16-G128-300i |
| Execution policy | Existing native W4A8 target, FP16 storage, MTP2 |
| Parallelism | TP4/EP4 on NPU devices 0,1,2,3 |
| Maximum context | 262,144 tokens |
| Maximum concurrent requests | 4 |
| Prefill chunk budget | 2,048 tokens |
| Logical KV reservation | 88,673,894,400 bytes |
| Compilation | FULL_DECODE_ONLY, `[3, 6]` |
| Collectives | Custom fixed SUM, graph replay tracking disabled |
| Tools | Automatic tool choice, `qwen3_xml` parser |
| Reasoning | `qwen3` parser |
| Public endpoint | `qwen38-flash-next`, port 8001 |

The launcher retains coherent → packaged → retained OPP precedence, CANN
Python paths, TASK_QUEUE_ENABLE=1 and the existing timeout/cache environment.
It enables the existing upstream development routes only behind the new
administrative authentication middleware. Attach the previously qualified
six-core worker affinity and thermal monitor during startup; this minimal
launcher does not create additional PID-discovery scripts. Verify all four
rank load/capture messages, cache capacity, API identity, thinking output and
a tool call before reporting the endpoint ready.

## Authenticated local administration

[The API wrapper](../../tools/sidecard/qfn_resident_server.py) generates
`RUN/control.token` with mode 0600, inside the 0700 run directory. It never
prints the token. Every non-public HTTP route requires both a real loopback
peer and the matching `x-qfn-resident-token` header. Public inference retains
its existing access behavior. Uvicorn proxy headers are forcibly disabled:
`X-Forwarded-For: 127.0.0.1` cannot grant a remote client administrative access.

Run the resident client on the remote host, through SSH if necessary:

```python
from tools.sidecard.qfn_resident_server import QfnResidentClient

client = QfnResidentClient(
    "http://127.0.0.1:8001",
    "/absolute/run/directory/control.token",
)
status = client.rpc("resident_status")
```

This client inherits the existing `ResidentClient.switch(control)` transaction,
including all-rank acknowledgments and unchanged PID/weight-storage checks.
Use maintenance admission control around a graph-changing switch. The existing
transaction pauses with `mode=wait&clear_cache=true`, resets request state,
prepares/applies the source, recaptures graphs, resets dummy-capture state and
resumes only after all ranks agree. Weights stay resident. Scheduler prefixes,
KV/Mamba request state and graph caches are invalidated as required because
capture dummy forwards mutate those buffers. A same-code execution-mode switch
can use the existing cache-preserving transaction instead.

The separate cold numerical gate is available through authenticated
`qfn_draft_real_weight_gate` worker RPC while inference is paused and drained.
Candidate code preparation, ABI/weight gates, pause/reset/capture details and
native lease accounting belong to the worker extension. Do not call a raw
apply or capture RPC during generation.

## Actual shutdown ordering

Qualified upstream `serve_http.handle_shutdown` calls synchronous
`engine_client.shutdown()` from an executor thread **before** its engine
context exits. A context-manager-only teardown hook would therefore be too
late for peer imports and graph-owned native storage.

The wrapper intercepts this cold-bound engine method. On an actual shutdown,
it first pauses/drains generation, invokes all-worker `npu_collective_close`,
requires four distinct closed receipts, and writes `RUN/collective-close.json`.
Only then does it call ordinary engine shutdown. Context exit handles other
normal exits in the same order. Errors prevent the wrapper from calling
ordinary shutdown; do not force cleanup of retained owners after an ambiguous
retirement failure. Startup failures before engine construction and external
process kills are outside this successfully-constructed-engine guard.

The wrapper directly invokes the single-process API entry point, preserving
EngineCore/TP subprocesses. It bypasses CLI frontend spawning so authentication
and shutdown hooks cannot disappear into a newly spawned API process.
These hooks do not run per token and do not add graph replay observers.

## Validation

Thirty-three offline tests passed: token permissions and symlink rejection,
loopback authentication, forwarded-header rejection, duplicate-token headers,
public inference, maintenance admission, four-rank close receipts, shutdown
ordering, close-error behavior, in-process entry invocation, and refusal to
send credentials to a non-loopback origin. Scoped pre-commit checks passed.
Parent-run hardware validation subsequently verified two live resident switches,
unchanged worker/weight-storage identities, collective lease accounting, public
streaming chat and automatic tool calls. The latest focused lifecycle, API and
reset-barrier suite passed 47 tests. Full Kilo agent sessions and broad model
quality evaluation remain separate from these API checks.

## Initial startup correction

The first coordinated launch loaded the model and reached API construction,
but the qualified vLLM app builder had already materialized Starlette's
middleware stack. Calling `add_middleware` afterward failed before HTTP became
ready. The wrapper now wraps that existing stack during cold construction,
preserving the FastAPI object and state. A regression test covers that exact
rejection and confirms public access plus administrative denial afterward.

The failed launch wrote `resident-live-v1/collective-close.json`: all four
ranks reported `closed` before ordinary engine shutdown. This hardware receipt
validates the normal-exception close ordering. The corrected startup succeeded;
the separate `resident-live-v2` receipts validate live HTTP reload. That service
is left running for further experiments.
