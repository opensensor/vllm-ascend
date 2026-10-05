# Qwen resident experiment startup

The user requested the next server startup while the isolated probes were
finishing. The selected batch-2,560 runtime started in tmux session
`qwen38_resident_20261005`, on port 8001, as
`qwen38-w4-batch2560-cann-finalize-candidate`. The target, draft, expert backend,
cache budget, OPP stack, and two decode graph sizes retain the selected launch
configuration. No experimental math is enabled at startup.

The API became ready at 10:58:54 on 2026-10-05. Rank load times were
161.58/208.92/188.64/238.38 seconds. The engine reported 1,068,936 cache tokens,
4.08x maximum concurrency at 262,144 tokens, and successful capture of both
decode graphs. All four resident status RPCs returned graph mode, clean graphs,
and the selected baseline Python. A [generation smoke](startup-smoke.json)
answered `17 + 25` with `42`; its 0.54-second request is not a throughput
benchmark. Resident mutation/recapture remains unqualified for Qwen. The server
is left running with the selected math and operator stack.

## Matched performance sanity checks

After the user observed 336.4 prompt tok/s and 25.7–26.8 decode tok/s on
21,769–26,426-token conversational requests, the running command was checked
against the selected measured configuration. Binding SHA-256 matches
`fb859c36a57a46df1bf0c8f1a22264c7bc109b9f87a0dc81de66d081d810dc39`.
The API environment selects the coherent batch-256 package, embedded fallback,
and retained fallback in that order. Worker CPU masks are 0–5, 8–13, 16–21,
and 24–29. No eager-decode fallback warning was observed. No configuration or
math change was made during these checks.

| Matched request | Saved selected run | Current resident run |
| --- | ---: | ---: |
| 23,410-token cold prefill, case 1 | 61.840 s TTFT | 62.392 s TTFT |
| 32-token prompt, 512-token decode | 30.910 tok/s | 31.874 tok/s |

The [prefill replay](matched-prefill-replay.json) has the identical case-1
prompt SHA-256, zero cached tokens, and exactly the same 32-token generated
output as the saved run. Its TTFT is 0.89% longer, consistent with broadly
unchanged performance in this single observation. This is not a controlled
thermal or repeated A/B experiment.

The [decode replay](matched-decode-replay.json) uses the saved request body,
temperature and seed, and the same streaming timing helper after a 32-token
warmup. The warmup output matches exactly. The 512-token response diverges in
wording after character 726, so the main replay is not a fixed-output speed
comparison. Its higher rate does not establish an optimization gain. It does
not show the large regression suggested by comparing the user's long-context
requests against the short-context reference. Long-context decode still needs
its own matched benchmark/profile; a short-context replay cannot qualify it.

The selected prefill profile remains active. Neither performance sanity check
restarted the server, reset its prefix cache, or applied an experimental patch.

The [launcher](start-resident.sh) adds the existing development RPC API and
`tools.qwen4exp.resident_worker.QwenResidentExtension`. This extension reuses
the GLM resident control's pause, source preparation, restoration, cache reset,
and graph recapture. Its direct-dispatch hook has no attention metadata copies
or audit logging. GLM's real-hardware control qualification does not establish
Qwen recapture correctness: qualify Qwen separately before relying on switches.
Access to the development API permits executing candidate Python in workers;
keep this experimental endpoint on a trusted network.

## First prepared candidate: reuse a hyperconnection operand

[`shared_hc_operand.py`](../../../tools/qwen4exp/resident_candidates/shared_hc_operand.py)
reuses the normalized FP16 down-projection input for the later injection
projection. The gated mean still uses the original FP32 normalized values.
When the two projection operand dtypes match, this removes one FP32-to-FP16
activation conversion and halves the normalized tensor retained across the
block. For 2,560 rows and 10,240 channels, that saved tensor shrinks from
100 MiB to 50 MiB per mixer. This is the saved tensor size, not a measured
whole-model peak allocation reduction.

Mixed projection dtypes, disabled combination, and enabled gradients retain
the original saved FP32 tensor. Nine host tests cover exact mixed and combined
outputs, fallback cases, and gradient parity under a simulated native operand
policy. The [standalone benchmark](../../../tools/qwen4exp/benchmark_shared_hc_operand_310.py)
uses the real hyperconnection dimensions and checks exact NPU outputs before
paired eager and small decode graph timings. Its first invocation stopped
before measurement because constructing parameters inside `inference_mode`
removed their version counters, which the existing affine cache requires.
The benchmark now constructs them under `no_grad`. Hardware measurements are
pending; the candidate is inactive in the server.

## Control commands

Run from `/srv/ai/src/qwen38-prefill-batch256-runtime-20261005` on Threadripper:

```bash
python -m tools.glm_perf.resident_harness --help
python -m tools.glm_perf.resident_harness status
python -m tools.glm_perf.resident_harness switch --mode graph \
  --candidate tools/qwen4exp/resident_candidates/shared_hc_operand.py
python -m tools.glm_perf.resident_harness switch --mode graph
```

Use the serving virtualenv and hardware environment. Switching must drain the
server and recapture target and draft graphs before resuming. A controller
failure during mutation deliberately leaves it paused for restoration.
Keep one controller and avoid concurrent external requests during comparisons.

The named sparse-gather operator is qualified in its isolated experiment only;
this startup does not install its supplemental binding or vendor. It cannot be
selected live in this process. Native package replacement still needs a planned
restart. The shared-operand candidate needs only Python replacement.
