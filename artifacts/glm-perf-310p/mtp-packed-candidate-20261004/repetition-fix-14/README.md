# MTP repetition: KDA gate lifetime fix

Date: 2026-10-05. Shared main checkout; TP4 Ascend 310P, selective packed W3.

## Root cause and correction

The remaining repetition came from `_safe_gate_for_layer` lazily saving
weight-derived tensors on its first decode call. That call can occur during
graph capture. The graph captured first owns the operations producing these
tensors; another batch-size graph consumes their storage without producing
its contents. Bypassing graph replay afterward does not repair the saved
operands, which explains the earlier failures in direct execution too.

On an actual 25-token subtraction prompt, layer 0's cached calculation returned
**-2.5 for every gate**. Computing from the loaded weights instead gave a range
of **-4.93324 to -0.00000613**. See `gate-evidence.json`. This wrongly shortened
the recurrent memory carried into decode.

The initial prefill comparison reused the defective cached gate in its CPU
reference and therefore appeared to implicate the prefill kernel. Correcting
that reference made native prefill agree: maximum absolute error **3.41e-6**
for output and **2.59e-4** for final FP32 state on the saved layer-0 tensors.
The saved input tensors remain under the corresponding remote experiment
folder, `prefill-rank[0-3].pt`.

The production fix:

- Prepare the exponential and FP32 bias after checkpoint loading, outside
  every graph capture; refresh them on every weight load.
- Reuse those operands during decode, preserving the original optimization.
- If preparation is absent or a parameter object was replaced, compute the
  gate without saving any intermediate from forward. Each captured graph then
  contains its own producers.
- Ignore the legacy cache, permitting resident recovery from the bad state.

## Resident validation

All resident switches retained worker PIDs and registered/unregistered packed
weight storage fingerprints. No checkpoint weights or quantization changed.

- The uncached diagnostic fixed all three formerly looping cases, both for
  subtraction in direct mode and the complete three-case set with full graphs:
  `45`, `63`, and `BLUE-ORCHID-7319`, each terminating normally.
- The production prepared operands were installed in all **34 KDA layers**
  outside capture, then full graphs `[2, 8]` were recaptured.
- Strict gate: **17/20**, all 20 requests stopped normally. Remaining strict
  misses: reverse returned `pial`; first-color returned `Red` instead of `red`;
  slice returned the correct substring in backticks instead of bare `lan`.
- **4/4 concurrent bounded generation requests passed** and the tool-call
  request emitted the correct `get_order_status` call for `A-1042`.
- Full records: `qualified_gate.jsonl` and `qualified_gate.summary.json`.
  Aggregate `passed` is false because of the three strict quality misses.
- **12 targeted CPU tests passed** using `--noconftest`. The normal repository
  conftest cannot import the local upstream package's missing
  `vllm.third_party.flash_linear_attention`; the full suite is not qualified.
- **2 NPU graph tests passed**, covering both prepared and fallback operands.
  They capture sizes 8 then 2 and replay size 2 first, before the first graph
  has ever replayed, then change inputs and alternate sizes.
- Ruff checks and formatting pass for the changed production and test files.

The standalone hardware test initially omitted the serving runtime's
`jit_compile=False` setting and attempted uncapturable ACL operations. The
corrected test sets the same mode as `NPUWorker310`; the final log is
`gate_graph_test.log`.

## Resident control limitation observed during diagnosis

An early candidate factory used `inspect.getsource` on an already patched
function and failed during preparation. Preparation failures still escape the
worker RPC, unlike capture failures, leaving rank reply queues out of alignment.
`lagged_rpc_probe.py` recovered this diagnostic process using paused mutation
phases and repeated status barriers. It is an experiment-specific recovery
controller, not a general fix for that separate harness defect. Factories
subsequently read original functions from source via AST. The final serving
restart clears this control state and does not enable development RPCs.

## Serving configuration

`serve-mtp-fixed.sh` and `launch-fixed.py` start normal serving on port **8001**,
MTP1, TP4, full decode graphs `[2, 8]`, scheduler batch 640, four sequences,
prefix caching enabled, aligned Mamba state. The matched OPP and torch binding
include the preceding physical page-stride fixes.

The configured context remains the **32,768-token diagnostic cap**. This run
does not qualify 192K or maximum-context capacity. Short-answer timing is not a
throughput benchmark.

## Clean startup qualification

The ordinary server started successfully with the production load-time fix,
without resident patches, audit wrappers or development RPCs. Full graph
capture completed in four seconds (0.56 GiB). Its first request used the small
single-request graph, before any concurrent generation.

The complete repeat qualification again scored **17/20**, with all 20 quality
requests stopping normally. The same three strict case IDs missed (reverse
returned `pail` on this run). **4/4 concurrent requests and the tool call passed.**
See `startup_gate.jsonl` and `startup_gate.summary.json`.

Server left running at **192.168.53.187:8001**; API PID **3715824**, engine
**3718150**, workers **3718575, 3718970, 3719485, 3719852**. Worker affinity
was reapplied with disjoint CPU groups and SMT siblings; `affinity-fixed.json`
records the masks. `serve-fixed.log` contains startup and request timing.

A separate coding completion produced a correct `stable_unique` function and
three assertion examples, including empty input: **98 tokens, normal stop**.
The reviewed code compiled and all three assertions passed locally; see
`coding-response.json` and `coding-verification.json`.
