# GLM sidecard service

**Running:** `http://192.168.53.187:8001/v1`, model `glm-sidecard`.
The current prefix-enabled profile advertises 645,120 context tokens and uses
512-token prefill batches. Measured uncached prefill is approximately 56
tokens/s; an 8,192-token repeat reused 7,680 tokens and reached first text in
9.16 seconds. Cold large prompts still take minutes. See the
[performance evidence](../../artifacts/sidecard-glm-20261011/fast/README.md).
The initial 8,192-token profile passed seven operational API cases. All 42 routed layers executed CUDA
requests and passed 126 real-input expert shadow checks. Greedy repeated
completions varied; the full validation flag remains false. The cause of that
variation is not isolated. Model quality and determinism remain unqualified.

The first serving profile uses the qualified GLM-5.3-Flash native mixed
W2/W3/W4 checkpoint, its Ascend backbone and four Ascend TP ranks. CUDA owns
experts 0–71 in every main-model routed layer, 3–44. Ascend ranks 1–3 own
72–143, 144–215 and 216–287. Rank 0 keeps its attention, shared MLP and other
backbone work on Ascend. Both physical Ascend cards participate.

This is an initial joined service. CUDA attention/state, the planned
72-CUDA/54-per-chip ownership, optimized packed CUDA kernels, graph capture
and MTP are follow-up work. The CUDA correctness kernel stores unpacked INT8
weights, preserving the source integer codes and FP16 scales. The original
rank-zero NPU expert banks remain resident for shadow checks. The measured
serving improvement does not establish whole-model quality or completion of
the intended CUDA-attention design.

## API

The service name is `glm-sidecard`. The deployed host is `192.168.53.187`,
port `8001`. The replacement performance profile is staged under
`/srv/ai/storage/artifacts/sidecard-glm-service-20261011-fast-v5` on that host.
The original `sidecard-glm-service-20261011-v1` process receipts are historical.
The [local evidence directory](../../artifacts/sidecard-glm-20261011/)
contains the collected validation report and process receipts.

```bash
curl http://192.168.53.187:8001/v1/models
curl http://192.168.53.187:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-sidecard","messages":[{"role":"user","content":"Hello!"}],"reasoning_effort":"low","max_tokens":64,"temperature":0}'
```

## Execution and readiness

An independent CUDA process loads all 3,024 assigned experts before the
Ascend server is switched. Metadata is parsed once, every selected code/scale
tensor is checked, and each selected shard's payload SHA256 is authenticated.
The loader stages one packed CPU layer at a time and admits the actual
unpacked CUDA bytes, scratch space, an 82 GiB allocation budget and at least
4 GiB of remaining device memory. Preparation uses eight CPU threads in that
isolated process; it does not change the server's thread settings.

A private Unix socket connects the NPU rank-zero model worker to the resident
CUDA process. Socket credentials, configuration/checkpoint/placement hashes,
complete ownership and the existing binary packet contract are checked.
Requests retain global expert IDs and original route slots. Native EP padding
with zero weights is discarded. Route rows are compacted into at most 32 input
tokens and 256 routes per exchange. Larger prefill passes are split into such
chunks.

The CUDA worker returns raw FP16 expert results encoded as FP32 rows. The
caller multiplies and adds in original slot order in FP32. Existing GLM
composition applies the routed scale, shared addition and Ascend-only HCCL
reduction. CUDA and NPU contexts remain in separate interpreters. Host buffers
are pageable and exchanges are synchronous.

`service_prepare --tensor-transport` uses contiguous tensor buffers throughout
the socket path, preserving the original wire bytes and validation contract.
It avoids constructing Python floats for every hidden/output element. The
exact CUDA projection uses eight lanes and signed DP4A integer dot products,
with the original per-group FP32 multiply/add and FP16 rounding order. Thirty
selected W4/W3/W2 cases passed bitwise comparisons against the scalar kernel;
these microgates do not establish end-to-end serving speedup.

The loader hook is opt-in through `sidecard_worker.SidecardWorkerExtension`.
It patches only the native loader completion boundary and replaces composed
expert methods. No model-runner or checkpoint file is modified. A future
upstream loader completion callback should replace this experiment's hook.
Eager TP4/PP1, target-only execution is required. Internal memory-profiling
passes use the original native method; real generation never falls back on a
CUDA worker error.

The first three real calls at each routed layer are compared with the original
Ascend method using `rtol=0.002, atol=0.002`. `model-boundary-metrics.json`
records real calls, real CUDA routes, shadow checks and maximum errors.
Service validation requires real CUDA work and all three checks in every
layer; startup and `/v1/models` alone are insufficient.

## Context capacity

The initial context was a conservative launcher setting, not the maximum
cache capacity. `service_launch` accepts `--max-model-len -1` to use the
installed vLLM planner's automatic fit across all four Ascend ranks, bounded
by the checkpoint's 1,048,576-position limit. `--kv-cache-memory-bytes`
sets a per-rank cache budget; it must be measured against actual allocations
and operator workspace headroom on the smallest chip.

`--capacity-first` disables prefix caching, selects compact live KDA state
with `--mamba-cache-mode none`, and limits execution to one active sequence.
Concurrent API requests queue. This removes history snapshot/page padding
overhead; it preserves the full attention history and ordinary generation.
CUDA still holds routed experts and Ascend owns attention caches. Free CUDA
VRAM is not automatically available to an NPU cache. The original CUDA
attention and 54-experts-per-chip design remains separate work.

For repeated coding prompts, select `--enable-prefix-caching --max-num-seqs 1`
instead of `--capacity-first`. The former enables aligned KDA prefix state.
`--max-num-batched-tokens 512` increases prefill batching; the launcher permits
up to the inherited 640-token qualified workspace bound. Cache budget and
batch size must leave sufficient operator workspace. The 9 GiB-per-rank
capacity experiment allocated successfully but failed its first generation
with an attention workspace OOM; 8 GiB completed generation. Advertised
capacity alone is insufficient evidence of an operational profile.

Measure completed requests and prefix reuse separately:

```bash
python3 -m tools.sidecard.prefill_benchmark \
  --lengths 1024 8192 --timeout 600 \
  --output /srv/ai/storage/artifacts/sidecard-glm-service-next/prefill-benchmark.json
```

Time to first text includes API overhead, actual prefill and the first decode
step. The report records actual cached usage and completed output; repeated
synthetic text does not qualify coding accuracy or long-context quality.

Run `context_validate` after ordinary service validation. It checks that a
32,000-token output budget can complete a chat request, admits a stream with
input plus output at the advertised context boundary, then disconnects it,
and confirms that an output budget beyond the context is rejected. The
saved report explicitly distinguishes request admission from completed
prefill and long-context quality. It does not evaluate a full-capacity prompt.

```bash
python3 -m tools.sidecard.context_validate \
  --base-url http://127.0.0.1:8001 \
  --output /srv/ai/storage/artifacts/sidecard-glm-service-next/context-validation.json
```

Kilo's custom model `limit.context` must match `/v1/models`; reserve output
inside that same total. For an output budget of 32,000, use
`limit.input = limit.context - 32000`, `limit.output = 32000` and
`options.maxTokens = 32000`. Reload Kilo after changing its profile.

## Reproduce a deployment

Run on the hardware host with the existing qualified interpreters and
compiled CUDA oracle library. Use a fresh deployment directory. When
retiring the joined service, stop its API parent with SIGTERM and wait for
the model workers to exit. A reconnectable CUDA worker remains resident;
stop it separately with SIGINT. Confirm the old CUDA bank has exited and
the old NPU allocations have been released before preparing another bank.
Two complete CUDA banks exceed available VRAM. SIGINT unwinds CUDA
allocations without a device reset.

The following example reuses the captured qualified source and runtime
environment. For subsequent restarts, use the last service's process receipt
and source directory instead. The source tree must contain the current
`tools/sidecard` files.

```bash
cd /srv/ai/storage/artifacts/sidecard-glm-service-20261011-fast-v5/source
python3 -m tools.sidecard.service_prepare \
  --qualified-source /srv/ai/storage/artifacts/sidecard-glm-service-20261011-fast-v5/source \
  --previous-process /srv/ai/storage/artifacts/sidecard-glm-service-20261011-fast-v5/service-process.json \
  --runtime-environment /srv/ai/storage/artifacts/sidecard-glm-service-20261011-fast-v5/runtime-environment.json \
  --checkpoint /srv/ai/storage/models/GLM-5.3-Flash-two-card-native-candidate \
  --library /srv/ai/storage/artifacts/sidecard-glm-service-20261011-context-v4/cuda-dp4a.so \
  --cuda-python /srv/ai/venvs/laya-host-only-20261004/bin/python \
  --tensor-transport --max-sessions 16 \
  --output /srv/ai/storage/artifacts/sidecard-glm-service-next
```

Wait for `cuda-ready.json`; inspect `cuda-worker.log` on failure. Readiness
requires all 42 layers, all 3,024 assigned experts, upload completion and
unchanged checkpoint metadata. Then launch from the new source tree:

```bash
cd /srv/ai/storage/artifacts/sidecard-glm-service-next/source
python3 -m tools.sidecard.service_launch \
  --root /srv/ai/storage/artifacts/sidecard-glm-service-next \
  --host 0.0.0.0 --port 8001 \
  --max-model-len -1 --kv-cache-memory-bytes 8589934592 \
  --enable-prefix-caching --max-num-seqs 1 --max-num-batched-tokens 512
```

To tune context without reloading expert banks, prepare CUDA with
`service_prepare --max-sessions 16`. The daemon accepts sequential peers,
keeps banks after clean EOF, and closes on malformed packets or backend
failure. An idle listener expires after one hour; readiness does not
guarantee that all 16 sessions remain unused. Stage a fresh API directory:

```bash
python3 -m tools.sidecard.service_restart \
  --previous-root /srv/ai/storage/artifacts/sidecard-glm-service-previous \
  --output /srv/ai/storage/artifacts/sidecard-glm-service-next
```

Then run `service_launch` from the newly staged source with the previous API
PID as `--replace-pid`, plus `--max-model-len -1`, a measured
`--kv-cache-memory-bytes` budget, and optionally `--capacity-first`.
Staging verifies bank/checkpoint identity and preserves old receipts. The
launcher performs its normal import preflight and receipt-checked SIGTERM
before the new worker connects. A reconnectable daemon must be stopped
separately with SIGINT when retiring its bank.

When replacing a still-running qualified Ascend-only server, add
`--replace-pid` with the PID from its saved process receipt. The launcher
checks that PID's complete command, runs an import preflight, sends SIGTERM
and waits for exit. It refuses an occupied port, an existing service receipt,
or mismatched CUDA readiness. It never resets a device or sends SIGKILL.

After the API starts, validate actual requests on the host:

```bash
python3 -m tools.sidecard.service_validate \
  --base-url http://127.0.0.1:8001 \
  --metrics-path /srv/ai/storage/artifacts/sidecard-glm-service-next/model-boundary-metrics.json \
  --output /srv/ai/storage/artifacts/sidecard-glm-service-next/service-validation.json \
  --prefill-budget 512
```

Validation covers completions, chat, deterministic repeat, chunked prefill,
two concurrent requests and streaming, then checks all 42 routed layers.
The default command fails on changed greedy repetitions. Add
`--operational-only` to complete the API checks while retaining separate
`operational_passed` and full `passed` results; it does not turn a repeat
failure into a full pass. The report also requires additional CUDA routes
in every layer between its before/after API snapshots.
Check `service.log`, `cuda-worker.log` and the saved process receipts if a
worker exits. The initial service is a foreground vLLM process launched in its
own session, with persistent log files; no systemd restart policy is installed.
