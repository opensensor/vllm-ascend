# Qwen W4 grouped prefill batching experiment

The [GLM grouped projection study](../../artifacts/glm-perf-310p/wide-l1-20261004/expert-grouping.md)
compared four 640-token, top-8 calls with one 2,560-token call. The same
20,480 route rows were processed in both arms. Its W3 gate/up median fell
from 292.26 to 91.66 ms, W4 gate/up from 153.88 to 56.49 ms, and W2 down
from 97.88 to 33.96 ms. Those timings cover grouped projection calls only;
routing, gathers, and output reordering were prepared outside the timer. The
outputs were bitwise identical. Concentrated W3 routing improved less
(132.18 to 113.00 ms), so route distribution is material.

Qwen W4 has top-10 routing and a 20,480-route grouped native-INT4 cap. The
retained 2,048-token prefill chunk filled that cap. Four 512-token
chunks versus one 2,048-token chunk reproduce GLM's 5,120-versus-20,480
route-call geometry without changing an OPP package. The comparison is
informative, but it cannot transfer GLM's speedup directly: Qwen uses native
W4A8 kernels and approximately 128 local experts per rank, while GLM's
uniform study had 72 local experts and 5,122 local routes in its 20,480-route
call. Qwen's uniform expectation is about 5,120 local routes over more
experts, reducing rows per expert and weight reuse.

The [Qwen runtime runbook](../../docs/source/developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md)
records no cold-prefill gain from a 2,560-token scheduler batch. That test
did not enlarge the native grouped projection: `W4SparseMoE` still splits it
at 2,048 tokens. For a 23,410-token request, a real 2,560-token grouped cap
would reduce projection pairs from 12 to 10 per MoE layer, but requires the
operator, activation pack, and model route caps to grow from 20,480 to 25,600
rows. It also requires a matching scheduler batch and memory/capacity check.

## One-card result and service gate

`benchmark_w4_prefill_batching_310.py` loads one real checkpoint expert bank,
computes one fixed router result from synthetic activations, and compares
512-, 1,024-, and 2,048-token grouped chunks on the same 2,048 tokens. It
checks output parity before timing and alternates the order of timed arms.
The timed scope includes grouped dispatch, both dependent native projections,
activation packing, and finalization. It excludes router evaluation, shared
experts, collectives, and the rest of the model. The output also records local
route counts and expert visits so a speedup can be interpreted against the
actual routing distribution.

Dry-run planning uses the host only:

```bash
python -m tools.qwen4exp.benchmark_w4_prefill_batching_310 --dry-run \
  --tokens 23410 --chunks 512 1024 2048 2560
```

The isolated one-card gate ran on 2026-10-04 with real checkpoint expert
weights and bitwise output comparison. Its 2,048-token input took 59.45 ms
when split at 512 tokens, 81.03 ms at 1,024, and 59.89 ms at 2,048. A native
kernel schedule switch at 8,192 total routes caused the 1,024-token
regression. Moving that switch to 16,384 routes in an isolated OPP package
made 1,024-token chunks 35.7% faster, while the 512- and 2,048-token timings
were unchanged. With that kernel, 1,536-token chunks took 51.54 ms for the
same 2,048-token input, 14.0% faster than the retained 2,048-token path.
All compared outputs and router IDs were bitwise identical. These are
single-layer timings, not model TTFT. The raw data and build details are in
the [experiment record](../../artifacts/qwen38-prefill-batching-20261004/README.md).

The service candidate keeps the scheduler at 2,048 tokens, changes the
model's grouped native chunk to 1,536, and uses the isolated OPP package.
Across three matched 23,410-token cold prompts, median TTFT fell from a
saved 72.864 s to 68.950 s (5.37%). All three prompts were uncached, and
generated-text hashes matched the saved baseline. The cold-prefill service
gate and its cross-run limitation are recorded in the experiment record.

To repeat a one-card gate with the coherent custom OPP described in the
runtime runbook:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.qwen4exp.benchmark_w4_prefill_batching_310 \
  --model /path/to/qwen-checkpoint \
  --output /path/to/new/qwen-w4-batching.json \
  --tokens 2048 --chunks 512 1024 2048
```

The native projection tasks were profiled to separate kernel work from
dispatch and packing. A later 2,560-token experiment still needs a matched
OPP package with validated 25,600-row caps, exact output parity, and a
repeated cold-prefill service A/B. Include sustained temperature and capacity
in that gate.

## Larger native-W4 batch candidate

Source now has an explicit native-INT4
`ascend_expert_quantization.grouped_prefill_chunk_tokens` override bounded by
2,560 tokens and 25,600 top-10 routes. The default remains the measured
1,536-token split. The native pack, experimental SwiGLU pack, and grouped
matmul host/tiler row limits agree at 25,600. The matmul candidate moves its
large-M switch from 128 to 256 average routes per expert. With 128 local
experts, this keeps the existing 32-row schedule through 2,560 top-10 tokens
instead of switching just after 1,638 tokens.

The paired layer benchmark now defaults to 1,536, 1,638, 1,639, 2,048, and
2,560-token chunks and can use `--grouped-finalize cann_v2` to match the
current service candidate. `--route-cap` is the installed package's cap; use
20,480 with an older package and omit the 2,560-token case. For a rebuilt
25,600-route package, first run the pack and matmul capacity tests, then the
real-weight layer benchmark with parity and separate traces. Only after that
gate should the model override be used in a four-rank cold-prefill comparison.
Check operator workspace, four-request cache capacity, generation quality,
and sustained temperature. The grouped CANN finalizer's large-row behavior
also remains to be verified.

The [isolated 310P result](../../artifacts/qwen38-prefill-batch256-20261005/README.md)
passed 25,600-route operator capacity and bitwise one-layer output checks.
For a 23,410-token layer partial, 2,560-token chunks took 420.97 ms versus
457.74 ms with 1,536-token chunks. A clean five-operator coherent OPP package
and isolated launcher then passed a TP4/EP4 service check. Three matched cold
prompts fell from 64.292 to 62.032 seconds mean TTFT, with zero cached tokens.
The service reported 4.08 concurrent 262,144-token requests and captured both
decode graphs. The two stable 32-token decode cases remained about 30 tok/s;
there is no measured decode-speed gain. A sustained thermal, concurrent-load,
and broader quality gate remain before changing the source default.
