# Qwen W4 grouped prefill batching experiment

The [GLM grouped projection study](../../artifacts/glm-perf-310p/wide-l1-20261004/expert-grouping.md)
compared four 640-token, top-8 calls with one 2,560-token call. The same
20,480 route rows were processed in both arms. Its W3 gate/up median fell
from 292.26 to 91.66 ms, W4 gate/up from 153.88 to 56.49 ms, and W2 down
from 97.88 to 33.96 ms. Those timings cover grouped projection calls only;
routing, gathers, and output reordering were prepared outside the timer. The
outputs were bitwise identical. Concentrated W3 routing improved less
(132.18 to 113.00 ms), so route distribution is material.

Qwen W4 has top-10 routing and a 20,480-route grouped native-INT4 cap. Its
current 2,048-token prefill chunk already fills that cap. Four 512-token
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

## Prepared one-card gate

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

After the NPUs are released, run an isolated one-card gate with the coherent
custom OPP described in the runtime runbook:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.qwen4exp.benchmark_w4_prefill_batching_310 \
  --model /path/to/qwen-checkpoint \
  --output /path/to/new/qwen-w4-batching.json \
  --tokens 2048 --chunks 512 1024 2048
```

If the 2,048-token arm saves meaningful time, profile its two native
projection tasks against the 512-token arm to separate kernel reuse from
dispatch and packing. A later 2,560-token experiment needs a matched OPP
package with validated 25,600-row caps, exact output parity, and a repeated
cold-prefill service A/B. Include sustained temperature and capacity in that
gate. This source-only probe has not used NPU hardware or changed the server.
