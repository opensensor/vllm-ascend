# Qwen TP4 named prefill profile procedure and result

This file and the profiled launcher were first prepared on the host only.
The profile was later captured on all four NPUs; see the
[analysis and PNG figures](named-prefill-trace/README.md). The profiled launcher
is a copy of the selected
2,560-token candidate with one additional `--profiler-config` argument. Its
SHA-256 is `ba56f3cb7c539120227ee5b5897c2c9aaa2fa02e9fdeb7100dc517aa5358bc72`.
The same copy is staged at
`/srv/ai/src/qwen38-prefill-batch256-runtime-20261005/examples/start_qwen38_flash_next_w4_310p_batch2560_profile_prefill.sh`.
Both local and remote `bash -n` and `--show` checks passed before launch.
After capture the Qwen service was stopped and the NPUs were released.

The capture launched this copy on port 8001 in a
persistent session. Use the same coherent five-operator package, cache budget,
MTP2, and `[3, 6]` decode graphs as the measured 2,560-token candidate.
The profiler captures four early scheduler iterations of one cold prompt.
`/start_profile` is available only when the server was started with this
configuration. Verify `/v1/models` readiness before capture.

Run the client from the isolated Qwen runtime on the Threadripper:

```bash
cd /srv/ai/src/qwen38-prefill-batch256-runtime-20261005
python -m tools.qwen4exp.profile_runtime capture \
  --base-url http://127.0.0.1:8001 \
  --phase cold-prefill --steps 4 \
  --trace-dir /srv/ai/src/qwen38-prefill-batch256-runtime-20261005/results/named-prefill-20261005 \
  --output results/named-prefill-capture-20261005 --execute -- \
  python -m tools.qwen4exp.benchmark_w4_finalize_service \
  --base-url http://127.0.0.1:8001 --arm cann_v2 \
  --cases 1 --skip-warmup --max-tokens 1 \
  --output results/named-prefill-one-case-20261005.jsonl
```

The capture helper was dry-run checked with these arguments. It starts the
recorder before the unique 23K-token request and stops it even if the client
fails. Capture TTFT is diagnostic only. The raw trace was parsed offline with
`python -m tools.qwen4exp.profile_runtime analyse <trace directory>` and
summarized with `tools.qwen4exp.summarize_trace`.

## Decode findings from the saved capture

The [512-token profile](live-decode-profile/README.md) measured 30.91 output
tok/s with MTP2 and 320/384 accepted draft tokens. The rank-0 Python sample
placed 4.54 of 11.58 sampled seconds in `RejectionSampler.parse_output`, where
the `.cpu().numpy()` readback waits for preceding device work. It cannot be
interpreted as 4.54 seconds of NumPy overhead. QSA's
`copy_current_positions` appeared as a 0.70-second leaf and
`_repair_native_group_indices` as 0.58 seconds across the same sample.
These numbers are inclusive of any device waits at their call sites and come
from one profiled request with sampler errors. The named trace should separate
the preceding kernels from those waits.

Qwen MTP currently rejects async scheduling because PLE n-gram history is
built from the authoritative CPU token table; async scheduling leaves token
placeholders there until output processing. Changing the scheduler alone would
change model behavior. The opt-in W8A8 grouped MTP experts were already tested
on the real service: their five-run c1 median was 29.74 versus 29.86 tok/s for
the routed backend, while c4 aggregate median improved 59.52 to 60.88 tok/s.
See [the prior grouped-MTP result](../qwen-next-gains-20260930/README.md).
The next decode experiment should first measure named verification and draft
costs, then compare MTP2/MTP3 by accepted tokens per millisecond under the
two-size graph budget. A position-copy change needs changing-input MRoPE
replay tests because its host causal positions may differ from the first RoPE
axis.
