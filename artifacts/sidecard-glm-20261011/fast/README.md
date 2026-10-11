# Prefill performance correction

The previous capacity profile logged approximately 25 seconds per 128-token
prefill chunk for a 25,456-token coding prompt. The replacement profile uses
512-token prefill batches, prefix caching, one active sequence and an 8 GiB
cache budget per Ascend rank. CUDA owns the same 72 routed experts per layer;
attention and state remain on Ascend. This does not implement the target
CUDA-attention ownership.

The socket path now transfers contiguous tensors without converting each
element into a Python float. Wire bytes, payload hashes, ownership/identity
checks, finite-value checks and ordered FP32 combination remain unchanged.
The CUDA projection uses signed DP4A dot products while preserving per-group
FP32 arithmetic and FP16 output rounding. Expert weights remain unpacked INT8.

[cuda-projection-raw-gate.json](cuda-projection-raw-gate.json) records 30 exact
selected-expert comparisons with authenticated W4/W3/W2 banks, activation
precisions A4/A8, and 1/8/32/128/256 routes. Every FP32 expert output matched
the scalar implementation bitwise. Device and full Python/copy timings are
reported separately. For layer 33, A8, 256 routes, the scalar device time was
12.94 ms and the optimized raw-tensor device time was 1.49 ms. Full call time
fell from 51.92 ms to 2.64 ms. First-call startup costs remain in the report.
These are single-expert microgates, not a whole-model serving speedup.

The running CUDA partition is deployed at
`/srv/ai/storage/artifacts/sidecard-glm-service-20261011-fast-v5` on
`192.168.53.187`. Its daemon accepts sequential peers so API context/batch
tuning can reuse resident banks. The API parent is PID 1050523; CUDA worker
PID 1027790; Ascend workers 1051734, 1052161, 1052534 and 1052962.
[service-process.json](service-process.json) retains the complete command.
[cuda-ready.json](cuda-ready.json) admits all 3,024 experts in 42 layers.
[service-source-receipt.json](service-source-receipt.json) pins 32 source files;
all matched the corresponding local files when captured. Subsequent repository
lint cleanup replaced the checkpoint planner's standard `re` import with the
project's `regex` dependency; the deployed header reader and numerical paths
were not changed. The actual compiled input is retained
as [cuda-compiled-source.cu](cuda-compiled-source.cu); the current source differs
in comments and formatting only.

## Actual service results

[service-validation.json](service-validation.json) records seven completed
operational API cases, all 42 routed layers and 126 real-input expert shadow
checks. Maximum absolute shadow error was 1.3754e-5, maximum NRMSE 7.6624e-5.
Full validation remains false because greedy repeated completions varied.
This does not establish whole-model quality or repeat stability.

[prefill-benchmark.json](prefill-benchmark.json) records completed requests:

| Input tokens | Cached tokens | Time to first text | Uncached prompt rate |
| --- | --- | --- | --- |
| 1,024, first | 0 | 17.20 s | 59.55 tok/s |
| 1,024, repeat | 640 | 6.66 s | 57.62 tok/s |
| 8,192, first | 640 | 135.24 s | 55.84 tok/s |
| 8,192, repeat | 7,680 | 9.16 s | 55.88 tok/s |

The longer first case reused part of the shorter case; it was not fully cold.
The prompt is synthetic repeated text, not the user's original coding request.
Typical steady 512-token iterations logged approximately nine seconds, versus
the reported old 128-token iterations of approximately 25 seconds. This is
roughly an elevenfold rate improvement, not a controlled identical-prompt
comparison. Cold large prompts still take minutes. Decode gaps were about
0.33–0.39 seconds per token. The prior initial service logged about 0.58 seconds.

The planner advertises **645,120 context tokens** with prefix caching. Kilo's
profile was updated to context 645,120, input 613,120 and output/maxTokens
32,000; its own config parser accepted the update. Reload Kilo to adopt it.
[kilo-output-budget.json](kilo-output-budget.json) records actual short chat
generation with `max_tokens=32000`, answer `4` and a normal stop. This profile's
full context was not prefilled; long-context quality remains unvalidated.
The earlier prefix-disabled capacity experiment admitted 709,120 tokens;
its throughput was unsuitable for coding prompts.

[model-boundary-metrics.json](model-boundary-metrics.json) separates input
transfer/pending NPU work, socket exchange, ordered combination and shadow
timings. Input transfer includes a fence for pending backbone work and is not
a pure PCIe transfer measurement. [service.log](service.log) preserves actual
iteration timings and cache hits.

The CPU suite passed 240 tests. Ruff check/format and targeted spelling and
Markdown checks passed. The scoped suite excludes the repository-wide NPU
conftest, which cannot import the local upstream flash-linear-attention module.
