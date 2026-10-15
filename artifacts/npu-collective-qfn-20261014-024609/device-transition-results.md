# Device acceptance and next-input integration

The worker bridge is hardware-qualified on four Ascend 310P ranks. It does
**not** remove the engine's synchronous scheduler or complete autonomous
distributed model iteration.

## What ran on hardware

Private captured graphs chain device acceptance, native recurrent-state commit,
and next-input construction. Each rank passed 16 changing replays and 32 bank
comparisons. This proves the component dependency chain, not model continuation.

The live bridge uses actual model state: 36 convolution and 36 recurrent banks.
It admits all possible MTP2 acceptance counts before submission, commits from
device counts, and hands device-generated token IDs to the 310P input hook.
The corrected integration's ten-request validation round checked, per rank:

- 196 acceptance/commit steps;
- 216 bank contents against a host reference;
- 186 next-input batches, including token IDs and computed positions.

Validation included short prompts, the 2,048-token chunk boundary, an 8,192-token
prompt, four concurrent mixed-length requests, and a return to one request.
The text gate is a semantic smoke check, not a model-quality benchmark.

## Serving measurements

Diagnostics were disabled during the following measurements. C1 is the median
of three fixed-budget HTTP trials, including the short prefill and API overhead.
C4 is aggregate HTTP throughput, including prefill and request tails. Neither
column is pure device decode throughput.

| Configuration | C1 HTTP tokens/s | C4 aggregate tokens/s |
| --- | ---: | ---: |
| Device transition enabled | 31.027 | 64.065 |
| Device transition disabled | 31.121 | 65.009 |

There is no demonstrated serving speedup. The device chain is useful groundwork,
but CPU output parsing, engine scheduling, host metadata preparation, and PLE
history/lookup still serialize the next iteration. Generated positions are
checked; the full attention metadata path still uses host preparation.

Receipts are in `device-transition-live-v1/`: `install-v2.json`, the validation
and control receipts, and `decode-benchmark-device-transition-{on,off}-v2.json`
and `concurrency-device-transition-{on,off}-v2.json`.

## Remaining work

The prepared engine-control extension provides a drained update path into the
engine process. It does not turn on asynchronous scheduling. Device PLE history,
lookup ordering, compact-cache ownership across queued steps, termination, and
output retirement must be qualified before removing the synchronous boundary.
Keep the async guards until those dependencies are satisfied.
