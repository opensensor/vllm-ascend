# Live Qwen decode profile at NPU handoff

The Qwen TP4/EP4 server was sampled on 2026-10-05 before the user handed the
NPUs to another agent. The server used the 2,560-token expert chunk, built-in
SwiGLU, CANN finalizer, MTP2, and decode graphs `[3, 6]`. A 32-token warmup
preceded one serial 512-token request. The [request record](decode-512.jsonl)
reports a 32-token prompt, 0.327 s TTFT, **30.91 output tok/s** over 16.532 s
of decode, and 320 accepted of 384 MTP draft tokens. This is a profiled,
single-request observation, not an A/B decode-speed claim.

The [rank 0 Python sample](worker0.speedscope.json) contains 579 main-thread
samples. Of its 11.58 s of represented stack time, 4.54 s (39%) ended in
`RejectionSampler.parse_output`, which calls `output_token_ids.cpu().numpy()`.
That frame includes waiting for preceding NPU work; it does not establish
4.54 s of NumPy computation. Graph replay appeared in 2.72 s (23.5%) of
inclusive samples, and draft proposal in 1.68 s (14.5%). Inclusive stacks
overlap and must not be added. The [engine sample](engine.speedscope.json)
spent nearly all represented time waiting for worker responses through
`shm_broadcast.dequeue`. The worker and engine samplers reported 69 and 280
sampling errors, respectively, so these shares are directional.

CANN system profiling captured all four NPUs without restarting the service.
The selected exported counters are under [system](system/); complete raw data
remain at
`/srv/ai/src/qwen38-prefill-batch256-runtime-20261005/results/live-profile-20261005/decode/system`.
System mode lacks operator names, and its DDR CSV reports implausible units;
those counters are not used for a bandwidth claim. The service was stopped
at the user's handoff, and `npu-smi` then showed no running processes.

The next decode experiment should isolate the synchronous MTP output readback
from the verification graph's completion time, then compare MTP2 and MTP3 by
accepted tokens per millisecond under the two-graph capture budget. A new
server launch with `--profiler-config` is needed for named NPU operator traces.
The planned live 23K prefill capture did not run before the handoff; the
existing [one-layer batching result](../README.md) and older operator traces
remain the available prefill evidence.
