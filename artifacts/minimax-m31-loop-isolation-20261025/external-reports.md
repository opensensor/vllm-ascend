# M3.1 Flash reasoning-loop reports

Reviewed October 25, 2026. These are public observations and documentation,
not reproductions on our Ascend service. The [local investigation](README.md)
contains the hardware tests and numerical measurements.

The [deeper evidence audit](../../reports/m31_loop_evidence_audit.md) adds
request-forwarding defects, active intervention in Atomic Agent's harness,
quantization receipt definitions and a source-preserving expert format. In
particular, benchmark completion does not establish unaided model recovery.

## Findings

Similar overthinking and repeated tool inspection are reported on hosted
MiniMax paths that do not use our custom W4 conversion. This weakens attribution
to our conversion alone. Hosted backend precision is unknown; these reports
are not an unquantized control and do not establish a shared root cause.

| Source | Observation | Evidence limit |
| --- | --- | --- |
| [Hermes / MiniMax subscription users](https://www.reddit.com/r/MiniMax_AI/comments/1xed4rt/minimax_hermes_unendliches_denken/) | A simple question about memory files became prolonged session inspection and memory auditing. Users describe excessive thinking on small tasks. | Firsthand anecdotes; exact model, request settings and traces are incomplete. |
| [M3.1 Flash reasons itself to death](https://www.reddit.com/r/opencode/comments/1xaq8wv/minimax_m31_flash_reasons_itself_to_death/) | A user reports 18 minutes of reasoning and two subagents on a simple question through Hermes/T3. Another commenter reports good subscription results. | Mixed experiences, no controlled provider comparison. |
| [Atomic Agent's own low/max experiment](https://atomicagent.io/blog/minimax-m3-1-flash/) | One Python task, three runs per effort through OpenRouter: repeated shell/test commands triggered its loop detector in all three max runs and none of the low runs. All six runs eventually passed the task. | Small, self-reported experiment from the harness vendor; not independently reproduced. It shows temporary repetition, not failure to terminate. |
| [Experimental VQ quantization author's model card](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF) | Warns about reasoning loops and degraded text in an approximately 2-bit routed-expert conversion, including possible long greedy reasoning loops. | Different quantization and runtime. Long free-generation/tool quality is explicitly unmeasured; no matched reproduction of our failure. |

The closest qualitative match is the Hermes report: an informational task
turns into more inspection instead of an answer. Atomic Agent supplies a
specific effort-dependent repetition observation, but not evidence that low
effort fixes our conversion.

## Integration details that matter

[MiniMax's official invocation guide](https://platform.minimax.io/docs/guides/text-generation)
documents five effort levels and a hosted API default of `max`. The local
checkpoint template only emits an effort tag when `reasoning_effort` or `effort`
is supplied. An omitted local tag must not be described as a verified hosted
`max` setting. Our earlier low-effort continuation still requested README/git
inspection, so lowering effort alone is not an established fix.

[OpenClaw issue 160037](https://github.com/openclaw/openclaw/issues/160037)
documents a Preview integration that could not forward the five effort levels
and consequently used the provider default. It is closed and linked to a fix.
This is evidence to inspect the rendered request, not evidence of the same
problem in Kilo.

[Official checkpoint discussion 6](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/discussions/6)
reports vLLM streaming reasoning leaking into content, while non-streaming
separation works, with source revision `54e4e69` and an unchanged template.
That is a distinct parser symptom. Our captured Kilo history includes all
completed tool turns/results and reasoning; our replay produces repeated tool
calls even with automatic tool parsing disabled. This report does not explain
our observed generation pattern.

## Assessment and next discriminating comparison

The measured 14.77–20.26% complete-expert normalized RMS error remains a serious
source-fidelity concern. It covers four selected experts at one layer and one
activation, not a 15–20% model accuracy loss. Turning off INT8 compute still
uses the lossy W4 stored weights, so that experiment cannot clear the conversion.

Stale prefix reuse is substantially weakened by the full-history cold/warm
comparison: reasoning, text and tool arguments matched with zero versus 29,568
cached prompt tokens. This does not rule out every cache defect or explain the
original onset. Hosted reports likewise do not prove our implementation sound.

The next useful experiment is a fresh, bounded multi-turn task, with identical
tool results, prompt/template and sampling, comparing explicit low effort with
the current default. Record rendered effort, repeated calls and whether the
model finishes. Replaying only the late, already repetitive history is not a
clean test of how the loop starts. A matched source-preserving run remains
necessary to attribute excess looping to W4 conversion. Until then, model/agent
behavior and quantization degradation are plausible interacting contributors.

This research made no serving or sampling changes and ran no additional NPU
inference. The public reports do not establish a fix.
