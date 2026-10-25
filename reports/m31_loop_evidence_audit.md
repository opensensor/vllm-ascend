# M3.1 reports expose several distinct failures

Research audit, October 25, 2026. Public reports confirm that excessive reasoning
and repeated tool inspection occur outside our Ascend conversion, but **none
establishes our loop's root cause**. Deeper inspection found a concrete effort
forwarding defect, active loop intervention inside a published benchmark, and a
source-preserving expert format that avoids our lossy requantization. These
findings change which comparisons are useful. They do not clear the measured
W4 loss or establish a cache fix. This audit read sources and local diagnostic
records; it made no service changes and ran no inference.

## Selected low effort can still mean hosted defaults

The original “reasons itself to death” reporter explicitly says they selected
low or no reasoning. The visible thread contains no verified remedy or outgoing
request capture. Thus it cannot be classified as a known max-effort run.
([Reporter follow-up](https://www.reddit.com/r/opencode/comments/1xaq8wv/comment/phw2had/))

**Hermes has a concrete mechanism for losing that setting.** Issue 143065 reports
that MiniMax ignored manual thinking budgets, while explicit
`output_config.effort=low` reduced thinking in a small probe. The inspected
adapter maps MiniMax's low/high controls to legacy token budgets, produces no
effort field on that path, and returns empty options for disabled reasoning.
MiniMax's documented hosted default is max. The issue remains open with a fork
fix; the probes are reporter measurements, not our reproduction. No evidence
connects the Reddit users to this exact adapter version or transport.
([Hermes issue](https://github.com/NousResearch/hermes-agent/issues/143065),
[pinned adapter](https://github.com/NousResearch/hermes-agent/blob/09b1348ab8f27effe5fd70f6759400c3d3078133/agent/anthropic_adapter.py),
[official effort contract](https://platform.minimax.io/docs/guides/text-generation))

OpenClaw merged effort forwarding for the Preview model, but a later open report
identifies the released model ID falling back to an older profile without that
wrapper. The initial regression evidence is source inspection, not a wire trace.
([Merged fix](https://github.com/openclaw/openclaw/pull/160089),
[release-ID regression](https://github.com/openclaw/openclaw/issues/177387))

Our Kilo path uses an OpenAI-compatible local server, not Hermes's adapter.
Its default has no explicit effort setting. The earlier diagnostic explicitly
passed `chat_template_kwargs.reasoning_effort=low`; the local template supports
that argument, although no rendered-token receipt was recovered. It still
requested one further README/git inspection. The Hermes defect therefore does
not explain that local test.
([Local response](../artifacts/minimax-m31-loop-isolation-20261025/history-late_low_effort.json))

## Successful benchmark runs do not prove unaided recovery

Atomic Agent reports loop warnings in all three default-effort runs and none of
three low-effort runs, with every run passing its task. Its “max” condition
actually omitted effort and relied on the hosted default. No per-run transcripts
or complete replay package were found.
([Experiment](https://atomicagent.io/blog/minimax-m3-1-flash/))

**The named harness version actively intervenes.** Its tagged source injects
warnings into prompts, can veto repeated calls, and can force a synthetic final
reply when a breaker fires. Several detector types share the generic
no-progress message. Without run logs, we cannot identify which interventions
occurred or whether the model would have stopped unaided. This is evidence about
a model-plus-harness system, not isolated model recovery.
([Tagged agent loop](https://github.com/AtomicBot-ai/atomic-agent/blob/6d286528d67739d6044a61304096bf1904a8aba0/src/agent/agent-loop.ts#L741),
[detector implementation](https://github.com/AtomicBot-ai/atomic-agent/blob/6d286528d67739d6044a61304096bf1904a8aba0/src/agent/loop-detector.ts))

## Quantization warnings lack a matched loop reproduction

The experimental VQ model warns about loops but explicitly leaves long free
generation and tool quality unmeasured. Its published generation samples are
short sanity prompts; no paired source/VQ multi-turn loop reproduction was
found. Its warning is relevant caution, not evidence that a particular numerical
error causes our failure.
([Model card](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF),
[published samples](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/quality/reference_generation_samples.txt))

The numerical comparisons also require separation: **88.229% assistant top-1
agreement** measures VQ against original weights over 12,191 assistant positions;
the PR's **96.15% runtime agreement** compares an engine against a reference
using the same VQ bytes. Runtime agreement cannot clear conversion loss. Neither
metric is comparable to our single-token KL or four-expert output sample.
([Conversion receipt](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/quality/release-final/result.json),
[runtime quality code](https://github.com/Baekpica/ds4-dfm-rs/blob/main/crates/ds4-cli/examples/minimax_quality.rs),
[PR results](https://github.com/Baekpica/ds4-dfm-rs/pull/110))

## Source FP4 can remain compact without our requantization

Unsloth's converter preserves the original nonlinear E2M1 expert codes, E4M3
scales per 16 values, and inverse global scales. Its author reports exact native
expert bytes and 31/31 full-checkpoint spot checks. This differs materially from
our uniform signed INT4 with shared 32-by-32 scales. However, the converter
discards trained activation input scales, so preserving stored weights does
not preserve original W4A4 arithmetic.
([Conversion PR](https://github.com/unslothai/llama.cpp/pull/320),
[pinned converter](https://github.com/unslothai/llama.cpp/blob/822be707b23345b291f3042defa7508aa813cc4b/conversion/minimax.py#L205))

A mathematical inference follows: doubling E2M1 magnitudes yields
`0, 1, 2, 3, 4, 6, 8, 12`, all exactly representable in INT8. A nibble lookup with
halved scales can preserve weight values. **This is not a validated 310P kernel**:
per-16 scaling still needs correct, efficient application, and ordinary
single-scale INT8 GEMM is not equivalent. Memory, accumulation and activation
quantization require separate accounting.

## The next comparison must isolate onset and conversion

Parser repairs and a contested duplicate-compaction report describe different
failures; neither demonstrates a fix for repeated discovery. Our cold/warm
full-history equality substantially weakens stale prefix reuse for the observed
continuation, while FP16 compute still uses the lossy stored W4 values.
([Parser fix](https://github.com/ai-dynamo/dynamo/pull/17197),
[compaction discussion](https://github.com/anomalyco/opencode/issues/57805),
[local controls](../artifacts/minimax-m31-loop-isolation-20261025/README.md))

The useful next test is a fresh bounded tool trajectory with captured rendered
effort and fixed tool results, followed by a matched source-preserving weight
comparison. Record any loop-detector intervention separately from natural
completion. Our 14.77–20.26% sampled expert-output error remains a serious fidelity
concern; the public reports neither prove it caused the loop nor justify accepting
it as harmless. Source-preserving expert decoding now has concrete external code
to guide that investigation.
