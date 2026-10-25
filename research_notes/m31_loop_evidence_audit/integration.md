# M3.1 integration failures and relevance

## Can a low reasoning setting fail to reach MiniMax?

### Takeaway

Yes. The most relevant new evidence is an open Hermes defect: its Anthropic
adapter sends a manual token budget instead of MiniMax's effort field. A separate
OpenClaw regression loses effort control when switching from the Preview ID to
the released ID. Neither establishes that the Reddit reporters used these paths.

### Cited Findings

- Hermes issue 143065, opened October 22, reports probes on `minimax-cn` with
  `MiniMax-M3.1-Flash`: budgets 4000/16000/32000 yielded 67/86/72 thinking tokens.
  Explicit `output_config.effort=low` reduced thinking by roughly 60% in a small
  experiment. The issue remains open; a fork fix is linked, with four focused
  tests and 199 neighboring tests reported passing. This is reporter evidence,
  not our live reproduction. The claim that `enabled` is the only accepted
  thinking type conflicts with the successful adaptive probes in OpenClaw's PR;
  do not repeat that claim as fact. [Issue and follow-up](https://github.com/NousResearch/hermes-agent/issues/143065)
- Independently inspected Hermes main at `09b1348ab8f27effe5fd70f6759400c3d3078133`:
  `_supports_adaptive_thinking` accepts Claude/Kimi, not MiniMax. MiniMax reaches
  the manual branch: low/medium/high/xhigh map to budgets 4000/8000/16000/32000;
  unknown levels use 8000. No `output_config.effort` is produced there. Explicit
  disabled reasoning returns an empty options object for MiniMax. Thus low and
  off in this adapter do not establish low effort on the wire. This is a static
  source finding, not proof that no other caller can override it.
  [Pinned adapter](https://github.com/NousResearch/hermes-agent/blob/09b1348ab8f27effe5fd70f6759400c3d3078133/agent/anthropic_adapter.py)
- MiniMax documents five levels and hosted default max. The protocol fields are
  Anthropic `output_config.effort`, Chat Completions `reasoning_effort`, Responses
  `reasoning.effort`. Disabling thinking is rejected; omission is not a documented
  disable. [Official guide](https://platform.minimax.io/docs/guides/text-generation)
- OpenClaw PR 160089 merged October 10. It normalizes effort before Anthropic
  history conversion to preserve signed thinking, then sends adaptive thinking
  and the selected effort. Its recorded low/max CN probes returned OK, but do
  not demonstrate loop reduction. [Merged PR](https://github.com/openclaw/openclaw/pull/160089)
- Follow-up issue 177387 remains open: releases 2026.10.4/10.5 recognize only
  `MiniMax-M3.1-Flash-Preview`. The released `MiniMax-M3.1-Flash` falls into the
  older M3 profile and skips the effort wrapper. Initial evidence combines
  source inspection and resolved settings, explicitly not a wire capture.
  The linked replacement-ID fix was still unmerged in the reviewed discussion.
  [Release-ID regression](https://github.com/openclaw/openclaw/issues/177387)

### Inferences

- A user's selected low/off level is weak evidence unless the outgoing request
  and endpoint are known. The Hermes issue supplies a plausible mechanism for
  the apparent contradiction in the Reddit low/off report, not an attribution.
- Before comparing quality, capture the actual field at the provider boundary
  and, for local inference, confirm the effort tag in the rendered prompt.

### Gaps

- No link establishes that the Reddit reporters used the affected Hermes commit,
  the MiniMax Anthropic endpoint, or the OpenClaw path.
- The current local Kilo config was inspected read-only: its MiniMax provider is
  `@ai-sdk/openai-compatible` with a local served-model alias, interleaved
  `reasoning_content`, temperature 1.0/topP 0.95 and no effort variants. It does
  not use Hermes's Anthropic adapter.
- Read-only inspection of serving-host `/workspace/m31-history-cases.json`
  confirms the earlier `late_low_effort` request contains
  `chat_template_kwargs.reasoning_effort=low`, temperature 0, max_tokens 512.
  The checked-in response receipt has 5558 prompt tokens, 103 generated tokens
  and one further README/git call. The request therefore did explicitly pass
  the local template argument; this audit did not recover a rendered-token
  receipt. [Local response](../../artifacts/minimax-m31-loop-isolation-20261025/history-late_low_effort.json)
- The local source `chat_template.jinja` lines 85–88 explicitly render
  `<effort>low</effort>` when passed `reasoning_effort=low`; the template otherwise
  describes adaptive thinking and emits no effort tag. This branch supports the
  diagnostic request's argument. The actionable local comparison is to add an
  explicit effort setting to a fresh Kilo request and verify rendering, rather
  than importing a Hermes Anthropic fix. No config was changed in this audit.

## Are repeated context or parser bugs confirmed causes?

### Takeaway

There is a concrete parser repair and a disputed duplicate-compaction report.
Their observable failure modes differ from our model generating additional
discovery commands with complete tool history.

### Cited Findings

- Dynamo PR 17197 takes MiniMax reasoning-parser start state from the rendered
  prompt because M3.1's template ignores `thinking_mode`; it also finalizes the
  parser at stream end and keeps guided output as content. It is merged into
  the special release branch. This repairs output classification, not a
  documented repeated-tool loop. [Implementation PR](https://github.com/ai-dynamo/dynamo/pull/17197)
- Official-checkpoint discussion 6 reports streaming `<mm:think>` text delivered
  as content while non-streaming separates it correctly, using source revision
  `54e4e69` and the MiniMax vLLM image. It has a simple arithmetic reproduction.
  [Checkpoint report](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/discussions/6)
- OpenCode 57805 reports paired compactions in v2.0.32 on a MiniMax subagent,
  each reusing a 267451-token cached prefix. The author supplies message rows
  and proposes skipping a repeated auto-check plus waiting for the checkpoint
  projection. They report their patched Windows binary removes the pairs.
  The project bot failed to reproduce and closed the issue October 24. Its
  closing claim that requested details never arrived conflicts with the
  reporter's October 21 detailed follow-up. Treat this as contested, not a
  verified upstream fix. [Report and full discussion](https://github.com/anomalyco/opencode/issues/57805)

### Inferences

- Client checkpoint races are different from stale NPU KV reuse. A repeated
  cached-token count alone does not establish bad KV contents.
- For our session, inspect for actual compaction events before applying any
  compaction workaround. The quoted public case crossed a 266K input ceiling;
  our reconstructed continuation was about 29.6K against 160K capacity.
- Existing local cold/warm equality and raw XML calls with parsing disabled
  reduce the relevance of these two public defects to the observed generation.
  [Local evidence](../../artifacts/minimax-m31-loop-isolation-20261025/README.md)

### Gaps

- No controlled public experiment found here connects the parser fix or the
  duplicate-compaction patch to the specific repeated-discovery behavior.
- No live hosted calls, local inference, service modifications, or third-party
  code execution were performed in this audit. Source files were read only.
