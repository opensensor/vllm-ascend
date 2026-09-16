# Long-context retrieval probes (Qwen4Exp 1M on Ascend 310P) — plan T0.4

This directory holds the end-to-end harness assets for the 1M-context
needle-in-a-haystack (NIAH) validation wave. The prompt bodies themselves are
built by the deterministic generator at
[`tools/qwen38_1m/corpus_gen.py`](../../../tools/qwen38_1m/corpus_gen.py); this
directory only carries the server-side configuration required to serve them
faithfully.

## What the generator produces

`corpus_gen.build_prompt(target_tokens, tokenizer, seed=...)` returns a
`BuiltPrompt` that encodes to **exactly** `target_tokens` under the injected
tokenizer, at the five documented sizes:

| Size label | Exact tokens |
| ---------- | ------------ |
| 8K         | 8192         |
| 128K       | 131072       |
| 256K       | 262144       |
| 512K       | 524288       |
| ~1M        | 1048060 (`1048576 − 516`, 516 reserved for chat template + generation) |

Each prompt embeds **8 answerable records** distributed near the *beginning,
quarter, middle, three-quarter and end* of the document, plus **1 adversarial
no-answer record** whose fact is deliberately omitted (expected answer =
`corpus_gen.NO_ANSWER`). Retrieve the grading keys with
`built.expected_answers()` → `{needle_id: expected_answer}`.

### Injected tokenizer (important)

The real Qwen4Exp checkpoint tokenizer is **not** on the authoring host, so the
generator takes the tokenizer as an injected dependency (`encode(text) ->
list[int]` plus an `eos_id`). Unit tests inject a deterministic whitespace
tokenizer so token counts are exact and reproducible. **Exact-count validation
against the real checkpoint tokenizer is deferred to the device wave (plan D1)** —
run `build_all(real_tokenizer)` on-device and confirm `token_count == target`
before trusting the counts.

## Serving requirements (R10)

The server must **not** truncate the input, and must **report the accepted /
prompt token count** so we can confirm the full context was ingested. The
snippets below configure both:

- [`server-config.yaml`](./server-config.yaml) — vLLM/vLLM-Ascend engine args.
- [`request-template.json`](./request-template.json) — OpenAI-compatible request
  body that disables truncation and requests usage (prompt-token) reporting.

### Grading protocol

1. Build the prompt for a target size with the checkpoint tokenizer.
2. Assert the server's reported `usage.prompt_tokens` equals `built.token_count`
   (R10): this proves no truncation occurred.
3. For each answerable needle, check the model output contains
   `expected_answers()[needle_id]`.
4. For the adversarial needle, a faithful model must decline (emit `NO ANSWER`),
   not hallucinate a code.
