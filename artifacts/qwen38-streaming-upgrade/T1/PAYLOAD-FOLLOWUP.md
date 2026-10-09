# T1: freeze concrete input payloads offline

Use [reference-v2.json](reference-v2.json) for subsequent implementation. The
original [reference.json](reference.json) was preserved byte-for-byte. Clean source,
arithmetic and capacity identities are unchanged; the full manifest identity
changed because its workload input bodies are now frozen.

The deduplicated payload catalog includes a short text request, 200 numbered
engineering paragraphs for the 23K-token target, 640 paragraphs for the long target,
a concrete temperature-fixture tool schema/input/result fixture, and a mixed
arrival case that introduces the prefill payload after generated token 16 of an
ongoing decode. Fresh/cached image cases share the frozen instruction and retain
explicitly unresolved image content. The 132 C1–C4/MTP0–2 workload variants bind
canonical conversation and rendered request hashes; warm controls match cold input
bytes and generation settings exactly.

These are authored deterministic benchmark fixtures. They do not claim recovery of
production conversation logs, actual tokenizer output, measured prompt lengths or
hardware behavior. Model ID, tokenizer ID/revision, exact token IDs and image bytes
remain pending. Desired prompt sizes and actual tokenizer counts are separate.
Filling any pending request identity produces a new manifest and invalidates
dependent evidence. Hardware admission remains blocked.

Validation: 41 CPU tests passed in 8.75 seconds, including concrete payload hashes,
tool schema, mixed arrival, altered rendered requests, and backwards validation of
the untouched original reference. Scoped Ruff passed. No HTTP, SSH, native loading,
NPU allocation or server changes occurred.

The follow-up adds `payload_revision: 2`, `request_payloads`, workload
`request_payload_ref`, `rendered_request_sha256`, and `actual_prompt_tokens` fields.
Revision-2 trial receipts require the rendered request hash. Existing canonical
digest, source, arithmetic and candidate binding APIs are unchanged.
