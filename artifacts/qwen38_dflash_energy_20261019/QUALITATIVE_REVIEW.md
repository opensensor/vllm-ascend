# Manual review of saved completions — 2026-10-19

This reviews the ten plain, K3, and K7 responses in [k0.json](k0.json), [k3.json](k3.json), and [k7.json](k7.json). It asks whether the visible text is understandable and whether differences matter to a reader. It is an informal inspection of one sample per prompt, not a blinded quality evaluation.

Every request was a raw completion forced to exactly 256 output tokens with EOS ignored. The cap cuts off code, explanations, and stories; a response continuing after its normal stopping point may also be misleading. These conditions are useful for decode throughput but poor for answer quality.

| Prompt | Plain vs K7 manual finding |
| --- | --- |
| Speculative decoding explanation | Both are readable and convey the same broad mechanism. They organize the explanation differently and stop before completing the requested verification and acceptance details. |
| Python LRU cache | Exact same output tokens. The code stops inside `put`, so neither saved response is usable as a complete implementation. |
| Rolling seven-day SQL | Both produce the same seven-*row* query and correctly warn that it is not a true seven-calendar-day window. K7 changes the later database-support claim to include SQL Server 2012+ for an interval `RANGE` frame; [Microsoft's `OVER` documentation](https://learn.microsoft.com/en-us/sql/t-sql/queries/select-over-clause-transact-sql?view=sql-server-ver17) says an offset `PRECEDING` bound is not allowed with `RANGE`. This is a material factual error in K7. Both responses stop while starting the corrected query. |
| Queue shutdown debugging | Both describe plausible races in readable prose, but they choose different examples. Neither reaches the requested robust shutdown protocol before the cap. |
| Collision probability | Both state the same main exact formula and birthday approximation. K7 later writes `m! falling factorial` in a formula, which is malformed notation. The plain response also drifts into stray meta text near the cutoff. |
| Latency and throughput | Both give the same sensible 100 ms single-request latency and 10 requests/s at concurrency one. Both cut off before a useful concurrency analysis. |
| Spanish translation | Exact same tokens, but the visible output remains inside `<think>` and does not deliver a final answer. |
| Radio-telescope story | Plain output remains inside planning/thinking for all 256 tokens. K7 exits thinking and starts a coherent story. Their visible outputs are materially different; neither contains the requested ending. |
| Serving-system summary | Exact same tokens. The summary is readable but ends in the middle of the caching tradeoffs. |
| Deployment checklist | Nearly the same first four items and readable, but both stop mid-checklist. |

K3 follows the same overall pattern: coherent fragments for most prompts, exact match on the Python and summary prompts, and similar incompleteness caused by the cap. Its math response also contains the malformed `m! falling factorial` notation; its story starts but has no ending.

**Conclusion:** Most nonmatching passages still make human sense and often convey similar high-level ideas. Exact token mismatch is not itself a quality failure. However, the SQL claim and math notation show that a near-tie token change can produce a meaningful error, and the story differs substantially in what a reader receives. These saved responses do not establish semantic equivalence or deployment quality. A useful next evaluation needs chat-formatted requests, natural stopping, sufficient token budget, repeated samples where appropriate, and task-specific scoring or blinded human review.
