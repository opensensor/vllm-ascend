# Resident prefill selector audit

The tiled scorer remains experimental. A matched cold retrieval measured
**89.976 s baseline versus 89.831 s candidate** (8192-token target, same
variant 7 and seed 42). Both answered `BLUE-ORCHID-7319-7`. This single pair
is effectively flat; it does not establish a serving improvement.
Workers and weight-storage digests stayed unchanged. Baseline was restored.

## Numerical discrepancy

The earlier 640-query exact-selection regression was reproduced in all four
resident workers. Query rotation matched exactly. Maximum score error was
`6.103515625e-5`; mean absolute error was `2.7297203359921696e-6`.

Only **one query row (436) exchanged one selected pool**: baseline selected
306, candidate selected 1755. Their baseline cutoff gap was
`1.9073486328125e-6`; the score ordering reversed under native reduction.
The CPU FP32 reference favored the candidate's choice in this example.
This explains the failure but does not make the candidate bitwise equivalent
or establish full-model quality. Sorting expanded token indices amplified
the earlier mismatch count to 1572 entries.

Complete synthetic selector medians across ranks were 18.22–18.38 ms
baseline and 15.01–15.48 ms candidate, including baseline key gathering.
That isolated improvement did not produce a meaningful cold 8K TTFT gain.
Longer live contexts could behave differently and remain unqualified.

Evidence: `selector-audit-results.json`, `serving-ab.jsonl`, `compare.log`,
and `comparison-restored.json`. No server restart was used.
