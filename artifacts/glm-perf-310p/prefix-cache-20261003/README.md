# GLM 310P hybrid prefix-cache promotion (2026-10-03)

## Outcome

Four-rank GLM-5.3-Flash W2/W4 now serves with `FULL_DECODE_ONLY` graphs and
hybrid prefix caching at 22,528 advertised context tokens. A repeated
7,269-token retrieval prompt reused 7,040 tokens: measured time to first token
fell from 169.97 s (cold) to 7.93 s (warm, 21.4x), with the exact answer
`BLUE-ORCHID-7319` in both runs. This is a repeated-prefix gain, not a decode
gain.

The small incomplete indexer K-pool state is intentionally not hashed for
prefix reuse. Full MLA and aligned KDA blocks remain cacheable. Passing the
spec's `prefix_cacheable` flag into each single-type cache manager prevents a
false hash of the 4/16-token tail against the 512/640-token scheduler block.
`--max-num-batched-tokens 640` aligns KDA checkpoints and bounds 8-way expert
routes at 5,120 per prefill chunk.

## Validation and limits

| Check | Result |
| --- | --- |
| Real checkpoint and graph server | 151.58 GiB checkpoint loaded; four-rank graph capture and first OpenAI-compatible request passed |
| Unit tests in isolated remote source | GLM cache config 21/21; hybrid coordinator 23/23; compressed prefix 11/11; cache interface 2/2 |
| Long retrieval, cold/warm | 7,269 prompt tokens, exact answer both times; 7,040 cached on warm request |
| Strict quality | 17/20, same failing IDs as previous promotion: `instr_reverse`, `instr_first`, `code_slice`; all 20 responses valid |
| 256-token serving, c1/c4 | 2.679 / 6.638 aggregate tok/s, five of five valid, no early EOS |
| Previous promoted c1/c4 | 2.672 / 6.255 aggregate tok/s; separate runs, so only no observed decode regression is claimed |
| Cache capacity | 25,003 tokens with prefix caching, versus 32,768 without; advertised context remains 22,528 |

The retrieval suite also submitted a separate 32,768-token target that exceeds
the advertised context and was correctly rejected (HTTP 400). The cold/warm
exact-answer test uses only the 7,269-token case. The suite's `early_eos` flag
and target-token label do not negate answer correctness. The 17/20 quality
gate remains unresolved; no bitwise-parity claim is made because first-token
outputs show known nondeterminism. MTP and multimodal paths were not enabled.

The candidate remains the live server at `192.168.53.187:8001` (model
`glm53-flash-ascend-graph`). The reproducible launcher is
[`serve-glm-prefix-22k.sh`](serve-glm-prefix-22k.sh). Remote source:
`/srv/ai/src/glm-prefix-20261003`; server log:
`/home/matteius/experiments/glm-gate-a-20261002/server-prefix-22k-20261003.log`.
The prior known-good launcher for rollback is
`/home/matteius/experiments/glm-gate-a-20261002/serve-glm-context-22k-20261003.sh`.
Benchmark outputs on the development host are
`/tmp/glm-prefix-quality-20261003.jsonl`,
`/tmp/glm-prefix-short-20261003.jsonl`, and
`/tmp/glm-prefix-retrieval-first-20261003.jsonl`, each with a `.summary.json`.
