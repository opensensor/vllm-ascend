# Selected dense INT8 serving optimization

Retained on six 310P chips with INT8 expert kernel v15, activation v2 and all
240 partial graph pieces. Context capacity remains 163840.

| Same short decode workload | Median tokens/s |
| --- | ---: |
| Compact v15 + unwrapped partial graphs | 3.776924 |
| Plus shared-expert and attention-output W8A8 | 3.958147 |

Three greedy 64-token counting requests: 3.961430, 3.958147 and 3.954692 tokens/s.
Timing excludes the first token; all three response messages match the baseline.
The separate short tool-call and Python-function checks passed. These are bounded
checks, not broad quality or full-context qualification.

348 local projections converted across six workers, with maximum random-input
NRMSE 0.01606 against FP16. Saved device weight storage totals 5.812 GiB, or
0.945–0.995 GiB per rank. Exact FP16 recovery copies remain in host RAM. Expert
shards and worker PIDs remained resident; dense parameter storage intentionally
changed. QKV/indexer, routing, vision and vocabulary weights are unchanged.

The isolated projection probe compares eager and captured FP16/dynamic W8A8
operations on real TP2 weights. Small K/indexer projections were slower with
INT8 and were excluded. This probe is synthetic input evidence, not a complete
layer-quality comparison. Run the saved `probe.py.txt` as a Python script on the
serving host with its environment sourced; it does not mutate the service.

Conversion releases old captures before replacing weights and reinstalls the
complete saved serving generation afterward. CPU tests cover exact recovery,
injected conversion failure, all-worker recovery, source mismatch and uncertain
RPC handling. A live FP16 restoration was not exercised. All 237 CPU tests pass;
24 NPU-specific unit tests skip locally. Real hardware results are recorded here.

The startup script now selects v15 directly and enables the dense conversion.
The generated source matched the live snapshot exactly, and shell syntax passed.
A fresh cold restart was not repeated. `checks.py.txt` and `probe.py.txt` are
historical executable snapshots; JSON files contain the recorded results.
