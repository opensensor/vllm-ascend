# M3.1 direct cache gathering and graph investigation

Direct cache-row gathering is retained in serving. Three matched 64-token
requests measured 3.328, 3.340, and 3.343 tokens/s after first output, versus
2.231 for the prior INT8 service. All three messages matched exactly. This is
49.7% above INT8 alone and 82.5% above the earlier 1.830-token/s FP16 v9 service.
Context capacity remains 163840; weights and worker PIDs were retained.

The original trace showed `aclnnIndex_SliceAiCore_Slice` copying an entire
allocated KV-cache head before gathering selected pages. It accounted for
40.6% of first-stage compute-task time, excluding communication. Direct indexing
removes that capacity-sized copy. Private-cache checks covered permuted pages,
NaN-filled unused pages, decode through 16K context, and a 64-token prefill.
The largest output difference was 0.0000153. A 160K allocation was tested with
shorter sequences; no full 160K prompt or broad quality evaluation was run.

`profile-summary.json` and `post-gather-profile-summary.json` summarize traces
stored on the serving host under `/workspace/m31-int8-decode-profile` and
`/workspace/m31-gather-decode-profile`. Task times overlap and are not additive
end-to-end latency. Trace analysis ran outside the daemon workers. The CPU
suite passed 200 tests with 24 hardware skips.

## Graph investigation remains open

Expert-only graphs were 2.22x faster in an isolated real-weight projection
probe but measured 3.334 tokens/s in serving, versus 3.340 without them. Replay
counters confirmed 114 graphs and 1254 replays during a separate short request;
the path was actually used. This narrow graph scope was removed. That result
does not rule out gains from broader or better-integrated graph execution.

HCCL graph replay passed changing-input checks on all six resident workers.
An isolated second communicator could not bind the device port already owned
by the service, so the test used existing communicators while drained instead.

A pipeline-stage graph captured dense layers, attention, experts, and TP
reductions, with three deduplicated metadata copies per worker. It measured
3.308 tokens/s and generated different reasoning, so it was not retained.
A four-token shadow run compared each layer against eager execution. Some
small output differences grew across layers; the cause is still under
investigation. Private attention graphs matched eager exactly while sequence
lengths, slots, and queries changed, so the suspected scalar-length issue was
not reproduced in that isolated test. These prototypes are not qualified
serving optimizations. Their sources and receipts are retained for follow-up.

Request caches were invalidated after the stage-graph experiment and around
the shadow run. The working direct-gather implementation was restored. There
were no worker restarts or shard reloads. The installed Python source includes
the direct gather for future launches; the resident module patch remains active.

Raw experimental source snapshots use `.py.txt` to preserve the exact executed
text. Copy them to their recorded `.py` names to reproduce; they are diagnostics,
not production entry points. The sealed INT8 kernel and layout-aware rollback
instructions remain in the preceding INT8 artifact directory and model tutorial.
