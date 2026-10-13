# Qwen TP6: experimental 1M YaRN and transfer repairs

The live profile serves `qwen38-flash-next` at
`http://192.168.53.187:8001/v1`. It extends the existing checkpoint's 262,144
native limit to 1,048,576 tokens using static YaRN factor 4. All five real-weight
startup checks passed: marker, arithmetic, Python code, fresh image, cached image.
The user elected to test long histories interactively; million-token retrieval
and language quality have **not** been qualified.

## Architecture and rotary policy

The model retains 48 layers, hidden size 2,560, 36 GDN layers, 12 QSA layers,
512 routed experts with top-10 routing, shared experts, PLE, and MTP2. QSA and
its indexer rotate the same leading 64 dimensions. Nested text-config rope
parameters take precedence over legacy top-level settings. YaRN frequencies
are constructed on CPU once and stored as nonpersistent device buffers before
graph capture. Shared step tables are keyed by the complete immutable policy.

At million-token positions, direct NPU FP32 trigonometry produced maximum
absolute table error 0.0189859. Three-part FP32 angle reduction reduced this to
4.41e-6 on all six chips, without FP64 or a hot-path CPU transfer. The native
unscaled rotary path retains its previous behavior. There are only 32 FP32
inverse frequencies per layer, not million-position precomputed tables.

## Startup failures and actual fixes

The first launch was rejected by upstream vLLM's long-context guard. The profile
now sets the existing `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` after implementing YaRN
in both the attention and indexer paths. The guard alone would not implement it.

The next launch passed text checks but failed fresh image processing. Device
plog identified `paged_attention_decoder_mask_0` with an invalid MTE burst;
the later Mean/Silu traceback was an asynchronous symptom. Exact dense QSA
only addresses its 2,048-token selection budget, but received the padded 8,192
page table for 1M capacity. Passing only the needed page-table prefix fixed
fresh and cached images. Tests cover six-row mapping, both block sizes, small
and large capacity, contiguity, and zero-copy single-row slices. This repair
does not truncate sparse long-context attention.

A retry encountered an orphaned worker from the failed device process. Only
that failed worker was terminated before relaunch. The final API PID was
3024361 with workers 3027388 through 3027393. All six acknowledged the candidate
and had clean graphs. Maintenance was released and the server was unpaused.
See `bootstrap-ready.json` and the five `bootstrap-smoke-*.json` receipts.

## Capacity and features

| Feature | Result |
| --- | --- |
| Real weights | Five short inference checks passed; no dummy-only sign-off |
| Dummy stage | Not used; retained architecture and converted weights |
| TP6 / EP6 | Enabled on all six chips; existing unequal expert shards |
| ACL graphs | Breakable FULL_DECODE_ONLY, capture sizes 3 and 6 |
| MTP | Two draft tokens; real startup generation passed |
| Images | Fresh 1,024-pixel image and cached repeat passed |
| FlashComm1 | Retained disabled setting; not newly qualified |
| Async scheduling | Retained disabled setting |
| Thermal control | 94 C hold, all six at or below 85 C resume, 96 C cutoff |
| Per-request maximum | 1,048,576 including generated tokens |
| Shared KV pool | Planner reports 1,690,963 tokens, about 1.61 full 1M histories |
| Active request slots | Two; two independent full 1M histories are not guaranteed |
| 128K / batch 16 baseline | Not run; user requested 1M interactive profile |
| Million-token quality | Unqualified; optional progressive retrieval gate provided |

The successful path retains graphs, EP, MTP, and image processing; eager or
text-only fallback was not needed after the table repair. Prefix caching shares
identical token prefixes across requests. Each request still owns its logical
history, position accounting, and context limit. Cached tokens count toward
that limit. Kilo determines compaction; vLLM does not summarize conversations.

## Transfer and performance evidence

Huawei's gather warning concerns INT64 gathered **data**. The NPU fast top-k
path now compacts selected group IDs and their gather indices to INT32. Index
ordering also passes through INT32 before FP32 to avoid a direct INT64 cast.
CPU selection remains INT64. Changed-input graph replay matched reference on
all six chips; see `gather-graph-probe-six-final.log.txt`.

Initial component comparisons measured roughly 1-6% less selection time; they
did not time the final INT32 intermediate separately and are not end-to-end
speed claims. At 1M capacity, a 2,560-row QSA workspace probe retained the
16 MiB score budget and used 354,422,272 additional allocated bytes (338 MiB),
with twelve sampled rows matching reference. Its 24.575-second cold duration
is not a warmed serving benchmark. See `workspace-npu-probe.log.txt`.

Short startup smoke decode rates ranged from 32.36 to 36.34 tokens/s. Outputs
were only 4-18 tokens and cannot qualify the requested 50% improvement over
the older two-card baseline. No long throughput or TTFT benchmark was run on
the new 1M profile before releasing it to the user.

The current expert path already uses integer Cube arithmetic: A8 activations
are expressed as two INT4 limbs. The isolated raw INT8 G128 probe produced
exact INT32 products and replayed a graph on chip 0, including zero and signed
extreme inputs. Full-K gate/up/down projections on real layer-0 expert-0
weights passed exact FP16 reference and graph replay on chip 5 at rows 1, 3,
16, and 17. The prototype retains G128 FP16 scales and ordered FP32
accumulation, and has CPU decomposition/layout regression tests. It is
**not a serving backend or a proven speedup**. Expanding all W4 weights
to INT8 doubles their stored bytes and could sacrifice the larger context
pool. Prepared-operand timings omit quantization, routing, and expansion.

The direct A8 packer retains the qualified per-row G128 max/127 policy and
round-to-nearest integer boundary while eliminating limb splitting and sums.
On chip 5, quantization plus gate/up/down projection matched both the CPU
reference and the retained native W4A8 operator exactly at rows 1, 3, and 17.
Graph replay passed; see `int8-projection-chip5-fused-v3.json`. These were
correctness probes while user traffic continued, with no timing or speed claim.
`--compare-w4 --fused-quant --time` enables a matched operator comparison in a
controlled idle window. The read-only traffic guard deferred all timings in
`int8-projection-chip5-guarded-comparison-v3.json` because live requests were
active. It also discards timings when requests overlap the measured interval.
Routing, expert diversity, TP reductions, and full-model
generation must still be qualified before integration. No probe was loaded
into the serving workers.

## Startup and validation runbook

Use the pinned environment and retained host checkpoint; do not upgrade
Transformers or independently substitute OPP libraries. The complete runtime
snapshot is `/srv/ai/src/qwen-yarn-1m-runtime-20261011`; the operational profile
is `/srv/ai/src/qwen-yarn-1m-tp6-20261011`. These host paths and port 8001 follow
the user's existing deployment, rather than a generic container example.

```bash
bash /srv/ai/src/qwen-yarn-1m-tp6-20261011/start-server.sh --check-runtime
bash /srv/ai/src/qwen-yarn-1m-tp6-20261011/start-server.sh
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8001/is_paused
```

Do not start a second instance over the live workers. The launcher restores
the admitted modules and checks text, math, code, and two image requests before
releasing maintenance. `start-server.sh.txt` and `bootstrap_profile.py.txt`
record the successful host profile. The live log is `server-8001.log` under
that profile; `bootstrap.log` contains startup verification. Both append
across launches, so use final PID 3024361 to distinguish earlier failures.

An optional controlled-window retrieval gate is
`tools/qwen4exp/validate_yarn_context.py`. It acquires maintenance and leaves
it active for the caller to inspect and release. It must not interrupt the
user's live session. The previous qualified 256K rollback profile remains
`/srv/ai/src/qwen-long-prefill-topk-tp6-20261011/start-server.sh`.

Kilo's matching provider/model entry now declares context 1,048,576, input
1,000,000, output/maxTokens 32,768. Reload Kilo to pick up the change. The
targeted edit preserved other providers and authentication; its backup is
`~/.config/kilo/kilo.jsonc.before-qwen-yarn-1m-20261011`.

## Offline checks and scope

The combined focused QSA and direct INT8 CPU suite passed 266 tests. Scoped
manual pre-commit checks passed. Repository-wide `bash format.sh ci` was attempted and failed
on existing unrelated findings; 144 incidental formatter edits were restored
only in this isolated worktree. The repository is not reported as lint-clean.

Owned changes cover rotary policy, attention/indexer propagation, bounded dense
tables, compact selection, regression tests, optional retrieval/rotary probes,
isolated INT8 experiment tools, tutorial/index, and these evidence records.
No live prefix cache was cleared during subsequent offline speed work.
No general language benchmark, full 1M retrieval result, measured bus traffic,
thermal reduction, or end-to-end INT8 improvement is claimed.
