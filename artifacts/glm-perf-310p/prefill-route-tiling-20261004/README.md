# GLM grouped-MoE prefill route tiling

Status: single-card parity passed; 1,280-token four-rank serving startup failed
the 128K cache-capacity gate before any request. Smaller chunks are pending.
No production launcher was changed.

The serving launcher uses `--max-num-batched-tokens 640` because GLM routes
each token to eight experts and the 310P grouped W2/W4 operator accepts at
most 5,120 route rows per call. The existing packed-NZ method raised if a
larger scheduler chunk reached it. The candidate partitions **whole tokens**
into at most `floor(5120 / top_k)` rows, applies the unchanged grouped operator
to each partition, concatenates routed outputs in token order, and applies an
optional shared expert once to the full input. It does not change the decode
path, code packing, grouped kernel, expert assignment, or current launcher.

CPU tests compare the tiled method against one unpartitioned fake grouped
projection at 640, 641, and 1,281 tokens, including masked peer routes and a
shared expert. They pass bitwise. The full focused W2 method and tiling suite
passes 25/25 using `pytest --noconftest`; Ruff lint, format, and whitespace
checks pass on the changed files. These CPU results are control-flow tests,
not a prefill speed claim. Once the devices were released, the real
310P grouped-operator probe in `tools/glm_perf/probe_grouped_prefill_tiling_310.py`
passed bitwise parity at 640, 641, 1,280, and 1,281 tokens (5,120–10,248
routes) using NZ-packed W4 codes, a fused gate/up bank, and masked routes.
That is still an operator-level result, not full-model or TTFT evidence.

## October 4 four-rank admission result

The known-good 640-token, no-prefix-cache 128K server allocated 150,361 KV
tokens and passed the same-checkpoint 2K and 8K retrieval probes. Their exact
prompt lengths and TTFT were 1,733 tokens / 30.55 s and 7,877 tokens /
177.33 s, respectively; both returned `BLUE-ORCHID-7319`. Its steady
640-token prefill iterations were about 15 s, and its graph-captured decode
remained roughly 3.6 tok/s by the server's inter-token-gap log.

An isolated source tree containing only the route-tiling and KDA-mask changes
was then launched with `--max-num-batched-tokens 1280`, the same 128K context,
and the same 0.70 KV-headroom fraction. Startup failed before any request:
vLLM estimated 6.08 GiB needed for one 128K request against 4.46 GiB
available. The failed run released all four workers and NPUs; no 1,280-token
TTFT or answer-quality result exists. The code-level tiling probe still
passes, but this launcher setting must not be promoted.

The capacity increase is structural: `AscendIndexerKPoolStateSpec` advertises
all compressor-state pages touched by an in-flight chunk so the sliding-window
manager can admit prefill safely. Those state pages use independent global
block IDs, while the GLM physical pool prices each ID as a full MLA plus
indexer page. Doubling the scheduler chunk therefore spends much more memory
than the compressor state itself occupies. This is an accounting/allocation
constraint, not evidence of an OOM or numerical failure in the grouped op.
Avoid simply suppressing the admission check: doing so could stall a long
request at runtime. The next hardware step is a smaller chunk/fraction sweep
that preserves 128K admission and workspace headroom, then matched TTFT and
quality tests; a longer-term fix would decouple these virtual state IDs from
the full-MLA physical page allocation.

There is a narrower path worth prototyping before a general block allocator
rewrite. `SparseAttnIndexerKpool._write_pools` gathers the previous state
before scattering, reconstructs all within-chunk pools from local keys, and
persists only the request's **final** pool (`valid_state` is restricted to
`positions >= final_pool_starts`). With prefix caching disabled, that suggests
one stable live tail page per request, independent of the prefill chunk size.
An implementation would need request-ID-to-lane assignment, a fixed physical
tail pool, device slot mappings using `(lane, position % pool_size)`, and a
scheduler contract that no longer allocates one historical state ID per pool.
Test chunk boundaries at every pool offset, decode rollover, concurrent
requests, lane reuse, and graph replay before claiming this safe. Merely
shrinking `max_memory_usage_bytes` without changing the runtime manager and
addressing would be incorrect.

The saved October 4 rank-0 `op_statistic.csv` and `task_time.csv` under
`/home/matteius/experiments/glm-gate-a-20261002/profile-kda-nz-grouped-20261004/`
provided the next hardware priorities before this new NPU test. The 84
grouped calls of the prefill step total 1,177.6 ms; the next 31 groups of 84
total 2,525.3 ms, or 81.5 ms of grouped task time per decode step. These task
times can overlap other streams and do not establish a critical path or an
end-to-end improvement. The full capture's grouped share is 46.6% of summed
task time. The profiling export is incomplete (only rank 0's CSV was present,
and `trace_view.json` is truncated), so cross-rank timing remains unknown.

## Four-rank hardware gate

1. Keep the existing 640-token launcher as rollback. Extend the single-card
   probe to W2 and 2,560-token inputs before treating the operator boundary
   as fully covered.
2. A direct 1,280-token launch failed 128K admission. First find a smaller
   chunk and KV fraction that retain both 128K capacity and workspace
   headroom; compare that configuration with 640 on matched retrieval prompts.
   Only test 2,560 if 1,280 is eventually both faster and admissible.
3. Recheck 128K startup and long-request admission. The indexer compressor
   state currently reserves pages proportional to the scheduler's maximum
   in-flight tokens; a larger scheduler chunk can consume cache capacity even
   though the grouped operator itself is tiled safely. Do not trade away the
   usable 128K context for a speculative prefill gain.
4. Run the strict 20-case quality gate and c1/c4 decode checks before any
   promotion. This candidate is not a decode optimization; grouped-kernel
   weight traffic remains the measured decode target.
