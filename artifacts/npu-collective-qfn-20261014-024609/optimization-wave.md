# QFN optimization wave — 2026-10-14

Baseline: the promoted pinned-position custom collective runtime in
[profile-c1/promotion.json](profile-c1/promotion.json), C1 median 31.42 tok/s.
The timing overlaps the prior baseline; this is not an established speed gain.

## Stable client identity

- Kilo model: `vllm-310p-w4/qwen38-flash-next`
- API model ID: `qwen38-flash-next`
- API base URL: `http://192.168.53.187:8001/v1`
- The API was restored in `resident-live-v3` with QSA disabled after its
  boundary experiment failed. Experimental artifact names do not become client IDs.
- Kilo's stale default model reference was corrected; provider URL/model entries
  were preserved. Its configured context advertisement is separate from this
  benchmark's qualified 262144-token context and does not qualify a larger limit.

## Parallel ownership

| Agent | Isolated candidate | Main correctness constraint |
| --- | --- | --- |
| `qfn_qsa_graph` | `qsa_graph_selection.py` | Changing metadata, causal boundaries, exact selected group order |
| `qfn_draft_graph` | `draft_graph_experts.py` | Preserve W8A16 expert math and route-slot accumulation |
| `qfn_fp16_projection` | `hc_projection_layout.py` | Preserve checkpoint weights and compute precision |

Agents own new candidate, benchmark, test and note files only. Shared model edits
and hardware execution belong to the parent. Candidates first pass CPU gates,
then changing-input hardware parity and graph replay, then unprofiled C1 model
comparison. Components are evaluated separately before combining them.

All candidates must preserve custom collective native ownership and unobserved
replay. No graph observer is added to Python replay. Existing qualified cache,
loader, OPP, context and precision settings remain fixed for comparisons.

The profiled opportunities are graph fragmentation (22 native launches and 19
eager callbacks per MTP2 round), small FP16 matrix operations and layout costs.
Recorded device-task gaps include dependencies; they are not a guaranteed pool
of removable overhead. The HC row-padding candidate completed a hardware component test with real
layer-0 attention hyperconnection weights. Outputs were byte-identical for
changing inputs in eager and graph modes, but complete mix/combine graph event
latency regressed from 0.10411 to 0.11477 ms at 3 rows and from 0.11645 to
0.12178 ms at 6 rows. This candidate is rejected for performance; no further
module sweep or end-to-end run is warranted from these results. The receipt's
qualification flag covers only that component scope, not model qualification.
See [hardware receipt](hc-projection-layout-v1.json).

The draft candidate passed the first synthetic one-NPU changing-route gate at
1/3/6 tokens (30 steps each, including no-local and all-local routes). Component
wall latency was 1.50→0.49, 1.72→0.75 and 2.12→1.02 ms. These small-bank results
are not model throughput. The candidate retains one dynamic-dispatch callback
and captures router/shared-expert/reduction work around it. The private TP4 model trial completed at **32.94, 32.71 and 32.44 tok/s**,
median **32.71**, compared with the earlier promoted baseline median 31.42
(+4.1%). These are sequential short-prompt wall measurements including prefill,
not interleaved A/B or broad quality qualification. All four workers closed,
retired four captures and released all 56 native leases each. Production replay
tracking stayed disabled. Draft FP32 eager SUM calls disappeared from the timed
C1 path, confirming that work moved into capture. Graph segment and eager-break
counts remain unchanged; the callback now contains less work. The new candidate
was subsequently installed in the resident API through hot reload. The promoted
cold baseline remains unchanged.
See [model comparison](draft-expert-candidate/model-comparison.json) and
[all-rank result](qfn-c1-draft-v2/result.json).

QSA component gates also passed on hardware: 13 cases across single and packed
queries, including the 2051/2052 selection boundary and full 262144 context.
Single-query selection dropped from 0.383→0.067 ms at 512 context and
0.564→0.174 ms at 8192; full-context improvement was small (2.195→2.173 ms).
The packed 2052-token bucket regressed 0.636→0.761 ms and should retain a fallback
or use a narrower separately tested capacity. These are component results only.
The model dispatcher and per-capture lease retirement are now implemented and
offline tested, but remain undeployed pending full-model boundary validation.
See [QSA findings](qsa-graph-selection/README.md).

The persistent API measured 31.56 tok/s baseline versus 31.69 with draft
partitioning on a different short prompt: effectively neutral. Hot reloads took
2.89 and 2.80 seconds with unchanged worker and weight-storage identities. The
real loaded draft banks passed 240 exact local comparisons. API streaming and
tool calls passed; full-model generated sequences were not identical across
all trials. See [resident service handoff](resident-live-v2/README.md) for the
running endpoint and limitations. Leave that server up for further experiments.

The later QSA trial removed twelve target eager callbacks and measured 32.18
tok/s median at short context, but failed during live boundary recapture because
it attempted capture on the default stream. The engine exited. Healthy idle
devices allowed restoration of the prior serving configuration as
`resident-live-v3`. The source factory is now blocked pending exact-window and
full-model qualification. See [QSA failure record](qsa-resident-v1/README.md).

## Corrected QSA follow-up

Cold capture variants now isolate GDN derived metadata and remain enabled on
`resident-live-v3`. Eight full-model boundary/concurrency/return-to-prefix
requests passed with unchanged workers. Matched C1 HTTP medians were 31.55
before and 31.66 tok/s after, effectively neutral. The qLens owned-buffer
capacity fix is retained. Native profiler capture and subsequent candidate
reloads retired/rebuilt five captures and 70 leases per rank without restarting
the service. See [current evidence](qsa-resident-v1/README.md).
