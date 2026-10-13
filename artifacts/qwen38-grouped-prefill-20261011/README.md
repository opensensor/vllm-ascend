# Qwen TP6 grouped prefill qualification

The prefill-only local-count schedule is admitted on the live 1M YaRN server.
It reduces cold time to first token by approximately 4% in the matched serving
tests. It does not establish the requested 50% overall throughput improvement.

## Change and boundaries

The grouped projection allocation includes routes owned by other ranks. The
previous schedule used that allocation to choose a 128-row, 64-column Cube
tile. At 2,560 tokens on TP6, local experts have substantially less work than
that capacity suggests. Read the final INT64 group boundary on-device and use
the existing 32-row, 160-column schedule when local work is below the dense
threshold. Keep the original weight layout, quantization, ordered FP32 group
accumulation, FP16 outputs, route reduction, and collective behavior.

The resident binding changes only supported grouped projection shapes whose
allocated rows exceed 256 times the local expert count. Routed INT32 decode,
smaller prefill batches, and MTP execution retain their installed operators.
There is no expanded weight bank, host-side expert count, or additional
activation conversion. The production OPP source has the matching policy fix;
the live deployment uses an isolated direct kernel rather than replacing OPP
packages. A rebuilt production OPP package remains a separate release gate.

The local-route gather experiment also replaces four DMA handoffs per route
with one: four disjoint UB regions permit all reads, then all writes, before
reuse. This cuts event waits from eight to two. Its improved component passed
the hardware checks, but it is not enabled in the admitted serving candidate.
No measured bus-byte or thermal reduction is claimed.

## Measurements

All serving prompts were cold: `cached_tokens=0`. Each generated 32 tokens.
The initial baseline and candidate have two trials per prompt length; the
return to baseline has one. The return measurement helps distinguish the code
effect from changes caused by recapture, cache clearing, or temperature.

| Prompt tokens | Initial baseline median TTFT | Candidate median TTFT | Return to baseline | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | 18.33 s | 17.60 s | 18.43 s | 4.01% |
| 32,768 | 75.46 s | 72.44 s | 75.62 s | 4.01% |

Real-weight MoE components include dispatch, both projections, activation
packing, and local route finalization. They exclude shared experts,
collectives, and the outer model. Inputs are synthetic; the router and expert
weights come from the actual TP6 checkpoint.

| Layer / rank | Tokens | Installed | Local-count schedule | Exact output bytes |
| --- | ---: | ---: | ---: | --- |
| 0 / 0 | 2,560 | 46.81 ms | 38.89 ms | Yes |
| 47 / 5 | 2,560 | 38.93 ms | 31.93 ms | Yes |

The gather revision plus scheduling measured 38.69 ms and 31.57 ms respectively.
At 640 tokens, switching to the direct schedule added overhead; the resident
binding deliberately retains the installed operator for those batches.

## Correctness evidence

- All six chips passed 12 exact arithmetic/layout checks each, including
  empty groups, partial local groups, and poisoned inactive route indices.
- A controlled serving diagnostic computed both projections on the actual
  activations and returned the original result. Across all six workers,
  1,728 comparisons found zero different FP16 output elements. The diagnostic
  deliberately synchronizes scalar results and is not left enabled.
- Three of four initial serving output hashes match between baseline and
  candidate. The remaining prompt also produced alternate hashes when the
  diagnostic returned original operator results. Whole-generation
  reproducibility remains unresolved; it must not be mislabeled as byte-exact
  whole-model validation. The serving probe uses temperature zero and does
  not specify a seed.
- 175 scoped CPU tests passed. These include native gather bodies, production
  schedule dispatch compiled against recording stubs, decode fallback, and
  earlier INT8, QSA workspace, YaRN, and request-timing regressions.
- Scoped manual pre-commit checks pass. The required repository-wide
  `bash format.sh ci` still fails on existing unrelated lint errors. Its
  unrelated automatic edits were restored in the isolated worktree.

The final three 256-token coding/engineering decode probes have a median
27.98 tokens/s. Decode remains approximately at the earlier qualified TP6
rate; this prefill change does not meet the historical 28.89 tokens/s target
for a 50% increase over the older two-card reference. Marker, arithmetic,
Python code, fresh image, and repeated-image response checks all passed. The
repeated image reused 1,024 cached tokens; its TTFT fell from 3.10 to 0.47 s.

The final mixed-traffic probe completed without an engine hang: serial decode
was 29.73 tokens/s; decode while an 8K cold prompt ran was 16.96 tokens/s, with
a maximum 0.71-second streamed-chunk gap and 65.33-second long-request TTFT.
This does not establish a mixed-traffic improvement. Prefill interference
remains a substantial bottleneck despite the standalone cold-prefill gain.

Raw receipts, native build hashes, per-layer timings, all-rank real-activation
checks, and the exact serving probe are retained in this directory. See
`summary.json` for the aggregate cold-prefill and arithmetic checks.

## Runtime and next work

The live API and six worker PIDs remain unchanged. Resident transactions
invalidate prefix caches and recapture decode graphs; they preserve model
weight storage. The original profile remains available for rollback:

- Admitted profile: `/srv/ai/src/qwen-grouped-prefill-tp6-20261011`.
- Native build: `/srv/ai/src/qwen-grouped-prefill-20261011-v3`;
  namespace `qwen_prefill_v5`.
- Original profile: `/srv/ai/src/qwen-yarn-1m-tp6-20261011`.

The admitted profile includes a launcher and bootstrap for the two native
manifests. Their syntax is checked; a fresh startup of that profile is not
part of this qualification. The tested deployment is the resident cutover.
The model still advertises 1,048,576 tokens with static YaRN; full-window
quality and concurrent full-window capacity remain separate qualification
work from these 8K/32K probes.

Next, qualify generation reproducibility and mixed-expert INT8 execution.
The actual router produced up to 604 rows for one expert in a 2,560-token
chunk, while many experts had few rows. That supports testing per-expert
crossover dispatch. Packed INT8 expansion should be restricted to groups
where its measured reuse amortizes conversion; decode keeps the faster
qualified path. The wider prefill path alone does not solve latency while
another request decodes, because those paced batches deliberately stay small.
