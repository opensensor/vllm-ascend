# Private real-weight GDN projection gate

Status: **private hardware gate passed on all four ranks** on October 14, 2026.
No model binding or serving change is included. This tests the next experiment described in
[the decode analysis](decode-projection-next.md).

## Scope and admission

The only replacement is
`NPUWorker310.qfn_cpu_profile_stop`, supplied by
`tools.qwen4exp.resident_candidates.qfn_gdn_projection_probe.replacements()`.
Non-JSON output arguments retain the previous RPC behavior. Invoke only after
coordinated pause(wait)/drain, with CPU and native profilers stopped. The RPC
also checks runner execution state and rejects an active capture.

Three separate all-rank RPC actions are required, in this order:

```json
{"action":"preflight"}
{"action":"load"}
{"action":"validate","rows":[3,6,64],"repeats":3,"replays":5}
```

Each payload is the existing RPC's `output` string argument, not a new method.
The parent controller must inspect every rank's receipt before proceeding.
An optional `module` is an exact `get_model().named_modules()` path and must be
supplied consistently to every action. The default selects the first exact
pinned `_GDNAttention`; each receipt identifies the selected module. No DSO,
custom kernel, environment variable or service restart is needed.

Admission verifies exact pinned class, `_linear`, and operand-policy sources,
exact class identity, TP4, FP16 weights and FP32 compute policy. The logical
weights are QKV `[2560,2560]` and Z `[1536,2560]`. The source hashes are embedded
in `tools/qwen4exp/gdn_projection_gate.py` and were obtained from the qualified
runtime's model source. A newer local model is deliberately rejected.

Memory preflight requires 384 MiB free per rank: 128 MiB for the private packed
weights, outputs and graph resources plus 256 MiB reserve. It is rechecked
before allocation and validation. One combined weight is 20 MiB; retaining
its ND and NZ copies costs about 40 MiB before allocator overhead. The gate
never duplicates all 36 modules (720 MiB per rank for one packed copy each).

## Exactness and lifetime contract

- Original projections call the qualified `_linear` with the borrowed real
  weights. The candidate calls it once with concatenated QKV/Z weights and
  materializes both output slices as contiguous tensors. Their raw sliced
  row stride would otherwise differ from the original projection outputs.
- Four deterministic changing CPU input patterns cover multiple magnitudes,
  signed zeros and FP16 boundary values at 3, 6 and 64 rows. Both FP16 and
  production FP32 source inputs are tested separately. The FP32 arm includes
  non-half-exact values and includes two baseline operand casts versus one
  candidate cast. Inputs are copied to private device storage. Every eager case compares a repeated baseline,
  candidate outputs and unchanged input bytes. All cases must pass before
  any graph is recorded.
- Each shape then gets independent baseline/candidate native graphs and
  private graph pools on a dedicated stream. Each replay changes the input
  and compares both captured outputs with the current eager baseline, using
  FP16 bit patterns rather than tolerances. No recurrent state, KV storage,
  model output, live collective owner or model graph is called or changed.
- Original weight bytes, pointers and available version counters are checked
  before successful cleanup. Inference weights have no version counter;
  their pointer and byte checks still apply.
- Every graph/pool/tensor owner is retained on the worker before capture or
  submission. A launch, capture, numerical or cleanup failure keeps these
  owners plus its exception traceback and prohibits automatic retry. Failed
  private graphs are deliberately not reset by this gate. Parent-controlled
  recovery is required. Successful gates synchronize and reset only their
  own graphs; the model and its graph owners remain untouched.
- This owner is one-shot. Repeating validation or swapping modules under the
  same owner is rejected. This avoids replacing resources whose native
  lifetime has not been reviewed.

The receipt includes alternating synchronized wall timings for the complete
projection/split/materialization group in eager and private captured modes,
labelled by source dtype. Use the FP32 arm for production latency comparison.
These are component measurements, not a model-speed or acceptance claim.
Preserving dtypes does not imply preserving accumulation order: wider GEMM
output geometry can change CANN tiling. Any bit mismatch rejects this exact
candidate, even if a tolerance-based comparison would pass.

## Offline verification

The CPU tests exercise the entire lifecycle with small real CPU matrix
multiplications and mocked graph recording/replay. They check fresh replay
outputs, baseline/candidate comparison, independent pools, caller stream
restoration, signed-zero equality, contiguous split outputs, original-weight
preservation, inference-weight handling, eager failure before capture,
failed graph/pool retention, partial preparation traceback retention, source
and type admission, bounded requests and memory rechecks.

CPU tests alone do not establish Ascend GEMM equality, graph support or
performance. The hardware results below supply the bounded component evidence;
no serving binding should be inferred from this gate.

## Four-rank private hardware result

Receipt: [gdn-validate-v1.json](qsa-resident-v1/gdn-validate-v1.json).
All four ranks passed all 48 comparisons each: four changing inputs × three
row counts × two source dtypes × eager/captured execution. Original weights
were preserved, all 12 private graphs per rank were reset, and the complete
before/after model/collective inventories were identical. The representative
module was `language_model.model.layers.0.attention` on every rank.

Production FP32 component timings, in microseconds per complete projection
pair including operand casts and contiguous output splits:

| Rows | Execution | Baseline | Concatenated | Per-rank latency reduction |
| ---: | --- | ---: | ---: | ---: |
| 3 | Graph | 152.51 | 144.73 | 3.35–7.01% |
| 3 | Eager | 194.08 | 180.24 | 5.12–7.73% |
| 6 | Graph | 153.98 | 147.34 | 3.47–6.91% |
| 6 | Eager | 192.56 | 179.81 | 2.65–8.89% |
| 64 | Graph | 164.40 | 160.51 | 1.21–2.90% |
| 64 | Eager | 191.82 | 188.79 | −6.85–2.46% |

Baseline and concatenated columns are medians across the four ranks' sample
medians. Each rank used three alternating synchronized measurements of five
calls/replays; these are small component samples without confidence intervals.
The range compares each rank with its own baseline, not the two aggregate
columns. Rank 0's three-row graph case was 151.27→146.20 µs; its 64-row eager
case regressed 194.49→207.80 µs. Thus the modest decode gain is reproducible
across these four rank-local cases, while the prefill arm is not a uniform win.

This qualifies one representative layer's private math and replay, not all 36
layers or complete model behavior. All-layer packing would require around
720 MiB per rank for a single packed copy before other lifetime costs. There
is no end-to-end decode throughput result, acceptance test, or all-layer
promotion. Preserve the current model until those separate gates justify the
memory and integration cost.
