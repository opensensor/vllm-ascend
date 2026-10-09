# T8: full candidate composition CPU validation

Added 33 CPU tests for the coordinator's explicit streaming candidate, without
editing its implementation or installing runtime hooks. Scoped Ruff and formatting
passed. The final run passed all 33 tests with 20 existing environment warnings in
10.94 seconds; see `tests-final.log`.

The tests execute complete T3 token packing/local gather, T5 builtin-FP16 activation,
packed-hidden handoff, bounded output-column projection/finalization, and actual
T6 `streaming_prefill` scheduling through injected CPU callbacks. An independent
per-token/per-route reference bypasses dispatcher ordering, gather, windows and
candidate orchestration. NumPy quantization supplies a separately implemented
reference for the activation values. The CPU fixture shares the standard Torch
FP16 activation operation with the candidate; it does not claim an independent
implementation of the native builtin activation operator.

Three complete EP4 comparisons partition global expert IDs across four distinct
128-expert shards and compare whole-MoE output for TP-sharded, replicated and absent
shared experts. Each rank processes a 129-token batch as 128 plus one tail token,
preserves complete 2,560-column output and three-tile output windows, and reduces
the independently computed four rank contributions in fixed order. Additional
fixtures cover all-peer rows, local group ends, token quantization once per chunk,
hidden packing once, routing once per complete batch and changing output windows.

Fallback tests confirm that capture/sparse/W8, width, activation/finalizer policy,
TP, bank dimensions, local expert count and shared placement mismatches do not
touch candidate callbacks. Partial packing/projection/window/reduction failures
poison the candidate and never fall back or silently resubmit the baseline.

Admission tests bind configuration, resource inventory, plan and exact worker rank
identities before hook-map creation. The five-target composition has one owner
and rejects stacked MoE/GDN wrappers. The resident factory rejects missing admission
before runtime imports. Native resource preparation rejects mismatched admission
before `torch_npu` import, native library registration or kernel loading. These
tests use an explicitly labeled fake admission and never claim hardware admission.

The projected CPU fixture uses declared toy per-expert arithmetic with real
production-shaped activation/output widths; it does not simulate complete packed
W4 weights or prove native G128 correction, hardware stream order or HCCL behavior.
Those require the separate native kernel tests and future real-weight T9 gates.
No NPU adapter, NPU allocation, native device library, server query or runtime
mutation occurred. Complete model/image/quality/thermal/performance qualification
remains pending. The interim EP4 log includes two failures caused by concurrent
Admission API evolution; the fake interface was aligned and final validation passes.
