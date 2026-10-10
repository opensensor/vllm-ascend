# Six-chip Qwen: eliminate duplicated shared and dummy-head execution

Later hardware results, including the GDN prefill corrections required by TP6,
are in the [six-chip qualification report](../qwen38-six-chip-hardware-20261009/README.md).
The offline status and estimates below describe this earlier delivery.

These are offline fixes on top of `b08df99ff`. The live image-enabled TP4 service
has not been changed. The generated [TP6 profile](tp6/profile.json) supersedes
the earlier replicated-shared/padded-execution candidate. Images remain enabled,
MTP remains disabled, and the requested six 262,144-token sessions are unvalidated.

## Shared expert

`shared_expert_execution=tp_sharded_uneven` places disjoint trained intermediate
channels on each rank: 107/107/107/107/106/106, totaling 640. Gate/up rows and
down columns load directly into rank-local parameters. The scalar gate stays
replicated. Each shared partial joins the existing routed MoE all-reduce;
there is no additional shared-expert collective or activation broadcast.

Logical shared projection storage falls from six full copies, 56.25 MiB in total,
to 9.375 MiB in total. Each rank keeps about 1.55–1.57 MiB instead of 9.375 MiB.
These figures exclude the small scalar gate and physical NZ alignment. Each
trained channel is computed once across ranks, instead of six times. This is
source-derived ownership/storage arithmetic, not measured DMA, heat or speed.
FP16 GEMM/reduction order changes, so real-weight quality gates remain mandatory.

## GDN live execution inside uniform cache pages

`gdn_head_partition=padded_compact` balances trained key groups as 3/3/3/3/2/2.
The live value heads are 9/9/9/9/6/6, totaling the trained 48. Projection,
convolution, gating, WY, recurrence, normalization and output use these live
shapes. Dummy-head weights and computation are absent. Checkpoint heads and
vocabulary remain unchanged. Strict and original padded policies are preserved.

All ranks still report the same nine-head/1,920-channel cache descriptor because
upstream `vllm/v1/core/kv_cache_utils.py:get_kv_cache_configs` explicitly rejects
different specs for the same layer across workers. Compact execution uses
zero-copy views within those pages. Convolution history is packed densely at
the start of each page; recurrent heads use a dense inner matrix and preserve
the allocator's outer page stride. Views are refreshed when cache storage changes.

Prefill gather/mask/scatter and recurrent decode receive only live heads. On each
of the two smaller ranks, live FP32 recurrent state is 393,216 bytes rather than
589,824 bytes per layer/slot. Across ranks, live recurrent payload falls from
54 to 48 heads, eliminating 11.1% of the padded logical payload. The earlier
12.5% aggregate allocation overhead remains reserved. Prefix checkpoint copies,
CoW and page clearing can still touch that reserve; no reduction in their
physical bus traffic is claimed. Removing reservation needs a separate
heterogeneous-cache allocation protocol, not a falsified uniform spec.

The old standalone native state-layout resource assumes dense pages and is
rejected for smaller compact head views. The candidate retains reference state
IO and reference WY. Its native convolution/recurrent operators must come from
a coherent OPP supporting strided state pages. Existing adapter/API/kernel
sources propagate the real page stride; queued hardware tests verify the
installed package rather than assuming source and deployed binaries agree.
Old padded and new compact pages use different encodings and cannot be hot-swapped
with retained caches. Use fresh workers and empty caches at a later cutover.

## Validation and remaining gates

Actual production loader/shared methods reconstruct the full FP64 shared expert
for one, three and 65 tokens. Tests verify complete channel coverage, reload
placement, malformed-shape rejection without mutation, and exactly one MoE
collective containing the shared partial.

Actual GDN methods run stateful prefill then decode against the full-head CPU
reference, with warm/cold requests, nonadjacent slots, initial/final states and
NaNs in unused storage. Live cache alias/stride/rebinding and untouched slack
are checked. Full projection-derived output/state comparisons use a fixed
1e-12 FP64 reference bound because GEMM batch shape changes can differ at the
last bits; this is not an NPU acceptance tolerance. Mock native handoffs check
live shapes and strided pointers, not actual kernel execution.

Final results are recorded in [the validation receipt](validation/receipt.json).
The supported regression suite passes 223 tests with five skips. A broader run
passes 239 and fails four existing lifecycle imports; the unchanged baseline
reproduces those exact four failures because its CPU `torch_npu` stub lacks
`float4_e2m1fn_x2`. Changed-file hooks pass. Repository-wide formatting retains
unrelated baseline failures; formatter changes were restored only in its checker.
Nine hardware gates skip by default, including the two new state-view tests.
No new hardware inference or timing result is available. Two queued real-OPP
tests compare compact convolution/recurrent views against dense state, including
page alignment slack and untouched slots. Changed-input graph replay, six-rank
real-weight/model/image accuracy, memory peaks, request cancellation/prefix CoW,
transfer/barrier attribution and sustained thermal qualification remain required
before cutover. The 94C hold, all-chip 85C resume and independent cutoff remain
in control; smaller weight inventories do not establish thermal improvement.

See [the cutover runbook](RUNBOOK.md). No server cutover is performed here.
