# Qwen streaming follow-up and six-chip candidate

The image-enabled TP4 baseline remains the live control. This work implements a
new offline streaming projection and an opt-in six-chip Qwen model/load profile.
It does not switch, pause, resume or restart the service. A read-only check during
this work confirmed the existing thermal controller paused at 94C and held for
all chips to reach 85C, as requested by the user.

## Streaming implementation

The versioned v2 projection retains v5 source and its rejection evidence. It uses
M32/N160, a projection-only 248,064-byte UB arena, three activation-metadata
slots, and the existing two physical operand/product slots. Unused future
quantizer/gate scratch is excluded from this stage's contract. The output arena
stages one FP16 metadata bank only before row execution; it never aliases a live
projection/store. Metadata converts in three bulk casts per expert/output tile,
with weight-sum scaling once, and remains FP32 across row batches.

Sparse tails use row-oriented vector correction; larger row tiles use output
strips with row repeats. Every FP32 correction and ascending G128 addition keeps
its original per-element order. The M32 loader explicitly traverses both M16
blocks per limb. Expert ends use a 32-boundary DMA cache, with scalar reads for
at most three unaligned tail entries. No boundary DMA overreads the E-element
allocation, including six-chip banks of 85/86 experts.

The new schedule submits the next operand load after the current Cube launch,
then consumes the previous product and reads back the current one. Three metadata
slots eliminate producer j+1 / consumer j-1 aliasing. CO1 stays single-owner until
readback acknowledgment. A dependency DAG rejects unordered memory ownership
and impossible future-consumer waits. Delayed, out-of-order CPU completion tests
exercise that model; they do not establish actual device overlap.

N160 gives eight gate/up tiles across eight cores. Down N2560 uses two windows
of 1,280 columns instead of three N128 windows. Peak routed allocation rises
from v5's 50 MiB to 62.5 MiB, still below a full 125 MiB routed output. Total
routed payload and output-copy traffic remain; these are allocation bounds,
not measured bus bytes. The epilogue rejects a window/resource tile-width mismatch.

The append-only bundle builder supports an explicitly selected v2 contract and
new entrypoints, plus the existing local-route gather resource. The default v1
build path survives. V2 rejects native WY and retains the reference path. The
standalone ABBA harness requires explicit execution, reads all sensors outside
timing, aborts at 90C, and never contacts a service. No NPU benchmark was run.

## WY disposition

The original native WY downstream failure remains unresolved and blocked from
promotion. New diagnostics preserve exact Q/K/V, gates, beta, normalization
identity, initial state, all five preparation stages and both output/final states.
Each variant receives independent cloned inputs and initial state. A failure in
output comparison no longer discards final-state evidence. Exceptions and input
mutation reject the trace while preserving original inputs.

Independent FP64 forward substitution, triangular-system comparisons and complete
chunk output/state equations agree with a token-by-token recurrence, including
chunk boundaries and nonzero initial state. The native hardware failure cannot
be attributed or declared repaired without its missing intermediate tensors.
No tolerance was relaxed and no failing WY kernel was enabled.

## Qwen on six chips

The new explicit checkpoint policy `gdn_head_partition=padded` allocates three
key and nine value heads per rank, padding the trained global 16/48 geometry to
18/54. Ranks 0–4 own three trained key groups each; rank 5 owns one, plus two
zero input/output groups. Trained config head counts remain unchanged. Loader
placement pads each Q/K/V projection, convolution, Z and gate parameter, and
slices row-parallel output columns. Reloads clear padding. Shared per-head norm
weights remain replicated.

The constructor and cache descriptor both use 1,920 convolution channels and
nine FP32 recurrent heads per rank. Actual method-body CPU checks exercise
constructor, placement and cache shapes together. Independent FP64 recurrence
checks preserve every trained head's output and initial/final state exactly.
These checks do not qualify native state kernels or numerical all-reduce order.

Existing expert partitioning assigns 86 experts to two ranks and 85 to four,
covering all 512 exactly. QSA assigns four query heads per rank and replicates
one KV head within each three-rank group. Vocabulary padding aligns both embedding
and LM head to 192 rows: 248,448 padded rows, 41,408 per rank, retaining the
trained 248,320-token vocabulary. Default strict head placement and TP4 padding
remain unchanged unless the explicit candidate policy is selected.

The candidate uses replicated shared experts and a data-parallel vision encoder;
vision geometry and preprocessing remain unchanged. It explicitly rejects MTP,
PP greater than one and external KV transfer. Those paths have additional
partition/state contracts. The six-chip candidate is capacity-oriented and does
not promise better decode throughput than the MTP2 TP4 baseline.

[The generated profile](tp6/profile.json) reports only routed backbone codes and
native FP16 metadata: roughly 10.33 GiB for an 86-expert rank versus 15.38 GiB at
128 experts. It excludes all dense/vision/embedding/PLE parameters, caches,
prefix states, graphs, HCCL, scratch and allocation/load reserves. The profile requests six sessions at 262,144 tokens with images enabled, but
marks these limits unvalidated. No whole-rank fit or thermal benefit is claimed.

[The runbook](RUNBOOK.md) explains isolated checkpoint overlays and deferred
hardware gates. An overlay links canonical checkpoint assets and writes a new
config; it does not rewrite packed weights or modify the trained source. It
requires the canonical safetensors index, not a TP4-specific prepacked export.
Profile generation and all implementation validation are offline.

## Validation boundary

The evidence receipt and logs under `validation/` record the final checks.
CPU native stubs validate actual body arithmetic/layout/bounds and matched event
pairs, but synchronously execute commands and cannot prove NPU ordering, CAST_NONE
rounding, speed or thermals. The separate asynchronous dependency model validates
its stated ownership graph, not SDK scheduling. Contained host CANN compilation passes; it does not load a bridge or execute
a kernel. The final CPU regression suite passes 631 tests, with four checks
skipped because a checkpoint is not mounted locally. Seven queued hardware
tests skip by default. Changed-file hooks pass. The full repository format check
has existing unrelated lint, spelling, import and formatting failures; its log
is retained. No actual device numerical, timing or thermal qualification is claimed.

Before promotion require hardware kernel parity and changed-input graph tests,
real-weight ABBA, complete model/image/cache/CoW/cancellation gates, six-rank
collectives, exact rank memory accounting and sustained thermal qualification.
Keep the faster baseline if those gates fail. Neither the v2 projection nor TP6
profile has been installed in the running service.
