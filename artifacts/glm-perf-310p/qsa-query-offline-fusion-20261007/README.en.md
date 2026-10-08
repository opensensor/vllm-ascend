# GLM fusion delivery summary (US English)

This delivery includes the accumulated GLM runtime, permanent weight layout,
native low-bit MoE kernels, MTP/cache fixes, resident hot-swap tools, tests,
and experiment evidence. It also publishes the 22 earlier local GLM commits.
Unrelated Qwen work remains outside this delivery.

The earlier 310P gates passed 48 fused QSA metadata cases. Archived four-rank
decode traces show AI-CPU FloorDiv tasks falling from 36 to zero and AI-CPU
casts from 14 to two after query cast binding. These are operator measurements;
the paired prefill measurements were essentially flat. They do not establish
10 tokens per second.

The new offline expert fusion keeps each route in native Cube column order
through FP16 rounding and stable FP32 weighted reduction, then rearranges
columns once per completed token. Both the decode variant (v952) and the paired
prefill producer/reducer (v953) compile for 310P. The route workspace size stays
the same. A full-MoE comparison harness covers W2/W3/W4, A4/A8, and
2/8/17/640-token graph replay with changed inputs and alternating timing.

The vector FP16-to-BF16 query converter (v954) also compiles. Its reversible
binding covers both target and draft indexers. The CPU oracle checks every
non-NaN FP16 bit pattern; hardware instruction behavior still needs testing.
Resident admission requires complete matching hardware gates and a unique
bridge namespace. Diagnostic RPC guards return per-rank error receipts.

The focused CPU suite passed 257 tests. The broader suite initially passed
1,511 tests, with failures from missing local upstream/NPU modules, stale
archived KDA source hashes, and a CLI expectation affected by the new flag.
The CLI expectation was corrected, and the historical KDA patch now uses its
exact archived prepare header. The final compatible CPU suite passed 1,508
tests, excluding three files that require locally unavailable upstream/NPU
modules. Source and Markdown checks passed. Checking all historical snapshots
also reports existing script formatting and technical-term spelling issues.
Historical experiment snapshots are preserved as recorded; their hashes must
not be changed merely to satisfy formatting checks.

No device calls, inference requests, recovery attempts, or hot swaps ran after
the user requested offline work. The service encountered a profiling RPC
protocol problem earlier; this delivery does not claim it is healthy.
All three new candidates remain hardware-unvalidated and undeployed.

Last verified context settings: 311,040 tokens including input and output,
640-token prefill chunks, four concurrent sequences, TP=4, and MTP=1.
Concurrent requests share KV cache capacity.

See the [Chinese report and pending validation commands](README.md).
The evidence archive contains 163 files with individually verified SHA256
digests. Original SDK databases remain on the remote host; exported CSVs,
attribution, receipts, and candidate bundles are included.
