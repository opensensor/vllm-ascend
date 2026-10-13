# GLM sidecard 16 MiB logical slots

This study expands the cold expert window from 512 to 1024 rows while retaining
32-row CUDA math, the same weights, and the same full Code prompt. The CUDA
input/result slots are 16 MiB contiguous allocations. Each exports two disjoint
8 MiB NVIDIA DMA-BUF regions. Ascend transfers those regions sequentially and
publishes the completion sequence only after the complete payload returns.

This is a 16 MiB logical contract carried by bounded transfers. The installed
UAPI 5 bridge still caps each registration and ioctl at 8 MiB. No kernel module,
card driver, clock, ECC setting, or checkpoint changed. One RTX PRO 6000 and
one Atlas 300I Duo on physical Ascend chips 2 and 3 perform inference. Physical
0 and 1 remain reserved for unrelated work.

## Transport and ownership

`pcie.segmented` borrows allocation regions with independent lease counters;
every region lease also retains the complete CUDA or HBM owner. Each region
uses its own private kernel endpoint, with at most two persistent GPU mappings
(input and result). The separate completion-control endpoint remains unchanged.
An uncertain piece retains the original owner and prevents cleanup. All region
mappings must close before exported descriptors or allocations can be released.

The CUDA IPC input descriptor identifies the complete 16 MiB owned allocation.
Result registration sends both 8 MiB descriptors before the control descriptor.
All parties derive those sizes and counts from the admitted cold window; mixed
capacities fail identity/descriptor checks. Older completion consumers fail
admission before owner allocation or future CUDA waits.

- [Raw qualification](raw/transport-probe.json): 45 repeated cases through
  1024 rows, including 511/512/513 and 1023/1024 boundaries.
- Both ranks preserve exact ordered FP32 sums, untouched GPU result tails,
  and poisoned host guards; mappings close cleanly.
- A 1024-row return uses exactly two 8 MiB DMA operations. Its input plus
  metadata and invocation record uses 8,454,208 bytes, also crossing that
  boundary. The final 8-byte sequence publication follows both return pieces.
- [Native consumer admission](raw/native-capacity.json): separate control
  admits 1024 rows; packed control remains limited to 64. Oversized 1025-row
  publication fails before kernel submission.

The raw probe coexisted with the qualified 8 MiB service. It completed and
returned the bridge refcount to that service's four references before model
ownership changed. The 16 MiB model uses six references: two region endpoints
plus one control endpoint per Ascend rank.

## Matched model measurement

One loaded runtime provisions 1024 rows and compares `--windows 512 1024`.
The baseline therefore also has 16 MiB logical slots, but transfers only the
regions required for its 512-row batches. Both paths reset model state, bypass
prefix snapshots and the API queue, and record zero cached prompt tokens.
CUDA attention, routing, hot/shared MLP, recurrent and KV state retain their
qualified 32-row arithmetic. Decode graphs are unchanged.

The complete Code request retains all 23 tools and all 23,820 token IDs. No
compact agent, shortened prompt, truncation, or warm cache is used. Timed
prefill excludes logit observers. All 64 teacher-forced full-vocabulary vectors
must match bitwise; decode benchmark timings include those observers.
The private request and environment files are not published.

The independent 2048-token gate compares every original CUDA chunk boundary
and all 64 output steps against the serial 32-row path and saved 8 MiB reference.
It passed with zero maximum logit error before timing began.

## Cold model results

| Prompt | 512 rows / 8 MiB payload | 1024 rows / 16 MiB payload | Time saved |
| --- | ---: | ---: | ---: |
| 2,048 tokens | 10.564s | 10.100s | 4.4% |
| 8,192 tokens | 43.700s | 42.891s | 1.9% |
| 23,820 tokens | 134.838s | 124.337s | 7.8% |

The full Code request improves from 176.7 to 191.6 prompt tokens/s. All six
paths have zero cached tokens, zero maximum logit error, and bitwise-equal
logits for every observed output step. Those hashes also match the preceding
qualified 512-row study. Smaller prompts show modest gains; 16 MiB improves
cold batching and does not change decode math.

This is one matched run per path and fixture, not a median or throughput
guarantee. See [benchmark](probe/cold-benchmark.json),
[prefill gate](probe/prefill-parity.json), [summary](summary.json),
and [updated design PNG](design.png).

## Retained worker spans

For the full Code prompt, the mean native host span per row falls from
157.0 to 129.6 microseconds on rank 0, and from 150.9 to 124.1 on rank 1.
The closed worker receipts retain all 690 complete 512-row calls and all 345
complete 1024-row calls per rank. The calculation isolates the final matching
cold run between single-row decode runs before selecting complete windows.
This excludes earlier 8K fixtures and the final 268-row partial window.

Native spans include submission and completion synchronization. Ranks and
CUDA overlap, so these values do not form an additive latency waterfall. No
new device timing barriers were inserted. See
[worker span summary](worker-span-summary.json).

## Deployment

The 1024-row configuration won the full Code comparison. It uses the fresh
1024-row completion consumer, 16 MiB logical slots in two 8 MiB regions,
separate control, and the qualified 32-row CUDA arithmetic. Context admission
remains 65,536 tokens; these experiments do not establish full-length 65K
performance. The API remains `http://192.168.53.187:8000/v1`, model `glm-sidecard`.
All serving startup gates passed. The actual cold 2K API request produced first
text in 10.034s, with zero cached tokens, then decoded at
11.74 tokens/s. [API receipt](api-cold-2048.json) and
[health](health.json) record this serving check.

The [frozen source archive](qualified-source.tar.gz) matches all 112 runtime
Python files plus the CUDA completion source byte for byte. Native helper
sources remain frozen from the preceding qualified service. Private
environment files and model weights are excluded.

## Validation

All 1,260 standalone sidecard unit tests pass, including region geometry,
full-owner leases, incomplete descriptor registration, preflight rejection,
second-piece uncertainty, ordering before completion publication, causal window
tails, and legacy consumer rejection. Actual transport and full-model gates
run on the assigned hardware; numerical qualification is separate from speed.

Full-repository manual checks run in an isolated worktree. Seven legacy hooks
fail; unrelated auto-fixes are discarded. Changed-file manual checks pass.
