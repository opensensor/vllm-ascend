# GLM two- and three-card hardware qualification

## Storage and the 10% transfer

The 10% was progress on a redundant network copy of the entire two-card export
(133.97 GiB), not model loading or qualification. That copy was stopped. The
remaining two-card shards were reconstructed from existing host-local native
codes, scales, and identical dense tensors. Original exported safetensors headers
and full-shard SHA256 values make each result byte-identical to its export;
there is no quantization or tensor-layout conversion in this reuse step.

Both exports are complete and fully checksum-verified on `/srv/ai/storage`,
the host's Crucial T705 4 TB drive. The original W2 checkpoint remains intact.
Eighteen identical dense shards share inodes between these two exports, saving
18,211,841,564 bytes. Storage receipts deliberately describe storage checks only;
serving evidence is separate below.

| Profile | Model directory under `/srv/ai/storage/models` | Export size | Topology | Runtime |
| --- | --- | --- | --- | --- |
| Two cards | `GLM-5.3-Flash-two-card-native-candidate` | 133.97 GiB, 190 shards | TP4 / PP1 | MTP1; decode graphs 2 and 8 |
| Three cards | `GLM-5.3-Flash-three-card-native-candidate` | 144.94 GiB, 104 shards | TP2 / PP3 | No MTP; eager qualification |

Two cards preserve the selective W2/W3/W4 policy: 22 W4, 8 W3, and 13 W2
expert layers including the draft. Three cards use 22 W4 and 21 W3 expert
layers; W3 was generated from the higher-precision source, never widened from
W2. The disabled draft layer is retained on disk but not loaded in PP3.
Both use `cube_n128_k256_v1`, FP16 stored scales with
`fp16_storage_fp32_compute_v1`, and A4 native math.

## Hardware changes

310P pipeline tensor dictionaries now use pairwise HCCL AllGather on INT8 byte
views. CPU messages carry shapes, dtypes, and control values; activation values
stay on NPU and undergo no float arithmetic during transport. Receivers own a
buffer of twice the incoming payload bytes. Subsequent packets cannot overwrite
previously returned tensors. The upstream synchronous tensor-dictionary seam
carries hidden states and all deferred mHC residual, post, and combine tensors.

Initialize the active adjacent-stage communicators before weight loading and
memory profiling. Late initialization previously failed to allocate 2,122,720
bytes after KV cache allocation. The qualified PP3 bootstrap preserves the
original three-pair process-group creation order, but initializes only the two
linear edges, with CPU barriers and NPU synchronization between them. The
first/last closing group stays dormant. Removing that group or initializing it
also failed in separate transport experiments; other hardware/topologies need
qualification before changing this bootstrap.

All six ranks passed actual vLLM coordinator byte checks at 1, 17, and 640 rows,
including FP16 hidden states, FP32 mHC tensors, NaN payloads, signed zero, empty
tensors, and retained packet storage. This proves transport bytes, not complete
model equivalence. See [pipeline gate](pipeline-byte-gate.txt).

The 310P broadcast adaptation also translates global source ranks to subgroup
ranks and preserves the `group_src` convention. This prevents an out-of-range
root when a subgroup contains global rank 4.

Every native-loader checkpoint tensor now receives private CPU staging before
NPU upload, including dense parameters and expert scales. During earlier loads,
a file-cache checksum differed from the manifest but returned to the correct
value after evicting that file's cached pages, without rewriting the disk file.
The driver-level cause has not been proven. Private staging preserves tensor
bytes/dtypes and eliminated the observed fault. The previously affected two-card
shards still matched SHA256 after real requests, without cache eviction. This is
startup copying, not online unpacking, repacking, or quantization. It costs extra
host RAM; qualify host-memory capacity on smaller systems.

## Serving evidence

Qualification uses real weights, automatic memory profiling, context 8,192,
640-token prefill chunks, four sequences, prefix caching, and aligned recurrent
state. No fixed-cache or dummy-weight admission bypass is used.

Two-card attempt r4 passed arithmetic, exact tool arguments, and repeated-prefix
accounting. The repeated prompt reused 1,920 tokens. A 40-token tool request had
4.605 s prefill and 4.802 s across 39 decode gaps, about 8.1 tok/s. Weight memory
was 34.64 GiB per rank; peak activation 0.46 GiB, graphs 0.42–0.43 GiB, and KV
cache 3.41–3.50 GiB. These are short-request measurements, not a general coding
quality or long-context performance claim.

Three-card attempt r6 loaded and executed requests across all six chips. An
answer test initially exhausted 64 output tokens before producing content.
A diagnostic with 128 output tokens returned exactly `391`, but spent 94 tokens
reasoning despite a zero budget. It also reused 2,304 prompt tokens. The tool
request passed. The missing reasoning configuration in the 310P input-batch
rebuild caused the budget to be ignored; this is separate from sharding or
weight quantization. The prepared runner fix is used in subsequent qualification
and is recorded by source SHA. The completed fix forwards the configuration
through both runner rebuilds and the actual 310P constructor; the regression
test exercises both boundaries.

Three-card attempt r8 passed both stricter serving tests in 56.02 s. Each
answer was `391` with zero reported reasoning tokens; the repeated prompt
reused 2,304 tokens. Cold request elapsed time was 30.459 s, repeated request
10.324 s. The tool request produced exactly `multiply({"a":17,"b":23})`: 10.115 s
prefill and 5.056 s across 16 decode gaps, about 3.2 tok/s. See
[serving results](three-card-e2e.txt) and [launch receipt](three-card-server-process.json).

PP3 r6 weight memory was 21.81 / 27.09 / 23.47 GiB per rank for stages 0/1/2.
With initialized communications included, KV allocations were approximately
12.9 / 8.6 / 11.5 GiB. The allocator reported 198,117 aggregate cache tokens,
while the enforced request context remained 8,192. This reported capacity is
not proof of a tested long-context limit. Single-request eager decode was about
3.1 tok/s. PP3 does not reduce single-request layer dependency, and this profile
also disables MTP and graph replay, so this is not a precision-only comparison.

Two-card attempt r5 also passed both stricter tests with zero reasoning tokens,
MTP1, and both captured decode graphs. The repeated prompt reused 1,920 tokens.
Cold request elapsed time was 28.42337792100443 s and repeated request
8.57880387500336 s. The exact 17-token tool call had 4.634 s prefill
and 1.942 s across 16 decode gaps, about 8.2 tok/s. MTP acceptance in the
corresponding logged interval was 9/10 (90%). See
[final two-card tests](two-card-e2e.txt).

The final running qualification service is two-card attempt r5, API PID
2370886, on host loopback port 8001, context 8,192, prefill 640, chips 2–5.
The three-card r8 service was stopped after passing its gates; devices 0–1
are free. Its model export remains on disk. Server logs are
`/srv/ai/storage/artifacts/glm-card-profiles-20261010/two-card-server-r5.log`
and `three-card-server-r8.log`. The SSH forward on local port 18701 remains up.

## Runtime and reproduction

The isolated host source is
`/srv/ai/src/glm-card-profiles-20261010-5652e3a81`, based on commit `5652e3a81`
plus the changes in the launch source receipts. Upstream vLLM is host commit
`3ab5dda29`, using `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`.
Both CANN and NNAL environments must be sourced. Preserve CANN's Python path:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=/srv/ai/src/glm-card-profiles-20261010-5652e3a81:/srv/ai/src/vllm-opensensor:$PYTHONPATH
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  /srv/ai/storage/artifacts/glm-card-profiles-20261010/launch_profile.py \
  --cards 3 --attempt 9
```

Choose a fresh attempt number and an idle port/device allocation. The launcher
holds a lock, checks port 8001, requires the SHA-qualified storage receipt, and
preflights auxiliary bindings and ATB before allocating model weights. It uses
chips 2–5 for two cards and 0–5 for three cards. Its service binds loopback.
Local qualification accesses it through an SSH forward on port 18701.

An older host extension lacks three auxiliary registrations. The worker extension
loads only missing operators from the SHA-qualified host libraries listed in
[the binding manifest](card-bindings/manifest.json). Their source paths are in the
manifest; binaries are not checked into Git. The current fork contains their C++
registrations, so a freshly built complete extension should replace this host
compatibility shim. The legacy OPP and host library stacks are preserved in the
process receipts, rather than silently rebuilt or reordered.

HCCL uses reserved ports 54000–54031. The temporary host reservation and previous
value are in [the port receipt](hccl-port-reservation.json). Keep the reservation
while this service runs; restore only this addition when all trials stop.

The isolated checkout also requires CPU reference modules currently imported
from `tests.ut.deepseek_w2` and `tests.ut.qwen38_1m.reference`. They were supplied
for host qualification. Wheel packaging and a clean-host install remain release
work; no Hugging Face upload was performed.

## Limits and next gates

These are basic text, tool-call, prefix-reuse, transport, and memory-admission
gates. Golden-logit equivalence, image requests, broad coding quality,
long-context operation, multi-request throughput, PP graph capture, and PP MTP
remain unqualified. Retain the two-card W2 profile; do not replace it with the
three-card export. Measure useful PP concurrency and qualify graph/transport
capture before claiming a decode speedup from the third card.

## Offline validation

Ninety transport, loader, reuse, export, and model-pipeline unit cases passed.
Eight reasoning-budget cases execute the actual 310P constructor and both
runner rebuild call sites; they passed for enabled, disabled, missing, and
non-speculative configurations. Seventeen existing coordinator registry
regressions passed. Scoped manual pre-commit checks, including Markdown lint,
passed without formatting unrelated work in this shared checkout.

The required `bash format.sh ci` was also run in an isolated worktree containing
only the staged GLM changes. It failed on existing repository formatting,
spelling, and forbidden-import issues, including `re` in the unchanged Qwen
weight mapper. Those unrelated automatic formatting changes were not applied
to the shared checkout. See [full-check receipt](repository-format-check.json).
