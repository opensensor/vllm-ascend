# Experimental Qwen channel-INT8 startup on six 310P chips

This profile changes routed-expert weights and activations from the W4 G128
recipe to per-channel weights and per-token activations. It is a lossy math
experiment. Passing a packing test establishes parity with the INT8 recipe,
not with the original W4 model's logits or language quality.

## Startup bottleneck and fix

The first full-model prototype expanded an entire expert bank on the CPU.
Six workers doing large intermediate operations had no useful progress output;
the first bank had not completed after roughly 15 minutes. Worker stacks showed
CPU weight preparation, not an idle or ready server.

`channel_weight_recipe.py` converts one expert at a time. Per-output-channel
maxima still span the complete K dimension, so batching leaves quantized codes
and scales unchanged. Startup temporarily uses at most six CPU threads in each
worker's affinity set and restores the previous setting even on failure.

Each completed bank can be saved as safetensors on the storage NVMe. The cache
binds the source operand digest, packing layout, tensor geometry, dtypes, and
payload digests. Mismatches abort rather than silently reuse a different bank.
A cache hit skips CPU requantization. This version still loads the original W4
checkpoint and reads its packed operands back to validate that digest; it is
not yet a direct INT8 checkpoint loader.

The next startup improvement is a separately validated, directly loadable INT8
checkpoint with explicit source provenance. That would remove the initial W4
packing/upload, readback, and conversion. It should follow full-model quality
and performance gates before investing in a new checkpoint format.

An additional `stage_w4_on_host` checkpoint override is an **offline candidate**
that has not been enabled on the running profile. It allocates unloaded W4
expert operands on the CPU, retains the original streaming loader and its
inventory validation, and uploads only the converted INT8 banks. This avoids
per-expert W4 host-to-device copies followed by the conversion readback.
CPU tests establish unchanged loader geometry and final recipe, but a fresh
real-weight engine must qualify it before deployment. Its host memory cost is
one staged packed expert shard per worker; do not assume it fits other hosts.

## Progress and readiness

The experimental extension emits synchronization, bank start, expert count,
bank duration, and final allocation messages. Its logger is a child of the
configured vLLM logger. A tools-only logger name does not inherit vLLM's output
handler and can hide informational progress.

The read-only status tool also counts atomically finalized cache files. Old
cache files are reported as inventory, not as new progress. An ETA from bank
completion covers conversion only; it excludes KV profiling, graph capture,
and resident correctness gates. Check all six ranks. Rank 0 finishing does not
make the service ready.

```bash
python3 tools/qwen4exp/startup_status.py \
  --profile-root /srv/ai/src/qwen-channel-int8-tp6-20261011-v3 \
  --cache-root /srv/ai/storage/models/qwen-channel-int8-cache-20261011-v1 \
  --watch
```

`Application startup complete` means the API started. A matching
`bootstrap-ready.json` means that profile's resident inference gates passed.
The status tool distinguishes these phases and reports bootstrap failures.

The checkpoint progress bar can show an hour-long ETA during the first roughly
48 shards. Most subsequent expert payloads are skipped by the existing rank
filter. Retain that filter and measure the slowest worker rather than
extrapolating the early shard rate across all 1,610 files.

## Memory and serving contract

Conversion replaces each resident W4 bank. It removes group offsets and weight
sums and does not retain a second device weight bank. This increases resident
weight bytes, so the 1M context must pass actual KV capacity and workspace
profiling. A configured maximum sequence length alone is not a capacity gate.

The router, shared expert, rank reduction, target FP16 boundaries, and MTP
configuration retain their existing paths. The new routed path quantizes token
activations once before routing and uses INT32 gather indices. Descriptor
storage is bounded for eager tails; admitted graph shapes retain their addresses.
Thermal holds remain at 94 C, with all six chips at or below 85 C to resume.

The paired-W4 profile is the rollback path:
`/srv/ai/src/qwen-paired-cutover-20261011/start-server.sh`. Its retained probes
passed text, fresh and cached images, and thinking. On the recorded prompts it
measured 22.636 seconds cold TTFT for 10,322 tokens and 32.200 tokens/s for a
512-token coding decode. These are W4 reference measurements, not INT8 results.

## Completed model storage migration

The complete model tree now resides at
`/srv/ai/storage/models/serving-models-20261011` on the 4 TB NVMe.
`/srv/ai/models` points to that destination, preserving checkpoint paths and
absolute overlay links. Existing unrelated models on the storage drive were
kept separate.

Verification checked every path, metadata, symlink target, hardlink topology,
and SHA-256 content for each unique inode pair: 5,282 entries, 1,119 unique
files, and 948,466,693,122 unique bytes. An atomic directory/symlink exchange
preserved the logical path. Only the verified old source tree was deleted,
freeing 948,391,321,600 bytes, approximately 883 GiB. The active API remained
healthy and was not restarted. Migration receipts are retained in
`artifacts/qwen38-channel-int8-serving-20261011/`.
