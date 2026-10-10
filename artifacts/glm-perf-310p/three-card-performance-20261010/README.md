# Three-card GLM performance qualification

The real-weight three-card profile uses all six 310P chips as TP2/PP3. Its W3/W4
quantization remains separate from the two-card selective-W2 profile. Both
original exports are preserved on the host's 4 TB NVMe drive.

## Decode graphs: hardware validated

Stage-local `FULL_DECODE_ONLY` graphs capture 1 and 4 rows. Actual successful
replays were counted on **all six workers**, 22 per worker during the real text,
tool-call, and prefix-reuse tests. The counter restores the original method
once the probe ends. Graph allocation measured 0.20 GiB per rank. These graph
settings are now the three-card recipe default; `--enforce-eager` remains
available in the recipe generator for diagnostics.

The identical 17-token tool response had 16 timed decode gaps:

| Mode | Decode ms/token | Decode tokens/s |
| --- | ---: | ---: |
| Three-card eager | 316.0 | 3.2 |
| Three-card decode graphs | 237.6 | 4.2 |
| Graphs + lossless native INT4 storage | 198.2 | 5.0 |

This is 24.8% lower decode latency and 33.0% higher throughput from the unrounded
timings. These are one-request measurements, not a general coding benchmark.
Cold 2,427-token prompt processing remained about 29 seconds. Decode graphs do
not capture prefill. Four concurrent requests returned their distinct expected
answers with zero reasoning tokens; that check verifies correctness, not a
four-request throughput claim.

## Lossless native INT4 storage candidate

W3 signed codes are widened to INT4 nibbles without requantization. This retains
W3 values and FP16 scales while enabling the native W4 unpack branch. It changes
storage width, not model quality precision or Cube operation count.

Both gate/up and down banks in all 20 resident W3 layers are selected. Header-only
memory analysis accounts for the owning pipeline stage instead of charging every
bank to every chip. Additional code storage is 2.53125, 0.84375, and 5.0625 GiB
per chip in stages 0, 1, and 2 respectively. Draft layer 45 is excluded because
this pipeline profile does not load MTP. Context remains 8,192 and prefill remains
640 for the comparison.

Full-shape, 144-expert fused-MoE operator gates passed bitwise equality, including
changed inputs, weights, scales, routes, all-peer routes, and zero route weights.
The actual decode bundle reduced T=2 from 9.023 to 4.517 ms. The prefill bundle
reduced T=640 from 193.208 to 138.961 ms. Most probe routes are local; these gains
are **operator results**; the full-model result below measures their serving benefit.

The compact exporter rewrites only touched shards, hard-links unchanged indexed
shards, preserves scale metadata and attribution, and copies both decode and
prefill bundles. Completed shards can be reused after exact tensor/metadata
verification when resuming an interrupted export. The final manifest is admitted
only after every rewrite succeeds. Full-file SHA-256 verification precedes loading. The first full pass caught one
changed code tensor in `w4-storage-native-rank0-layer014.safetensors`. Its source
file passed its original SHA-256. Rebuilding that replacement from the verified
source and checking every tensor corrected it; all 104 files passed the next
full SHA-256 pass. The cause of the earlier change is unproven. It was never
loaded as a qualified candidate.

## Full-model variant: hardware validated

The prepared variant loaded with zero transformations and zero backup layout
bytes. All six workers replayed real decode graphs (22 per worker). Text, exact
tool arguments, zero reasoning tokens, and 2,304-token prefix reuse passed.
Four simultaneous distinct-answer requests also passed. An 89-token Python
prime-checking response passed 12 functional cases.

Relative to the compact-W3 graph service, the identical tool response reduced
decode from 237.6 to 198.2 ms/token: **16.6% lower latency, 19.9% higher throughput**.
Cold prefill for the same 2,427-token prompt fell from 28,983.9 to 26,047.8 ms,
**10.1% lower**. The tool prompt's prefill fell from 10,100.3 to 8,907.2 ms.
The longer coding response measured 197.3 ms/token (5.07 tokens/s) over 88
decode gaps. Its timing is in `variant/request-timings.json`. All 104 shard
checksums passed again after model loading and real requests.
These are limited workload measurements; they do not establish broad coding
quality or a guaranteed throughput across prompt lengths.

The variant remains live as `three-card-w3`, API PID 2544275, with log
`/srv/ai/storage/artifacts/glm-card-profiles-20261010/three-card-server-r10.log`.
Configured context is 8,192 tokens; this is the validated setting, not a claim
about maximum possible capacity. Prefill chunks remain 640. MTP is disabled.
Weights occupy 24.35, 27.94, and 28.53 GiB per rank in stages 0/1/2. The packed
variant trades extra code memory for less unpacking work. The original compact
three-card export and selective-W2 two-card export are preserved.

## Reproduction

Use the scripts in this directory for the host-local preparation and storage
verification. `plan_w4_storage.py` reads only shard headers. `export_w4_storage.py`
uses six CPU workers and requires at least 32 GiB available host RAM.
`verify_w4_storage.py` measures SHA-256 for every shard and runs complete storage
admission. `qualify_variant.py` submits real requests and checks replay counts.
The launcher is in `../hardware-card-profiles-20261010/launch_profile.py`.

```bash
python launch_profile.py --cards 3 --attempt 10 --decode-graphs 1,4 --debug-rpc \
  --model /srv/ai/storage/models/GLM-5.3-Flash-three-card-W3-in-W4-native-candidate \
  --storage-receipt /srv/ai/storage/artifacts/glm-card-profiles-20261010/w4-storage-full-check.json
```

MTP remains disabled for PP3; its current implementation requires one pipeline
stage. Longer context capacity and broader model accuracy require separate gates.
The current changes passed 85 focused CPU tests. No model has been uploaded to
Hugging Face by this performance qualification.
