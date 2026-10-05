# Named sparse QSA gather, paired 310P experiment

The baseline and zero-fill gather coexist under separate operator names in one
process. The candidate clears fully masked groups locally rather than reading
fallback cache data. Dense tiles retain the existing gather. No serving default
changed, and these measurements do not establish TTFT or decode throughput.

## Paired operator results

Geometry: 64 queries, two KV heads, head dimension 256, 512 selection slots,
16,800 physical cache pages, and a 4,096-entry block table. Output allocations
are reused. Event medians use six paired trials of ten queued calls; wall
medians use twelve paired trials with synchronization. Trial order alternates.

| Active groups | K baseline | K zero-fill | V baseline | V zero-fill |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.2476 ms | 1.0279 ms | 0.9350 ms | 0.7873 ms |
| 256 | 1.1947 ms | 1.0622 ms | 0.8213 ms | 0.7778 ms |
| 384 | 1.1652 ms | 1.1060 ms | 0.7760 ms | 0.7713 ms |
| 512 | 1.1337 ms | 1.1571 ms | 0.7361 ms | 0.7684 ms |

The experiment selects a cutoff of 256 groups: both orientations improve by
at least 5% through that count. Fully selected tiles would regress with a
blanket replacement, so retain the baseline there. Valid gathered values match
exactly, and all masked candidate elements are exactly zero.

## Full synthetic QSA attention

One 2,560-query attention call, 24 query heads and two KV heads. First-chunk
positions are 0–2,559; the later dense chunk uses 2,560–5,119. Five paired wall
trials follow warmup. Both candidates use identical inputs within each pair.

| Chunk | Parallel gather | Baseline | Selective | Time change | Candidate tiles |
| --- | :---: | ---: | ---: | ---: | ---: |
| First | No | 194.294 ms | 189.184 ms | -2.63% | 16/40 |
| First | Yes | 192.566 ms | 188.835 ms | -1.94% | 16/40 |
| Later | No | 190.233 ms | 190.465 ms | +0.12% | 0/40 |
| Later | Yes | 188.363 ms | 188.490 ms | +0.07% | 0/40 |

All attention outputs are bitwise identical within each comparison. The later
chunk invokes the same baseline functions, so its small differences are timing
variation. Parallel gather is the selected serving configuration. Raw samples
and hashes are in [result.json](result.json).

**Dispatch remains experimental.** The benchmark knows monotonic causal group
counts on the host for one synthetic request. It uses tensor storage offsets
as host metadata to identify each slice, without reading device counts. Real
serving dispatch must account for multiple requests, prefixes, causal tails,
and selection masks using scheduler-owned lengths. This dispatch cannot simply
be copied into the model.

## Package and correctness

The separately named `QsaGatherValueNzZeroV310` is installed at
`/srv/ai/src/qsa-selective-gather-20261005/opp/vendors/qsa_selective_gather_310p_transformer`.
The supplemental Torch binding has SHA-256
`355d9e407ce0cb8d9284fcfa1d1fafed65323531c1b6b04c2e68db9ed7866401`.
Both [32 gather tests](named-regression.log) and the paired attention parity
gate passed after correcting the isolated vendor metadata.

The reused build directory emitted stale baseline entries in the candidate
vendor's JSON configuration. With that vendor first, CANN sought the baseline
binary inside the candidate package, where it did not exist. The isolated
[normalizer](normalize_vendor_metadata.py) restricts both metadata files to the
new operator; [metadata-fix.json](metadata-fix.json) records before/after hashes.
The baseline API/schema is identical, allowing the old-only operator-info
entry to be renamed. This repair is specific to this experimental package;
it does not fix the general incremental package builder.

## Reproduce

Stage the saved source tree and scripts into the remote experiment directory.
Build the snapshot operator using `csrc/build.sh --pkg --soc=ascend310p
--vendor_name=qsa_selective_gather_310p --ops=qsa_gather_value_nz_zero_v310 -j2`,
then install its package into the experiment's `opp` directory. Normalize the
installed metadata before combining it with the baseline stack:

```bash
python normalize_vendor_metadata.py \
  opp/vendors/qsa_selective_gather_310p_transformer --receipt metadata-fix.json
python build_probe_binding.py
bash run.sh regression
bash run.sh benchmark
```

The runner selects one otherwise idle 310P and explicitly includes the
qualified coherent, embedded, and retained vendor packages. No model weights
or server are loaded by these tests. A serving experiment needs the named
binding and vendor available at worker startup. The current resident Qwen
server retains the qualified baseline operator stack.
