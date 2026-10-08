# Isolated GLM grouped projection candidate

The default build keeps the grouped W2/W4 operator's existing GM workspace
schedule. Compile an **isolated** OPP package with
`--ops-compile-options -DGLM_W2_GROUPED_CANDIDATE` to select this candidate.
Use a clean source checkout and build directory for each package. Do not put
both packages on one `ASCEND_CUSTOM_OPP_PATH` in a measurement process.

The candidate retains the same `aclnnW2GroupedBlockedDequantMatmulV310` ABI,
signed W2/W4 byte packing, optional lossless NZ byte order, `[N/32,K/32]`
scales, and FP16 output. It changes only the schedule for an NZ-packed expert
with exactly one local route: that expert's decoded 32-by-K tiles go to L1 and
Cube without a GM dequant tile round trip. Groups with two or more routes use
the known-good GM path. Both paths share invariant decode masks and canonical
gather offsets across active experts of the same projection shape. A default
build still prepares those tables for each expert exactly as before.

For `[output,input] = [4096,4096]` W4, the GM path writes and reads a full
32 MiB FP16 decoded weight per active expert, or about 64 MiB of dequant
workspace traffic. The W2 `[4096,2048]` projection is 16 MiB, or about
32 MiB of write/read traffic. The singleton candidate removes that GM round
trip but still moves decoded weights through L1 and runs padded Cube work.
The earlier blanket L1 trial was slower; this narrow schedule is a separate
hypothesis, and parity and latency must decide whether to keep it. The host
tiler still reserves `GetLibApiWorkSpaceSize() + min(aic_cores, N/128) *
128 * K * sizeof(uint16_t)` bytes, including for singleton-only calls.
The PyTorch wrapper does not expose the resolved ACL workspace size; measure
that real allocation before promotion. The harness's PyTorch allocator peak
delta can include output allocations and omit ACL workspace, so it is not a
workspace bound.

Build each package from a clean isolated source tree. For the candidate's
310P package, pass the compile option to `csrc/build.sh` with the grouped op
and its unchanged single-expert reference selected:

```bash
cd csrc
bash build.sh --pkg \
  --ops='w2_blocked_dequant_matmul_v310;w2_grouped_blocked_dequant_matmul_v310' \
  --soc=ascend310p --ops-compile-options -DGLM_W2_GROUPED_CANDIDATE
```

Build the baseline without `--ops-compile-options`. Install each output in
its own OPP directory. Before timing, record SHA-256 for this source file,
the grouped kernel, the shared W2 kernel header, and the actual compiled
operator binary. The `tools.glm_perf.operator_bench measure` command requires
one isolated `ASCEND_CUSTOM_OPP_PATH` root, the grouped operator binary path
within that root, and source paths. Canonical `uint8` and NZ-packed `int8`
layouts compile to separate `.o` files. Select the object whose adjacent
`.json` metadata names the requested code dtype; the harness checks and hashes
both files alongside each case.
Rehash the binary after measurement and refuse results if it changed. Source
selection alone does not establish which kernel ran.

## Native build record, 2026-10-01

The isolated Threadripper build used clean source snapshot `5fdfb939` plus
this candidate's source edits, CANN 9.1, and `-DGLM_W2_GROUPED_CANDIDATE`.
The generated launcher contains that compile option. The package is installed
at `/srv/ai/src/glm53-t4-20261001-opp/vendors/custom_transformer`; the
installer SHA-256 is
`fe118e3c7d44a83bef8015a0d8e0e8d8045aaf87038ea38dc43788f13848fb5f`.
The grouped kernel source SHA-256 is
`56adc37eb81d35ef36623de2c5562e9ce7115d1abd8bbafe5bc73f95a56d7f5c`;
the shared W2 kernel header SHA-256 is
`3668b61146f7a6ca685c26eaa302380dad2882285691f00eda5217667cc09900`.

| Code layout | Installed `.o` SHA-256 | Adjacent `.json` SHA-256 |
| --- | --- | --- |
| NZ-packed `int8` | `9e685a8f67124342e28e3a3784c4a001923477782be70d333d7f9cdc869766bb` | `46fc2492730d599a85b91302b1b09a4f30b5ab5e16b2f9a5214d2d83d55866e9` |
| Canonical `uint8` | `e48fab194f5817c22d385ba1f6803139c8db65b88fbbcbc4229ab18e39bbcacb` | `b4fa6244b088294dd3101cd32d8f44f823c05b862025db8dbe1790521557d44b` |

These hashes identify the built package; they do not establish hardware
correctness or speed. Device parity and timing were deferred by the user.

Run baseline and candidate measurements in separate processes with the same
seed, layout, warmup, repeat count, 8/32 routed rows, and 416-row prefill.
The harness covers distributed, singleton, repeated, zero-local, and peer-owned
routes, including `[1, >1, 1]` local groups separated by empty experts. A
separate-process baseline/candidate comparison of all FP16 outputs **bitwise**
is a mandatory hardware gate before timing is trusted. Same-package canonical
versus NZ-packed parity and the unchanged standalone W2/W4 operator parity are
additional gates. If a future accumulation change
requires tolerance, declare dtype-specific absolute and relative bounds before
the run. Promote only after the isolated latency and matched serving gates in
the GLM performance PRD pass. The existing serving operator selection remains
on the known-good package meanwhile.
