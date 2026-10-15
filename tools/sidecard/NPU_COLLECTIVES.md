# Experimental local NPU SUM

This library specializes local, fixed-geometry FP16/FP32 SUM. The schedule and
owner API accept ordered groups of N ranks. Hardware qualification currently
covers N2/N3/N4 on two 310P Duos; larger groups remain untested. It uses SDK peer
memory writes, imported Notify objects and local ACLNN addition. Remote vector
reads are rejected. Other collectives remain on HCCL.

See [hardware receipts and limits](../../artifacts/npu-collective-qfn-20261014-024609/README.md)
and the [design PNG](../../artifacts/npu-collective-qfn-20261014-024609/design.png).

The [distributed graph ledger proposal](DISTRIBUTED_GRAPH_LEDGER.md) describes
a future bounded peer-DMA dependency protocol. It is not implemented and does
not establish device-initiated graph launch or autonomous token execution.

## Algorithms and ownership

`BinomialSchedule` handles arbitrary N with reduction followed by reverse-tree
broadcast. `recursive-doubling` requires power-of-two N and performs a SUM with
each XOR partner. Ordered membership, dtype, element count and algorithm are
part of the authenticated schedule identity. Select member order to reflect
local topology; the library does not discover or optimize topology implicitly.

Every rank owns one process/thread/context and one in-order stream. Each edge
has two alternating inbox/READY slots; local input is written into the peer's
inbox before its READY is recorded. Addition reads only local addresses. The
native aggregate plan forwards actual input/output addresses and alternates
persistent work buffers between stages, avoiding separate staging copies.

`NpuCollectiveOwner` borrows an already initialized SDK context. The Torch facade
borrows the caller's Torch stream and submits through a queued native operator.
It does not initialize/reset devices, install kernels, or change environment
variables. Producers, collective calls and consumers must use that stream.
Unsupported geometry can fall back before admission. After partial submission,
errors quarantine owners; automatic fallback or freeing is forbidden.

Shutdown is explicit on every rank: retire graph/queued owners, drain Torch and
SDK work, exchange all-rank drain proofs, close every imported peer object,
exchange import-closure proofs, then release exported resources. There is no
SDK-owning destructor or automatic recovery.

## Private build and gates

Use a fresh output directory for each build or probe. Do not install these
private binaries over qualified runtime libraries.

```bash
python -m tools.sidecard.npu_collective_build --help
python -m tools.sidecard.npu_collective_probe --help
python -m tools.sidecard.npu_collective_torch_probe --help
python -m tools.sidecard.npu_collective_graph_probe --help
```

The builder accepts explicit SDK/compiler paths. The Torch addon additionally
requires explicit Torch/Torch-NPU roots, Python headers and C++ ABI selection;
the builder never imports Torch. Products and source/compiler identities are
recorded in its build receipt. A separate HCCL reference supports matched raw
comparisons.

The raw SDK gate checks complete CPU/HCCL results and distinct device-address
histories. The Torch gate additionally checks queued tensor lifetimes on every
registered geometry. Neither synthetic gate establishes broad FP16 numerical
parity, model quality, graph support or token throughput.

## Optional vLLM integration

The communicator reads `additional_config["npu_collective"]` only for TP groups.
Default behavior is unchanged. The default facade requires explicit
`enforce_eager=True`. The optional `execution_mode: "graph"` admits registered
captures with fixed-address schedules; ordinary recursive doubling is rejected
because its host-side alternating slot cannot advance during replay. The new
`recursive-doubling-fixed` mode uses slot zero and a device READY/CONSUMED
handshake for every call. Its N4 hardware graph gate passes all 14 registered FP16/FP32 geometries.
The fixed schemas are registered up front. Other shapes use the original HCCL
SUM implementation.

Graph ownership is explicit and kept off the replay path. Before capture, the
cold lifecycle registry acquires native C++ leases for the collective plans.
Each graph entry caches Torch-NPU's bound native `replay` method. Successful
production replays invoke no Python observer callbacks, counters or facade
validation. The existing runtime still schedules launches and attention updates.

Shutdown stops request admission, drains the device, resets every registered
native graph, releases its native plan leases, then closes collective owners.
The C ABI rejects destruction while any lease remains. Release requires the
caller's explicit reset-and-drain proof; the C library does not inspect a
foreign Torch graph to establish that proof itself. Failed capture/reset/release
retains resources. The caller must serialize eager work, replay and retirement;
graph mode does not authorize concurrent reuse of collective work buffers.

Optional `graph_replay_tracking: true` enables diagnostic Python accounting.
It defaults to false and requires graph mode. Unobserved production receipts
label counts as capture-only; model reports use null for unmeasured replay
counts. Diagnostic counts are submission evidence, not completed device timing.

Configuration requires these fields and accepts optional `execution_mode`
(`eager` or `graph`) and boolean `graph_replay_tracking` (default false):

```json
{
  "schema_version": 1,
  "acl_library": "/absolute/path/libascendcl.so",
  "acl_sha256": "<SHA256>",
  "native_library": "/private/build/libsidecard_npu_collective.so",
  "native_sha256": "<SHA256>",
  "torch_library": "/private/build/libsidecard_npu_torch.so",
  "torch_sha256": "<SHA256>",
  "algorithm": "recursive-doubling",
  "geometries": [{"dtype": "float16", "elements": 2560}],
  "physical_devices": [0, 1, 2, 3],
  "link_groups": [[0, 1], [2, 3]],
  "output_root": "/fresh/private/receipts"
}
```

Physical devices map one-to-one to ordered group members; link groups declare
sibling chips for this explicit local topology. Both private DSOs must share a
directory and match their hashes. The experimental runtime must include the
`tools.sidecard` modules; the normal installed plugin does not require them.

For controlled QFN qualification, `qfn_collective_benchmark.py` uses matched
weights, cache, operator packages and MTP2 in both arms. `--graph-profile`
selects `eager`, qualified `c1-c2` sizes `[3,6]`, or `c3-c4` sizes `[9,12]`.
Only two sizes are captured because of the qualified runtime's event budget. It
warms all four prompts, verifies completion budgets and records cached tokens,
per-trial custom/fallback call deltas, text and token IDs. Its worker extension
provides an all-rank close RPC before engine shutdown. The harness holds owners
on failure; it never reboots, resets or automatically kills a failed worker.
Model token IDs and quality must be assessed separately from ownership success.

The existing QFN HCCL path reduces in place. The current custom operator returns
a new tensor. The Torch microbenchmark compares out-of-place results and
therefore adds a baseline clone; real-model results are the relevant check for
this allocation difference.

The selected custom QFN research baseline is recorded in the
[promotion receipt](../../artifacts/npu-collective-qfn-20261014-024609/profile-c1/promotion.json).
On the benchmark host, `run-qfn-custom-current.sh` in that experiment root selects
the hardware-tested pinned-position runtime. Supply normal benchmark arguments,
a fresh collective owner directory, and apply the recorded worker CPU affinity
before timing. This promotion preserves the existing precision, context and
unobserved native collective replay. Its 31.42 tok/s C1 median overlaps the prior
31.23 tok/s range; throughput is classified as neutral. The
[fresh profile report](../../artifacts/npu-collective-qfn-20261014-024609/README.md)
records the remaining graph breaks and compute costs.
