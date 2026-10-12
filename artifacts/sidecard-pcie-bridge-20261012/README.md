# First CUDA–Ascend PCIe bus qualification

The [library](../../tools/sidecard/pcie/README.md) uses CUDA IPC, per-slot native
streams and events, and separate CUDA/ACL pinned buffers. It passed **2,400
byte-exact round-trip invocations**, including warmups, on the assigned
Ascend chips 2 and 3. Each two-buffer invocation verifies both returned
buffers. The GPU is RTX PRO 6000 Blackwell.

See the [design PNG](design.png) and [raw benchmark](roundtrip.json).
This is a transport qualification. **The GLM endpoint still uses its existing
socket backend**; no model speedup is established by this library test.
A post-benchmark [service confirmation](service-confirmation.json) records
HTTP 200 and a healthy 65,536-token expert-only endpoint. The latest full-model
[profile and restored performance](../sidecard-expert-profile-20261012/README.md)
remain approximately 9.64 decode tokens/s on the fixed 2K/64 fixture.

## Transport results

Each entry is **socket reference → bus**, median milliseconds for the complete
round trip. Fifty measured repetitions follow ten verified warmups per case.
Chips are tested separately, with one or two buffers per invocation.

| Bytes per buffer | In-flight buffers | Physical chip 2, ms | Physical chip 3, ms |
| ---: | ---: | ---: | ---: |
| 8,256 | 1 | 1.000 → 0.897 | 1.003 → 0.895 |
| 8,256 | 2 | 1.964 → 1.368 | 1.893 → 1.379 |
| 264,192 | 1 | 1.142 → 0.927 | 1.173 → 0.917 |
| 264,192 | 2 | 2.193 → 1.435 | 2.182 → 1.465 |
| 1,048,576 | 1 | 1.675 → 1.208 | 1.674 → 1.263 |
| 1,048,576 | 2 | 3.273 → 1.809 | 3.266 → 1.867 |

Two 8,256-byte buffers improve median round-trip latency by **27–30%**;
two 1 MiB buffers improve it by **43–45%**. One 8,256-byte buffer improves
by about **10–11%**. Raw samples and mean/minimum/maximum are preserved;
large-buffer bus cases have outliers, so median reductions do not guarantee
tail-latency improvements. The sequential socket reference represents the
existing transfer pattern; it does not explore an optimized socket pipeline.

Each raw test returns the same number of bytes as it sends. Expert input
and output sizes are exercised separately, rather than as an asymmetric
8,256-byte input / 16,384-byte output native MoE call. Producer upload and
byte verification are outside timing. Source readiness completes before
notification. No native expert work is included, and both chips are not
simultaneously driven by this benchmark.

## What the hardware supports

[Topology](topology.json) places RTX at `0000:01:00.0` under root `0000:00`
and Ascend at `0000:49:00.0` under root `0000:40`, in separate IOMMU groups.
The recorded current link speeds are an idle snapshot, not sustained transfer
speed; max capability is not the negotiated speed during the benchmark.

The capability probes record:

- [Shared host-registration probe](capability.json): CUDA can register a
  memfd mapping; `aclrtHostRegister` returns 207000 and V2/default flags
  returns 107000. Shared dual registration was not established.
- [ACL host allocation probe](allocation-capability.json): ACL pinned and
  device allocation succeed; CUDA registration of the ACL host allocation
  returns 1. ACL device IPC export succeeds, but cross-process ACL import
  is not qualified by this probe.
- [CUDA-pinned copy probe](copy-capability.json): ACL can copy from/to a
  CUDA-pinned pointer and reproduce bytes. This probe explicitly synchronizes
  ACL streams and does **not** establish asynchronous behavior or shared
  registration for that pointer; the library retains separate pinned buffers.

Huawei's [host-register product support](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclpythondevg_01_1066.html)
also lists the inference family as unsupported. Its
[asynchronous copy documentation](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/runtimeapi/aclcppdevg_03_0106.html)
distinguishes asynchronous pinned copies from blocking pageable copies.
The installed SDK is CANN 9.1; the driver probes above establish this host's
behavior. No direct cross-vendor PCIe peer-DMA mechanism was verified.

## Implementation and qualification boundaries

Each slot issues source D2H, waits by event query, performs one local CPU
memory copy, issues destination H2D, and waits by event query. The return
reverses the sequence. Independent slots advance together. There are **no
native device-wide synchronization calls** in the bus or benchmark.
CUDA IPC shares dedicated GPU transfer allocations with the ACL worker;
control sockets carry descriptors and invocation/completion identities only
in bus mode. CUDA and ACL frameworks are not imported into one interpreter.

The library checks capacity, endpoint ownership, source event readiness,
write aliasing, process/thread ownership, and pending buffer/event lifetimes.
A native failure poisons the bus and retains potentially active DMA resources
until the owning process exits. The producer/importer control protocol must
keep exported handles and input contents alive until completion.

The [exact executed source archive](executed-source.tar.gz) contains the
qualified Python modules. Their SHA-256 values in `roundtrip.json` match the
published runtime, bus, and benchmark byte for byte. The separate
[diagram source](../../tools/sidecard/pcie_diagram.py) draws this report's PNG.
**626 portable sidecard tests pass**, including 13 bus ownership/failure tests.
These portable tests supplement the raw hardware qualification.

No full-model switch, kernel module, IOMMU/ACS change, card reset, ECC change,
or power/clock adjustment was performed. Further GLM integration needs packed
device buffers, consumer-stream and native scratch ownership, full-vocabulary
parity, repeated decode, context/operational gates, and a new model profile.
The profile's native expert and CUDA launch costs remain separate priorities.

All scoped manual CI hooks pass. Required full-repository `format.sh ci`
still reports the seven existing Ruff, formatting, spelling, and forbidden-import
hook failures outside this change. See [validation](validation.json),
[scoped log](scoped-ci.log.gz), and [full repository log](full-ci.log.gz).
Formatting ran in an isolated worktree; unrelated research edits are preserved.
