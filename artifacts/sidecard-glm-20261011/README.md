# Initial joined GLM sidecard evidence

October 11, 2026. See the [research review](../../docs/source/developer_guide/Design_Documents/glm_sidecard_research.md)
and [tooling runbook](../../tools/sidecard/README.md).

| Artifact | Scope |
| --- | --- |
| [inventory.json](inventory.json) | Read-only hardware/host commands over SSH; device discovery, not inference validation |
| [placement.json](placement.json) | Actual local GLM config SHA256 and offline ownership seed; no memory admission |
| [offline-smoke.json](offline-smoke.json) | Five persistent CPU workers, synthetic resident SwiGLU weights, three passes, exact serial FP32 comparison |
| [canonical-w2-checkpoint-plan.json](canonical-w2-checkpoint-plan.json) | Real local W2 checkpoint headers, complete main-model tensor ownership and global-to-local expert maps; no payload checksum verification |
| [offline-partition-smoke.json](offline-partition-smoke.json) | Synthetic canonical W2/W3/W4 export, per-owner checksum verification and child loading, readiness acknowledgement, and three passes against an independent source-weight control |
| [HARDWARE.md](HARDWARE.md) | Initial real hardware transfer and single-expert arithmetic gates, measurements and limits |
| [hardware-split-expert.json](hardware-split-expert.json) | Real selected-expert packet execution on CUDA and all four Ascend chips, CUDA final reduction, 18 exact mixed-device cases and six exact reduction-contract cases |
| [service/README.md](service/README.md) | Running joined GLM service, complete 42-layer CUDA partition admission, seven operational API cases, real-input expert checks and unresolved greedy repeat variation |
| [context/README.md](context/README.md) | Capacity experiments, workspace OOM boundary, completed 32,000-token output-budget chat and request-admission limits |
| [fast/README.md](fast/README.md) | Current 645,120-token profile, exact DP4A and tensor transport gates, measured prefill/decode and prefix reuse |

The seed covers 42 routed layers and 288 experts per layer with one owner per
expert: CUDA owns 72, and each NPU owns 54. Uniform W4 logical expert codes
alone occupy 35.4375 GiB on CUDA and 26.578125 GiB per NPU. Scales, native
padding/layout, dense weights and runtime state are not included.

Validation: `python3 -m pytest -q --noconftest tests/ut/sidecard` passed 240
CPU tests; Ruff check and format passed. Tests cover partitioned versus serial
expert execution, route identity, retained zero-weight routes, idle owners,
reply reordering, cancellation-sensitive FP32 addition order, malformed/stale
replies, incomplete/immutable ownership, config mismatch and CLI admission labels.
Subsequent tests cover raw tensor transport, sequential daemon reconnection,
capacity tuning, pinned restart staging and completed prefill measurements.
Additional gates cover binary framing, shape/byte limits, wire dtype rounding,
changed payloads, fragmented reads, retained packet storage, deadline expiry,
stale peer replies, worker loss, child cleanup, complete-batch backend dispatch,
canonical tensor coverage and unsafe/corrupt safetensors headers.
Export/loader gates cover exact byte/dtype/shape preservation, reproducible
bundle identities, destination protection, incomplete exports, changed source
files, corrupt partitions, bounded reads, idle owner partitions, signed W2/W3/W4
fields, F16/F32 scales, native-layout rejection and worker preparation failures.

The actual canonical W2 source plan assigns 18,144 main expert tensors plus
text-model non-expert tensors to CUDA, and 13,608 expert tensors to each NPU.
Source tensor bytes total about 34.6123 GiB for CUDA and 13.4967 GiB per NPU.
It excludes 1,753 draft tensors. These are source bytes from the older W2
checkpoint, not W4 bytes, native-bank allocations or evidence of model quality.
The planner did not read, hash, repack or export real GLM tensor payloads.
The export implementation was exercised on small synthetic fixtures only;
no full GLM owner bundle was created. Sidecard native owner exports and
production CUDA/CANN expert backends remain pending. The initial joined
service reads the existing native checkpoint under its inherited EP partition.

After hardware authorization, host-staged transfer integrity passed on all
four chips and one real expert per W4/W3/W2 layer range matched the CPU native
arithmetic reference. Subsequent actual expert packets executed eight selected
real experts across CUDA and all four chips; 18 cases plus CUDA final reduction
matched CPU expectations bitwise. See [the hardware report](HARDWARE.md) for measurements
and limits. Those initial probes did not run joined requests, MTP, graphs,
quality scores or sidecard serving benchmarks and left the existing server
untouched. The later [joined service](service/README.md) replaced that server
and executed real API requests. Its operational and numerical results are
reported separately. The driver stack and original checkpoint were unchanged.
