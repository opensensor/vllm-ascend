# GLM offline scale fusion summary (US English)

The expert kernel currently rounds loaded FP32 weight scales to FP16 and back
to FP32 during every scale preparation. The new explicit disk format stores
those rounded FP32 values once. Its paired kernels skip both vector casts and
their synchronization. FP32 storage, multiplication, and addition remain;
quantization bits and packed code bytes stay unchanged. Scale scratch falls
from 4,096 to 2,560 bytes per kernel instance. GM scale traffic stays the same.

This includes the complete path: a CPU checkpoint exporter, authoritative scale
shards, loader markers and checksums, guarded target/draft bank dispatch, and
matching gate/up and down kernels. No scale transformation happens during load,
capture, or forward. The exporter links unchanged model shards and publishes
the complete marker last. Old unmarked banks cannot use the new kernels.

The checkpoint packager also now copies and validates the routed-input kernel,
which was missing from the permanent prefill bundle copy path. The full-MoE
comparison harness now supports scale preparation as an isolated feature and
includes rounding ties, subnormals, and signed zero before changed-input graph
replay and alternating timing. Standalone fixture preparation stays outside
the measured pipeline; production scales must come from the disk checkpoint.

Decode v956 and prefill v957 compiled successfully for dav-2002 using the CPU
SDK. They retain the previous native-column route reduction. The compatible
CPU suite passed 1,524 tests; three files requiring locally unavailable
upstream/NPU modules remain excluded. Scoped checks passed. Tests cover exact
CPU rounding, interrupted exports, unchanged source bytes, actual loader file
selection, checksum failures, and rejection of raw resident scales.

No NPU work, inference, recovery, or hot swap ran. The real checkpoint and
server remain unchanged. New kernels are hardware-unvalidated. CPU rounding
does not prove parity with the device's Cast instruction, and compilation does
not prove faster decoding or prefill. The next hardware gate must compare full
MoE output bits before real-weight gates, target/draft graph replay, and cold
prefill/c1/c4 timing. Language output quality remains for the user to assess.

The archive contains 44 build, source, and log files with recorded SHA256
digests. The [Chinese report](README.md) includes the pending export and gate
commands; none were run on the real model or devices.
