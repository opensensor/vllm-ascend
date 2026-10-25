# Fused M3.1 activation

The retained v2 kernel consumes dense FP16 gate/up device buffers and computes
clipping, sigmoid and multiplication in FP32. Launch geometry is cached per
size; there is no per-token host staging. Ineligible calls retain the reference
implementation. The append-only native loader seals the binary and helper with
SHA256 hashes. `manifest.json` identifies the exact installed build.

All six workers passed four exact FP32 comparisons, including every finite FP16
gate value with randomized up values. The isolated four-route probe measured
0.0268 ms native versus 0.1252 ms reference. The earlier v1 lacked the explicit
NPU storage-format guard; v2 adds that guard with unchanged arithmetic.

Serving v2 measured 3.4311, 3.4287 and 3.4063 tokens/s after first output on
three 64-token counting requests. All messages matched the baseline exactly.
This is a small gain over scale reuse alone (3.3886 median), not evidence of
broad quantization accuracy. Context capacity remains 163840; no full 160K
prompt was evaluated. Worker PIDs and model storage were unchanged across
activation loading and replacement. The native-load transaction cleared
request caches. A later graph experiment required a service restart; these
receipts precede that experiment.

`activate-native.sh` installs this combination on fresh workers after INT8
repacking. Do not rerun the whole script on already-repacked workers. Native
activation can be disabled independently while retaining INT8 expert dispatch.
Raw experiment sources use `.py.txt` to preserve exactly what was executed.
