# Fixing partial graphs forward

The candidate splits each decoder layer before and after sparse attention.
Sparse cache selection and masking remain eager; positions and stage inputs
use persistent device buffers, and successive captured layers share their
outputs directly. Capture does not depend on the context block count.

The first candidate used a pool per piece and exhausted capture memory.
The next used a shared pool but its admission check hit the 310P int64
collective wrapper's unsupported MIN reduction. Both stalled pipeline
requests and required service restarts. Logs are preserved here.

The corrected candidate uses one pool and stream per worker, reclaims unused
allocator blocks before first capture, and checks pair readiness using a
FP32 SUM vote on the CPU group. This bypasses the int64 compatibility wrapper
and keeps rank admission consistent. A 768 MiB free-memory reserve gates
new captures. These checks run during capture admission, not replay.

The summed readiness vote passed on all six workers for all-ready, one-ready
and none-ready cases. An idle-worker capture probe exercised two real layers
per stage. Every preparation tensor matched exactly; ten of twelve feed-forward
outputs matched exactly and two differed by at most 0.00000763. Reported tensor
allocation growth was 240128 bytes per worker; this does not include every
runtime allocation and is not a full-model memory claim.

The complete four-token shadow captured all 240 pieces across six ranks with
no memory admission fallback. Maximum layer-output differences were 0.001953,
0.003906 and 0.015625 by stage. This check was not bitwise exact. It returned
eager outputs while comparing the candidate and cleared request caches around
the comparison.

The direct-buffer serving candidate is retained. Three matched 64-token
requests measured 3.462157, 3.460800 and 3.462751 tokens/s after first output,
versus 3.428689 median fused eager. All response messages matched exactly.
Counters show 240 captures and 45360 replays, with 114 directly bound layers
across the six ranks. Each token needs two stage-input/position copies on the
first stage and three on the other stages, all device-to-device. The lowest
free-memory admission reading was 1326141440 bytes. Context remains 163840;
no full-context test or broad quality qualification was run.

The production source is `tools/minimax_m3/partial_graph.py`; enable it using
`tools.minimax_m3.resident switch --native-name minimax_w4_int8_v13
--activation-name minimax_activation_v2 --partial-graphs` on one shell line.
The installed post-start script uses this option. A post-integration 64-token
check measured 3.455752 tokens/s. CPU ownership tests cover shared pool/stream,
direct buffers, no context-boundary recapture, peer memory rejection, eager
prefill, and cleanup at the next drained switch.

The raw `.py.txt` files preserve experiments; use the production CLI for serving.
Two restarts were required for the initial failures, but the corrected capture
trial and production integration reused the final resident workers and shards.
