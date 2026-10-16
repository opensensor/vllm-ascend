# RTSP scene-description experiment

Standalone camera sampling and Qwen3-VL inference, independent of LightNVR.
The first hardware experiment ran on `192.168.53.187` on October 12, 2026.
Camera credentials and footage remain outside this repository.

The current service uses both chips of card 8217 in TP2 with INT8 text
projections, native decode graphs, specialized matrix/NZ kernels and DVPP
raw RGB capture. See [the serving integration](#opt-in-serving-integration)
for the selected configuration, measured gains and current capacity limits.
Earlier sections retain the baseline experiments that led to it.

## Continuous four-camera JSON log

`continuous_log.py` runs the four live cameras independently, sampling through
DVPP into raw RGB PPM at 1 FPS. Every eight sampled frames form a scene request
to the shared model endpoint. Capture continues while inference runs, with a
global limit of 16 unfinished requests. A failed or stalled camera reconnects
independently; inference errors and exhausted-budget drops are logged explicitly.

On the test host, `rtsp-scene-log.service` writes private daily UTC files under
`/srv/ai/storage/rtsp-scene-logs`. `scenes.jsonl` points to the current day's
file. Each line is a complete JSON object. Logs append across restarts and are
retained; temporary sampled frames are deleted after processing. A process lock
prevents two logger instances from owning the same output directory.

```bash
ssh matteius@192.168.53.187 \
  'tail ---disable-inotify -F /srv/ai/storage/rtsp-scene-logs/scenes.jsonl'
ssh matteius@192.168.53.187 \
  'cat /srv/ai/storage/rtsp-scene-logs/status.json'
ssh matteius@192.168.53.187 \
  'sudo systemctl stop rtsp-scene-log.service'
```

The test host's uutils coreutils 0.8.0 `tail` follow mode stalled while the
file continued growing. Its polling override uses three leading hyphens:
`---disable-inotify`. A 12-second check with that option received seven new
records; JSONL itself does not affect following appended lines.

For a refreshing view grouped by camera, run this on the test host. It shows
the last five completed scenes per camera in capture-time order (UTC); reduce
`[-5:]` if the output exceeds the terminal height. `watch -x` passes the jq
filter directly without another shell interpreting its quotes.

```bash
watch -n 2 -x jq -rs '
  map(select(.type == "scene"))
  | group_by(.camera)[]
  | "\n=== \(.[0].camera) ===",
    (sort_by(.window_start_utc)[-5:][]
     | "\(.window_start_utc)\n  Summary: \(.description.summary)\n  Description: \(([.description.events[]?.description] | join("; ")) as $events | if $events == "" then "No events described." else $events end)\n")
' /srv/ai/storage/rtsp-scene-logs/scenes.jsonl
```

Scene records contain `camera`, `session`, `window`, approximate
`window_start_utc`/`window_end_utc`, all eight `decoded_pts_seconds`, gap
metadata, the structured `description`, model/prompt identifiers, token usage
and inference duration. Event frame IDs index the eight decoded timestamps.
UTC is anchored to first frame arrival, not camera NTP; this is insufficient
for precise POS clock calibration by itself. Records arrive in completion
order; use the camera/session/window fields to reconstruct temporal order.

Other record types are `service_started`, `camera_connected`, `capture_error`,
`inference_error`, `inference_dropped`, `heartbeat` and `service_stopped`.
`status.json` updates every 30 seconds with per-camera counters and the number
of pending requests. Frame gaps inside windows and between windows are retained
in timestamps; reconnects begin a new session and discard any incomplete
window explicitly. The back-door reconnect during initial validation was
intentional; its receipt is `reconnect-validation.json` in the log directory.

Both `rtsp-scene-model.service` and `rtsp-scene-log.service` are enabled at
boot and restart on failure. The logger requires the model unit and starts
after its bounded HTTP readiness check. A model restart also restarts the
logger, which drains pending requests before the model stops. Use
`sudo systemctl restart rtsp-scene-model.service` for a coordinated restart.
The model owns one TP2 instance on devices 0 and 1; launching a second TP2
instance on the same chips can still conflict at HCCL initialization. Neither
full-frame-rate analysis nor precise camera wall-clock synchronization is implied.

To start a separate instance in the pinned Ascend environment, use a different
private output directory and avoid competing with an existing logger:

```bash
python -m tools.rtsp_scene.continuous_log \
  --config /private/cameras.json \
  --camera FrontDoor --camera back-door --camera Driveway --camera open-bulb-cam \
  --output /private/live-json --dvpp-binary /private/dvpp_decode
```

## Hardware-validated baseline

- Model: `Qwen/Qwen3-VL-8B-Instruct`, revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, FP16, eager execution.
- Hardware: one Ascend 310P3 device, logical device 0 on card 8217.
  Device 1 was initially unused; subsequent tests use both chips of this card.
  The existing GLM workload on devices 2 and 3 was preserved.
- Runtime: CANN 9.1.0, existing `qwen38-w4-test-ce1862` virtualenv,
  vLLM source `3ab5dda`, Ascend source `e5ad0bf1d`.
- Four authenticated RTSP sources worked: FrontDoor, back-door, Driveway,
  and open-bulb-cam.
- Two passes across all four cameras produced eight valid scene descriptions.
  Inference latency was 5.33–5.89 seconds, median 5.45 seconds. These windows
  were static and produced no action events.
- The back-door motion test captured a person turning with a carried object
  and walking into the yard. The original verbose response took 38.68 seconds;
  grouping repeated actions reduced the same-window response to 9.54 seconds.
- A continuous recording recovered the return segment. The model recognized
  movement toward the camera and a can, but incorrectly described descending
  steps. Camera-relative prompting did not eliminate that direction error.
- A neutral object prompt called the object a beverage can in wide frames.
  A crop from the original recording elicited “beer can,” but guessed the brand
  as Corona despite visible Modelo lettering. Category and brand recognition
  must be evaluated separately. An OCR-focused prompt guessed Coors instead;
  prompting did not fix brand recognition in this example.

These are small exploratory checks, not accuracy benchmarks or sustained
multi-camera capacity results. Eight ordered frames at 1 FPS are submitted per
window; native video input has not been tested. CPU FFmpeg and direct Ascend
DVPP capture backends are available. The sampled loop waits for
inference before starting its next capture, so it has coverage gaps. Continuous
recording and asynchronous analysis are required for uninterrupted coverage.

## Bounded capacity measurements

`capacity.py` replays saved windows with a unique text identifier before each
image sequence to avoid reusing an earlier KV prefix. Each concurrency level
uses two batches. Client latency includes queueing; server telemetry separates
queue, prefill, decode, and time to first token. These are short replay tests,
not sustained RTSP throughput or latency guarantees.

On device 0 with a maximum of two active sequences:

| Window content | Concurrent requests | Mean batch completion | Median client latency |
| --- | --- | --- | --- |
| Static scenes | 1 | 5.28 s | 5.27 s |
| Static scenes | 2 | 7.38 s | 7.36 s |
| Static scenes | 4 | 15.94 s | 12.35 s |
| Motion | 1 | 9.24 s | 9.23 s |
| Motion | 2 | 14.66 s | 14.31 s |

For four static requests, average server queue time was 3.93 seconds. For one
motion request, mean prefill was 2.25 seconds and decode was 6.96 seconds.
Thus an eight-second window arriving every eight seconds exceeds the measured
single-request motion throughput. Video decode offload alone does not remove
the text-generation bottleneck.

A second-device graph candidate captured batch sizes 1, 2, and 4 and reported
ready, but its first actual image request crashed in the attention graph-update
path with missing `attn_params`. It is not validated and contributes no capacity
claim. Logs are retained as `serve-graph.log`. A separate eager replica was
started on device 1 at port 8004 to measure the whole card.

With requests distributed across the two eager replicas, four static windows
completed in 9.30 seconds on average. Four motion windows completed in 15.58
seconds. These results support trying four-camera summaries on a 20-second
cadence, but do not establish uninterrupted eight-second motion analysis.

`stream.py` then held one back-door RTSP connection open and submitted six
consecutive eight-frame windows at 1 FPS to device 0 while capture continued.
All 48 frames and six descriptions completed successfully. Inference ranged
from 5.62 to 10.09 seconds; earlier pending requests at each new window were
`[0, 0, 0, 0, 0, 1]`. The run completed in 54.76 seconds including startup and
the final response, with no gaps between sampled windows or accumulating
backlog in this short test. It does not validate longer runs or busy scenes.

```bash
python -m tools.rtsp_scene.capacity \
  --captures /private/capture-a /private/capture-b \
  --output /private/capacity-results --concurrency 1 2 4 --rounds 2
```

Repeat `--endpoint` to distribute requests across independent model replicas.
Do not run unrelated client requests against these replicas during measurement,
because their requests would contaminate server metric differences.

```bash
python -m tools.rtsp_scene.stream \
  --config /private/cameras.json --camera test-camera \
  --output /private/continuous-results --windows 6 \
  --endpoint http://127.0.0.1:8003/v1/chat/completions \
  --dvpp-binary /private/dvpp_decode
```

This bounded continuous mode retains one RTSP connection, atomically writes
sampled frames, and submits windows asynchronously. It records outstanding work
at each arrival instead of hiding backlog by pausing capture. Model replicas
can be selected with repeated `--endpoint` flags.

## Direct Ascend video decoding

`dvpp_decode.cpp` calls the installed CANN 310P `hi_mpi_vdec_*` APIs directly.
The default capture format is now **raw RGB**, described below: video decode,
resize and color conversion run on DVPP, with no JPEG step in the model path.
Host work includes FFmpeg demux/Annex-B packet formatting, buffer copies,
file writes, base64/JSON transport and image preprocessing before the NPU
vision tower. This is not a zero-copy pipeline.

Select `--frame-format jpeg` for the earlier compressed evidence path. VDEC
resizes to NV12, then JPEGE encodes those buffers directly on the card at
quality 90. Only compressed JPEG bytes return to the host in that mode.
JPEG requests still use the standard API's CPU JPEG loader; this compressed
mode is explicit and is no longer the inference default.
The Python camera/window interface is shared with the CPU backend; other
hardware providers can implement the same frame and timestamp contract.

Build on the Ascend host with the FFmpeg development libraries
(`libavformat-dev`, `libavcodec-dev`, `libavutil-dev`). The helper no longer links
directly against libjpeg:

```bash
source /srv/ai/bin/ascend-env.sh
bash tools/rtsp_scene/build_dvpp.sh \
  /usr/local/Ascend/cann-9.1.0/x86_64-linux /private/dvpp_decode
python -m tools.rtsp_scene.runner \
  --config /private/cameras.json --camera test-camera \
  --output /private/dvpp-captures --capture-only \
  --decoder ascend_dvpp --dvpp-binary /private/dvpp_decode --decode-device 0
```

The same decoder flags work with `stream.py` and `multi_stream.py`. Hardware
decode with raw RGB output is the default; pass `--dvpp-binary`. Hardware
JPEG compression remains available with `--frame-format jpeg`.
Select `--decoder ffmpeg_cpu` explicitly for a comparison against the CPU baseline. Hardware failures are reported without silently switching
to CPU. Unsupported codecs and damaged frames fail the capture. Input URLs go
to the hardware helper through stdin and do not appear in its command line.
Each process owns its channel and keeps input/output buffers alive until DVPP
releases them; queued device buffers are bounded.

The helper waits for a timestamped keyframe when RTSP has not yet anchored its
RTP clock. It retains actual decoded PTS, writes each JPEG atomically, and
reports decoded frames, sampled frames, process CPU time and missed sampling
slots. Missing slots remain missing instead of generating catch-up samples.
Decoder CPU time includes initialization and teardown and can span several
cores; it must not be interpreted as a fraction of inference latency.

Hardware checks on October 12, 2026:

- Direct capture worked for all four local cameras: three H.264 sources and
  the 3840×2160 H.265 Driveway source. The initial live test rejected the first
  untimestamped RTSP packet; waiting for a timestamped keyframe resolved it.
- The final four-source check recorded 10 missing sampling slots on FrontDoor,
  one on back-door, and none on Driveway or the bulb camera. FrontDoor needed
  25.36 seconds to deliver eight samples spanning 17.04 seconds of media PTS.
  A follow-up FrontDoor run decoded 56 of 57 submitted frames and observed an
  incoming packet PTS jump of 2.08 seconds. These diagnostics do not isolate
  every earlier gap; camera/network cadence still needs investigation.
- Sampling the saved back-door recording used 0.56 CPU-seconds. Equivalent
  CPU FFmpeg captures used 1.17–1.20 CPU-seconds. One-shot DVPP wall time was
  3.26 seconds versus 0.33 seconds for FFmpeg, so this does not demonstrate a
  one-shot latency speedup. Initialization cost favors a persistent decoder.
- Persistent back-door capture delivered 24 sampled frames and three valid
  model descriptions. Capture used 0.85 CPU-seconds total. Window arrivals
  were 9.47, 17.44 and 25.76 seconds; inference took 5.65–7.97 seconds, with no
  earlier request pending at those arrivals. One missing sampling slot was
  recorded in the actual PTS and counters. This is a short single-camera test.

The direct decoder resize path follows the documented
[310P video decoding interface](https://www.hiascend.com/doc_center/source/en/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_000102.html).
Validate against the installed CANN headers when using a different release.

## Inference profiling and optimization priorities

Fresh motion requests used 8.05–11.41 CPU-seconds in the host engine during
9.16–12.17 seconds of inference. The API used 0.08–0.20 CPU-seconds; the client
used approximately 0.006 seconds. Engine idle activity was approximately
0.073 CPU-seconds per wall second. Host CPU work overlaps NPU execution;
these counters do not attribute CPU delay on the critical path.

During the baseline investigation, port 8004 was restarted with the built-in
torch NPU profiler for bounded CPU/device traces. It now serves the unprofiled
INT8 candidate. Profiler overhead and trace parsing must be excluded from serving
benchmarks. The short diagnostic deliberately capped output at 12 tokens;
its truncated text is not an accepted scene description.
After profiling stopped, an unprofiled motion request returned 79 tokens in
9.70 seconds, consistent with the earlier baseline.

`profile_summary.py` identifies text decode by the first M=1 QKV projection
with output shape `1,6144`, then aggregates recorded device tasks. The model
has 36 text layers. Shape-based separation is necessary because the configured
profiling delay did not exclude vision/prefill in this capture. It rejects mixed
devices and incomplete QKV layer groups, merges overlapping task intervals,
and reports limitations alongside the results.

```bash
python -m tools.rtsp_scene.profile_summary \
  /private/trace/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --output /private/decode-summary.json
```

For 11 captured decode forwards, 7,648 tasks spanned 1,008.84 milliseconds.
Their active interval union was 958.94 milliseconds (95.05%); intervening gaps
were 49.90 milliseconds. MatMulV2 accounted for 94.31% of summed task duration;
attention accounted for 3.19%. Matmul counters averaged an MTE2 ratio of 0.968
and a MAC ratio of 0.055, consistent with substantial weight-transfer cost.
These are profiled device intervals, not production utilization or a CPU/NPU
latency decomposition. The first decode normalization before QKV is excluded.

Prioritize controlled experiments in this order:

1. Benchmark text projections, gate/up and down projections and the vocabulary
   head at decode batch sizes 1, 2 and 4. Measure FP16/NZ weight traffic and
   investigate supported lower-bandwidth quantized projections. Validate image
   descriptions and carried-object accuracy before accepting quantization.
2. Batch camera windows to reuse weight reads while recording queueing delay,
   summary latency and dropped/missed windows in a persistent multi-camera run.
3. Isolate the graph attention-update failure and compare replay with eager
   execution under identical real-image requests.
4. Profile vision/prefill separately; select normalization, activation or layout
   fusions from those traces. Text decode already uses merged QKV/gate-up
   projections and AddRmsNorm. Small-operation fusion alone has limited room
   in the captured decode device-time breakdown.

## Validated enhancements: hardware JPEG and INT8 text

The compressed backend uses NV12 VDEC output directly as JPEGE input. The saved
motion clip produced eight complete 640×480 JPEGs, with 124 decoded frames,
155 submitted frames, zero missing sample slots and 0.562 CPU-seconds.
Total wall time was 3.65 seconds, including initialization and teardown.
Only 1,312,522 compressed JPEG bytes were copied back; no sampled RGB copy or
CPU JPEG compression remains. Visual inspection confirmed correct color and
content. A separate live check encoded eight samples from each of all four
cameras, including the 4K HEVC source. FrontDoor had five missing sampling
slots in that check, so hardware codec success does not imply complete coverage.

`projections.py` compares real checkpoint projection weights using FP16/NZ
and the existing Ascend dynamic INT8 operators. Inputs are synthetic; loading
and weight conversion are excluded. Each case uses 20 iterations and three
repeats, at batch sizes 1, 2, 4, 16 and 128. At batch size 1:

| Projection | FP16 | Dynamic INT8 | Speedup |
| --- | --- | --- | --- |
| Merged QKV | 0.277 ms | 0.173 ms | 1.60× |
| Attention output | 0.190 ms | 0.060 ms | 3.16× |
| Merged gate/up | 1.092 ms | 0.601 ms | 1.82× |
| MLP down | 0.548 ms | 0.296 ms | 1.85× |
| Vocabulary head | 6.215 ms | 3.176 ms | 1.96× |

All 25 cases produced finite outputs; relative L2 error was approximately
0.012–0.015 against FP16. These are operator measurements, not scene accuracy.
The vocabulary head was benchmarked but remains FP16 in the served checkpoint.

`quantize.py` creates a separate, uncalibrated checkpoint using symmetric
per-output-channel INT8 text weights. All 252 text projections are converted;
the vision tower, embeddings, norms and vocabulary head retain their original
floating weights. The runtime dynamically quantizes activations per token.
No custom kernel or upstream model source change was needed. Source weights
remain untouched; `conversion.json` records source hashes and selected weights.
The candidate stored 10.60 GB of tensors and loaded 12.47 GB of device weights,
versus 18.92 GB of FP16 device weights.

```bash
python -m tools.rtsp_scene.projections \
  --model /srv/ai/storage/models/Qwen3-VL-8B-Instruct \
  --output /private/projections.json --iterations 20 --repeats 3
python -m tools.rtsp_scene.quantize \
  --source /srv/ai/storage/models/Qwen3-VL-8B-Instruct \
  --output /private/Qwen3-VL-8B-Instruct-w8a8-text
OMP_NUM_THREADS=4 python -m tools.rtsp_scene.launch \
  /private/Qwen3-VL-8B-Instruct-w8a8-text \
  --port 8004 --device 1 --max-sequences 4 --int8
```

The pinned runtime expects the full per-parameter quantization description
inside the Hugging Face `config.json`, including its vision/text namespace
aliases. A format label alone failed during model construction; the converter
now emits the full configuration and the sidecar. Initial default mmap loading
also stalled in host memory pinning and huge-page splitting. The candidate uses
`--safetensors-load-strategy eager` and disables transparent huge pages **only
for its process and descendants**. Together these settings loaded the weights
in 14.42 seconds. They have not been tested independently. The launcher makes
this combined workaround reproducible without changing global host settings.

`compare.py` submits identical prompts and frames to both replicas, with unique
prefixes, temperature zero and the same strict response schema. All seven
scene pairs returned valid descriptions. On the departure clip, both produced
the same 79-token description: FP16 took 9.27 seconds and INT8 took 6.25 seconds
(1.48× faster). On return clips, the candidate recognized the carried can but
retained the baseline's incorrect descending-step description. This small
qualitative check does not establish statistical accuracy parity, direction
accuracy or reliable product/brand recognition.

A paired neutral prompt on the carried-object crop produced the beer-can
category in both models, but both guessed Corona despite the visible Modelo
label. FP16 took 10.35 seconds for 117 tokens; INT8 took 6.58 seconds for 118
tokens. The wrong brand persists after quantization.

Fresh motion replay on the INT8 candidate, two batches per concurrency level:

| Concurrent requests | Mean batch completion | Median client latency |
| --- | --- | --- |
| 1 | 6.18 s | 6.17 s |
| 2 | 10.23 s | 8.81 s |
| 3 | 12.25 s | 11.05 s |
| 4 | 14.58 s | 13.21 s |

The candidate allows four active sequences; the original FP16 baseline allowed
two, so the four-request comparison also changes scheduler capacity. One
motion stream fits an eight-second cadence in these bounded measurements.
Two continuously busy streams on one chip at that cadence exceed measured
throughput. A follow-up three-request replay completed its two batches in
11.97 and 12.53 seconds, averaging 12.25 seconds. Its six responses produced
75–117 tokens. This also falls short of three busy streams delivering windows
every eight seconds. The three-request evidence is in `capacity-int8-motion-c3`;
it used the same saved motion captures on port 8004. Longer runs and busier
scenes still need validation.

## Live multi-camera queue

`multi_stream.py` holds each RTSP connection open and shares a bounded inference
budget. It records pending requests, inference failures, decoder exit status,
actual timestamps and capture counters. When inference reaches the budget,
it explicitly drops a new analysis window while capture continues. Capture
failures also fail the run; they are never presented as successful quiet scenes.

```bash
python -m tools.rtsp_scene.multi_stream \
  --config /private/cameras.json \
  --camera back-door --camera Driveway --camera open-bulb-cam \
  --output /private/live-results --windows 3 --max-in-flight 8 \
  --endpoint http://127.0.0.1:8003/v1/chat/completions \
  --endpoint http://127.0.0.1:8004/v1/chat/completions \
  --dvpp-binary /private/dvpp_decode
```

Two live cameras on the single FP16 chip completed six descriptions in 38.19
seconds, including capture startup and the final response. There were no
inference drops or missing sampling slots; pending requests reached three and
responses took 6.58–13.52 seconds. Each camera's hardware capture/encoding used
0.81–0.82 CPU-seconds for 24 sampled frames.

A four-camera test using the FP16 and INT8 replicas completed nine descriptions
from back-door, Driveway and open-bulb-cam in 31.91 seconds. Their inference
latencies were 3.94–7.68 seconds, with no inference drops. FrontDoor failed with
a stream-read error before completing its first window; the test correctly
returned failure. Back-door recorded one missing sample slot; Driveway and
open-bulb-cam recorded none. The three completed captures each used
0.90–0.96 CPU-seconds for 24 sampled frames. This validates three-camera capture
and inference in a short, mostly static test, not sustained four-camera coverage.

Graph execution, vision preprocessing fusions and a device-resident vision
input path remain unvalidated. Hardware JPEG encoding is validated in the
optional compressed backend. The default raw RGB path eliminates JPEG input
decoding entirely, rather than moving that codec to another processor.

## Raw RGB model input: no JPEG round trip

The current default is `--frame-format ppm`. VDEC emits resized RGB888 on the
card; the helper copies sampled pixels to the host, strips row padding and
writes a binary P6 PPM header followed by the RGB bytes. This serialization
has no image compression codec. `describe()` sends those existing bytes as
`data:image/x-portable-pixmap;base64,...` through the unchanged local API.
The installed vLLM `ImageMediaIO` accepts this format. Its Pillow PPM reader
loads RGB pixels without JPEG decompression. The vision tower and text model
continue to execute on the NPU.

```bash
python -m tools.rtsp_scene.runner \
  --config /private/cameras.json --camera back-door \
  --output /private/raw-rgb-results \
  --dvpp-binary /private/dvpp_decode \
  --endpoint http://127.0.0.1:8004/v1/chat/completions
```

The same default applies to `stream.py` and `multi_stream.py`. Frame filenames,
timestamps, capture manifests and window copies preserve `.ppm` throughout.
The runner validates dimensions, channel depth and exact payload length before
sending a request. Invalid data fails without conversion to another format.
Saved JPEG captures remain readable for historical comparisons.

The saved motion clip yielded eight 640×480 RGB frames: 124 decoded frames,
155 submitted frames, no missed sample slots, 7,372,800 RGB bytes copied to the
host and **zero JPEG bytes**. Capture used 0.545 CPU-seconds and 4.26 seconds of
wall time including initialization/teardown. Its PTS matched the earlier JPEG
capture. A real INT8 request returned a valid scene description and mentioned
the carried can; the existing step-direction error remains. Color inspection
showed correctly rendered RGB. An isolated check of the installed vLLM loader
succeeded with `JpegImageFile.load` patched to raise on any call.

`transport_bench.py` stages a pixel-matched comparison from saved JPEG evidence,
decoding it once offline to make a raw RGB fixture. Live capture never uses
that offline JPEG conversion. Three sequential pairs alternate transport order
and use identical prompts within each pair. All six actual model requests
returned valid descriptions. They are a small quality check, not proof of
output equivalence: generated token counts and wording varied.

For the equal-length 100-token pair, JPEG inference took 7.44 seconds and raw
RGB took 7.41 seconds. Preparing the request took approximately 7.7 ms versus
50.0 ms. HTTP bodies were **1,753,126 bytes versus 9,833,750 bytes**, approximately
5.61× larger. Across all three rounds, differing output lengths prevent a clean
latency comparison. No end-to-end inference speedup is claimed.

`media_bench.py` isolates the actual runtime image loader using cached host
bytes, excluding HTTP and inference. With 20 eight-image windows per repeat
and five repeats, median loader CPU time fell from **17.23 ms per window for
JPEG to 2.50 ms for raw RGB**. All raw loads also passed with JPEG decoding
forbidden. The codec saving is small relative to model inference; larger
base64/JSON requests add host work. A future device-buffer interface could
remove these copies and serialization, but has not been implemented.

```bash
python -m tools.rtsp_scene.transport_bench \
  --capture /private/jpeg-capture --output /private/transport-comparison \
  --endpoint http://127.0.0.1:8004/v1/chat/completions --rounds 3
python -m tools.rtsp_scene.media_bench \
  --fixtures /private/transport-comparison --output /private/media-loader.json
```

A live raw-RGB run on back-door, Driveway and open-bulb-cam captured 72 frames
and completed nine descriptions, with no inference drops or capture failures.
It finished in 38.38 seconds including startup and the final response.
Inference took 3.58–13.22 seconds. Capture used 0.86–0.87 CPU-seconds per camera;
one back-door sampling slot was missing, while the other two had none.
This run placed back-door and open-bulb-cam on FP16 device 0 and Driveway on
INT8 device 1, so its aggregate latency is not an equivalent allocation to the
earlier compressed run. Both NPU chips of physical card 8217 were used.
Camera/RTSP transport problems are outside this codec optimization.

The offline review embeds raw RGB frames and paints them to a canvas, avoiding
JPEG/PNG conversion for previews. This keeps file-based review self-contained
but increases the size of the generated HTML. Keep the artifacts private.

The hardware RGB path follows the documented
[310P VDEC output formats](https://www.hiascend.com/doc_center/source/en/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_000102.html);
Pillow documents [PPM support](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#ppm).
Implementation and codec-bypass behavior were validated against the installed
CANN headers and runtime, not inferred solely from these references.

## Shared serving across the card

The current serving experiment has one API and one model partitioned across
logical devices 0 and 1 with tensor parallelism (TP2). Each rank holds 6.378 GB
of model weights, including the INT8 text projections and FP16 vision encoder
and vocabulary head. Both chips belong to physical card 8217. This is two NPU
chips, not two individual AI compute cores. Devices 2 and 3 remain assigned to
the existing GLM workload.

The initial shared service uses eager execution, 1,024 batched tokens, eight
active sequences, four host threads per worker, and disabled prefix/image
caches. `capacity-tp2-int8-motion` contains two batches per concurrency level:

| Concurrent motion windows | One INT8 chip: mean batch | Shared TP2: mean batch |
| --- | --- | --- |
| 1 | 6.18 s | 4.84 s |
| 2 | 10.23 s | 7.21 s |
| 3 | 12.25 s | 8.48 s |
| 4 | 14.58 s | 10.05 s |

Output lengths vary, so these are workload measurements rather than
equal-token kernel speedups. The three-request TP2 test reaches 0.354 windows/s;
three streams arriving every eight seconds require 0.375 windows/s. Increasing
the prefill budget to 2,048 tokens did not improve these results: three requests
averaged 9.05 seconds and four averaged 10.00 seconds.

`paced_capacity.py` schedules windows independently of response time, rotates
saved scene fixtures, and retains backlog and explicit drops. In
`paced-tp2-int8-three-motion`, three streams delivered six arrivals each at an
eight-second interval. All 18 descriptions completed, with no drops or errors.
Earlier pending requests at arrivals were `[0, 2, 1, 2, 1, 2]`. Median response
time was 9.25 seconds and the maximum was 12.66 seconds; ten responses exceeded
the arrival interval. This short replay had no growing backlog across its six
arrival points. It does not establish an eight-second response deadline or
long-running live camera capacity.

```bash
python -m tools.rtsp_scene.paced_capacity \
  --captures /private/capture-a /private/capture-b \
  --output /private/paced-results --streams 3 --windows 6 --interval 8 \
  --endpoint http://127.0.0.1:8003/v1/chat/completions
```

The shared eager trace contains 11 single-request decode forwards per rank.
Quantized projections take 53.0–55.6% of recorded kernel time, HCCL all-reduces
18.7–20.2%, the FP16 vocabulary head approximately 8.8%, and format conversion
approximately 4.7–5.0%. Recorded device activity occupies 78.7–79.9% of the
roughly 492 ms decode span. Profiling adds overhead; task fractions and gaps
do not attribute CPU latency on the critical path. Native profiling is parsed
offline with `torch_npu.profiler.profiler.analyse` because the serving workers
are daemon processes. INT8 QKV outputs use NZ layout; the parser checks the
per-token scale shape to establish M=1 rather than assuming a padded NZ block
means one logical row. Reports are in `npu-profile-tp2/rank*-summary.json`.

The selected native graph trace also records 11 decode forwards per rank,
spanning 369.14–369.90 ms, versus 491.71–491.76 ms in the eager trace.
Recorded quantized projections total 209.46–216.83 ms, the FP16 vocabulary
head approximately 34.3 ms, and conversion remains visible. Captured graphs
do not expose the same composite HCCL metadata: only ten `hcom_allReduce_`
entries appear versus 802 in eager mode. Missing labels do not establish
that communication work vanished. Device activity fractions and operator
percentages therefore must not be compared as a complete communication or
CPU latency accounting. These are bounded diagnostic traces, not serving
throughput measurements. Reports are in `npu-profile-tp2-graph8`.

These measurements prioritize graph replay, quantized projections and
communication. The pinned ModelSlim implementation does not support INT8
`ParallelLMHead`/`VocabParallelEmbedding`; the head remains FP16. Small
normalization fusions alone do not address the dominant recorded costs.

### Native decode graphs

The earlier `attn_params` failure was caused by invoking the parent ACL
attention task updater for native torch NPU graphs. The parent initializes
its graph parameter registry only when `use_aclgraph` is true. The 310P
attention builder already binds persistent device metadata for native replay.
`NPUModelRunner310._update_full_graph_params_if_needed` now skips the ACL
updater for native graphs and delegates unchanged for ACL graphs. This fix
adds no model-specific patch or mutable global state. It needs architecture
review before upstream integration; its hardware validation here is Qwen3-VL
on 310P, not all models or all Ascend hardware.

Testing uses the complete isolated runtime worktree at
`/srv/ai/src/vllm-ascend-rtsp-graph`, based on remote commit `e5ad0bf1d` with
this override. It reuses the compiled extension and custom OPP from that same
runtime and retains CANN's Python binding paths. The primary remote source
checkout is unchanged. The local source revision is newer, so local host
regression coverage and this remote hardware check are distinct validations.

TP2 captures two decode sizes, `[1, 4]`, with compile mode zero and
`FULL_DECODE_ONLY`. This respects the observed 310P HCCL graph event budget.
Prefill and batches above the captured maximum run eagerly. Real image
requests passed at concurrency one through four, including changing request
lengths and mixed arrival/prefill/decode work. Capture startup alone is not
the acceptance criterion.

Using identical JPEG fixtures and saved prompts to the TP2 eager run,
`capacity-tp2-native-graph-motion` measured mean batch completion times of
**3.87, 6.35, 7.87 and 9.21 seconds** at concurrency one, two, three and four.
All 20 descriptions passed schema/evidence validation. Output wording and
length differ, so these remain workload comparisons rather than identical
output speedups. The historical JPEG fixtures preserve the baseline inputs;
live capture and the raw replay below use the default raw RGB path.

`capacity-tp2-native-graph-raw-replay` uses raw return-scene frames first and
departure frames second. It measured 5.00, 6.61, 7.96 and 9.61 seconds for
those concurrency levels. Its input order differs from the eager motion
baseline; it must not be used to attribute a graph speedup.

A native dense DP2 comparison was rejected after hardware placement checks:
both replicas ran on chip 0 while chip 1 remained idle, including a retry with
explicit `--device-ids` and one API process. No two-chip DP2 throughput claim
is made, and the launcher does not expose this failing configuration. Logs
and the invalid measurements remain in `serve-dp2*.log`,
`dp2-explicit-device-smi.txt` and `INVALID-capacity-dp2-both-chip0` for runtime
debugging. This is separate from the verified TP2 chip placement.

An additional TP2 candidate captures `[1, 8]` to cover the full eight-sequence
decode budget. Its identical-fixture batch test averaged 4.31, 8.41 and
10.50 seconds at concurrency one, three and four. Outputs varied in length,
including one 145-token response. It is slower on these small batches than
`[1, 4]`, so selection also requires the independently paced larger-queue
test rather than treating a larger captured batch as an automatic improvement.

The selected `[1, 8]` configuration then completed a longer four-stream replay:
**120 windows, one window per stream every eight seconds, zero drops and zero
errors**. The 30 arrivals span 232 seconds; all responses finished in 242.49
seconds. Earlier pending requests were mostly four, with brief increases to
five or six and four pending at the final arrival. Median response time was
15.20 seconds, maximum 16.69 seconds; all responses exceeded eight seconds.
`paced-tp2-native-graph8-four-motion` retains the complete arrival and response
records. This supports four buffered streams for this four-minute, two-scene
replay on the entire two-chip card. It does not establish four streams per
chip, an eight-second response deadline, unsampled-frame coverage, or a
production-duration guarantee for arbitrary scenes. Prefix and image caches
remain disabled. The replay includes historical JPEG departure frames and
pixel-matched raw RGB return frames; default live input remains raw RGB.

During this unprofiled replay, Linux process counters measured 858.31 CPU
seconds across 242.59 wall seconds for the API, engine and both workers:
approximately **3.54 CPU cores on average**. The engine used 244.32 CPU seconds,
workers 295.51 and 295.37, and API 22.42. These counters include polling/waits
that consume CPU; they exclude the replay client and camera capture. CPU
work overlaps NPU execution, so this is host resource use, not a percentage
of response latency. Details are in the replay's `server-cpu.json`.

The selected service also passed a live three-camera raw RGB check using
back-door, Driveway and open-bulb-cam, all through the single API endpoint.
Hardware VDEC processed H.264 and 4K HEVC sources. It sampled 96 frames and
completed 12 descriptions in 37.55 seconds, including capture startup and the
final response. There were zero capture failures, inference failures, drops
or missed sampling slots. Requests took 4.85–5.71 seconds on these mostly static
live scenes. Capture used 0.84–0.98 CPU seconds per camera and copied zero JPEG
bytes; video decode and resizing run on DVPP and the model reads raw PPM.
The live evidence is in `live-three-tp2-raw`; it complements rather than
replaces the busier motion replay capacity test.

`paced-tp2-native-graph-four-motion` delivered ten four-stream arrivals at an
eight-second interval. All 40 descriptions completed in 82.59 seconds with
no errors or drops. Earlier pending requests were
`[0, 3, 4, 4, 4, 5, 4, 4, 4, 6]`. Median response time was 14.28 seconds,
maximum 17.30 seconds; 39 responses exceeded eight seconds. This validates a
short four-stream replay with buffering, not a long-running capacity or
eight-second response guarantee. The replay excludes camera transport and
reuses two scenes; it does not establish scene-description accuracy.

## Specialized INT8 projections and NZ FFN experiments

`dense_i8.cpp` implements a narrow INT8 Cube projection with frozen
output-tile-major weights, cached activations and optional overlapping weight
DMA. It accepts ND or NZ activations and writes ND or NZ results directly.
`nz_swiglu.cpp` keeps gate/up output in NZ through SwiGLU, FP16 activation
rounding, per-token quantization and NZ packing for the down projection.
There is no ND activation intermediate in global memory between these kernels.

These operators have isolated hardware gates and an explicit serving
integration through `SceneDenseWorker`. The serving worker retains the
original modules for unsupported shapes and charges the additional resident
weights before KV cache allocation. It preserves the existing row-parallel
collectives. No custom OPP package or upstream model patch is installed by
the build tool. The isolated timings below exclude the rest of the model;
whole-model measurements are recorded separately.

The frozen geometry is the Qwen3-VL 8B **TP2** text decoder: QKV `(N,K)` is
`(3072,4096)`, attention output `(4096,2048)`, gate/up `(12288,4096)` and down
`(4096,6144)`. Matrix kernels accept 1–16 rows; fused NZ activation accepts
1–8 rows and hidden width 6144. The serving integration must retain the
existing path for other geometries. Packed INT8 weights have physical shape
`[N/tile, K/32, tile/16, 16, 32]`. Each QKV/gate/up piece is sharded separately
before concatenation. Row-parallel shards retain the checkpoint's original
per-channel scales without requantizing their partial K dimensions.

On October 12, 2026, the installed CANN 9.1.0 direct
`aclnnQuantMatmulWeightNz` probe reproduced the Torch NPU result exactly with
ND output, but its NZ output declaration failed numerical checks despite
successful API return codes. That route was rejected. The custom matrix
kernel writes the NZ storage itself and passes cell-by-cell parity checks.
An earlier fused quantizer using FP32 multiplication and ties-to-even also
failed the FFN gate. The accepted implementation reproduces the installed
operator's vector FP32 scale calculation, FP16 reciprocal multiplication and
ties-away conversion, including zero and very small activations.

Hardware validation uses real layer-0 checkpoint weights and synthetic
activations on both shards of card 8217; devices 2 and 3 remain dedicated to
the unrelated GLM workload. Both output formats and 1, 4 and 8 token batches
are checked against the existing CANN path. Each projection uses three
changed inputs plus a captured graph replay with changed inputs. FFN checks
include input quantization, both matrix projections, SwiGLU and activation
quantization; the initial FP16 input remains mutable across graph replays.
Activation checks cover magnitudes 0, 0.0001, 0.01, 0.1, 1 and 3. Accepted
matrix/FFN outputs and activation codes must be exactly equal, with finite
results; scales are checked separately for parity. Timings alternate the
baseline/candidate order over eight trials of 100 calls with warm resident
weights. They include dispatch/replay overhead and exclude norms, attention,
LM head and TP communication. They are not whole-model decode timings.

Final FFN graph measurements for the 256-column/256-inner pipelined build:

| Chip / TP shard | Tokens | Reference | NZ candidate | Time reduction |
| --- | --- | --- | --- | --- |
| 0 / 0 | 1 | 484.2 μs | 431.8 μs | 10.8% |
| 0 / 0 | 4 | 493.1 μs | 433.9 μs | 12.0% |
| 0 / 0 | 8 | 503.6 μs | 443.6 μs | 11.9% |
| 1 / 1 | 1 | 453.6 μs | 426.3 μs | 6.0% |
| 1 / 1 | 4 | 469.9 μs | 429.8 μs | 8.5% |
| 1 / 1 | 8 | 474.9 μs | 438.4 μs | 7.7% |

All final activation codes, scales and FFN outputs matched the reference in
the tested cases. The hardware nightly gate passed independently on both
chips, and the scoped CPU suite passed 44 tests plus two subtests. These are
token batch sizes, not camera counts. Earlier paired runs varied from about
7–16% FFN improvement; treat the table as a bounded measurement rather than
a guaranteed speedup. Full receipts remain outside git as
`dense-kernels-final-report.json` and the `dense-nightly-rank*-final` artifacts.

The paired matrix sweep shows projection-dependent gains: QKV and attention
output improve, gate/up has a smaller gain, and standalone down can be
slightly slower. The whole local FFN is the relevant test for NZ retention;
an activation-only speedup is insufficient. The measured FFN improvement is
recorded separately for each chip and batch size in the private benchmark
receipts. CPU tests cover TP shard/scale handling, every packed Cube cell,
L0B tile capacity and rejection of a mixed binary build.

Build and run in the pinned Ascend environment, with the scene service stopped
and the selected device isolated from other inference work:

```bash
python -m tools.rtsp_scene.build_dense \
  --output /private/dense-build --columns 256 --inner-tile 256 --pipeline
python -m tools.rtsp_scene.dense_bench \
  --model /private/Qwen3-VL-8B-Instruct-w8a8-text \
  --build /private/dense-build --output /private/dense-results.json \
  --rank 0 --tokens 1 4 8 --iterations 100 --repeats 8
python -m tools.rtsp_scene.nz_bench \
  --model /private/Qwen3-VL-8B-Instruct-w8a8-text \
  --build /private/dense-build --output /private/nz-results.json \
  --rank 0 --iterations 100 --repeats 8
```

Select the physical chip with the existing `ASCEND_RT_VISIBLE_DEVICES`
setting before launching an isolated worker. Repeat with `--rank 1` for the
other checkpoint shard. Each build is a new directory; its manifest records
the compiler target, tile geometry, pinned Torch version and SHA256 of its
sources, bridge and binaries. Benchmarks verify binary fingerprints before
loading. Use a fresh process for each native build, because Torch library
registration and captured graph references outlive an individual timing call.

The nightly hardware gate runs both benchmarks, requires exact parity and
requires the complete NZ FFN graph to outperform its reference for each batch:

```bash
python -m pytest -q --confcutdir=tests/e2e/nightly/rtsp_scene \
  tests/e2e/nightly/rtsp_scene \
  --scene-dense-model /private/Qwen3-VL-8B-Instruct-w8a8-text \
  --scene-dense-build /private/dense-build --scene-dense-rank 0
```

It skips when no checkpoint/build is provided. The gate must be repeated on
the actual hardware and workload before enabling these operators in serving.

### Opt-in serving integration

`dense_setup.py` creates a fresh symlink overlay of the quantized checkpoint
and freezes the native build manifest. It does not modify the source
checkpoint. `launch.py --dense` selects an instance-local worker subclass;
there is no new environment variable or shared runtime patch. Mode `ffn`
keeps gate/up, SwiGLU quantization and down input in NZ. Mode `all` also
replaces the text QKV and attention-output quantization methods for eligible
inputs, retaining their original linear forwards and TP communication.

The optimized path accepts contiguous FP16 inputs with 1–8 rows and the
validated TP2 geometry. Other inputs, including ordinary large prefill
batches, use the original path. Original weights remain resident for this
fallback. On the two test chips, mode `all` added 3,473,728,000 bytes
(3.24 GiB) per chip; the worker includes this in model memory usage before
profiling cache headroom.

Before serving, each worker checks all 36 FFNs with their actual TP
reductions, plus all QKV and attention-output projections in mode `all`.
Checks use 1 and 8 rows at activation magnitudes 0.1, 1 and 3. All 648
comparisons on each chip were bitwise equal in the October 12–13 serving
gate. Startup records both equality and a tolerance gate and refuses to
serve if a comparison fails. Captured-graph inference and full scene
requests are tested separately after startup.

```bash
python -m tools.rtsp_scene.dense_setup \
  --model /private/Qwen3-VL-8B-Instruct-w8a8-text \
  --build /private/dense-build --output /private/dense-overlay --mode all
OMP_NUM_THREADS=4 python -m tools.rtsp_scene.launch /private/dense-overlay \
  --port 8003 --devices 0,1 --tensor-parallel-size 2 \
  --max-sequences 8 --max-batched-tokens 1024 \
  --int8 --decode-graph-sizes 1 8 --dense
```

Use the pinned environment and complete runtime worktree described below.
To restore the control, restart the same launcher with the original
checkpoint and omit `--dense`. Do not load different native builds into
the same worker process.

`fixed_decode.py` reuses the exact scene-request prompts and images from a
natural-description capacity run, with 64 generated tokens and two excluded
warmup requests by default. Its truncated outputs are diagnostic only.
`serving_compare.py` refuses comparisons unless request body hashes and
token workloads match. Natural descriptions use their separate validated
JSON results; their varying output lengths make wall time alone an
unreliable measure of kernel improvement. Even the control's wording
changed across restarts in the tested environment.

```bash
python -m tools.rtsp_scene.fixed_decode \
  --reference /private/natural-control --output /private/fixed-candidate
python -m tools.rtsp_scene.serving_compare \
  --control /private/fixed-control --candidate /private/fixed-candidate \
  --output /private/serving-comparison.json
```

The warmed control and full candidate used matching request body hashes and
input/output token counts, with two batches per concurrency and 64 completion
tokens per request. Prefill remained within 0.4% of the control. These are
bounded whole-model serving measurements on both chips of card 8217:

| Concurrent requests | Control decode mean | Full candidate decode mean | Decode time reduction | Total batch time reduction |
| --- | --- | --- | --- | --- |
| 1 | 2.111 s | 1.869 s | 11.5% | 7.4% |
| 3 | 3.685 s | 3.406 s | 7.6% | 4.6% |
| 4 | 4.352 s | 4.044 s | 7.1% | 4.4% |

FFN-only integration reduced decode time by 5.4–7.0% in the same cohorts.
The full candidate also completed all 16 natural scene-description requests
with valid JSON. This establishes serving compatibility, not calibrated
scene accuracy or a higher camera count. The expanded CPU suite passed 52
tests plus two subtests. Private receipts are `dense-serving-all-comparison.json`,
`dense-serving-all` and the overlay's `dense-gate-rank*.json`.

`paced-dense-all-four-motion` completed all 120 descriptions from 30
four-stream arrivals at eight-second intervals in 241.80 seconds, with zero
errors or drops. Earlier pending requests settled at four, with a maximum
of five; there was no growing backlog over this finite test. Median
response time was 14.57 seconds and maximum was 16.08 seconds. Every
response exceeded the eight-second arrival interval: this is buffered
throughput across **both chips**, not an eight-second response deadline,
an always-on capacity guarantee or four streams per chip. The replay uses
two saved motion fixtures and excludes RTSP transport.

The serving process group consumed 860.74 CPU seconds over 241.90 wall
seconds, averaging 3.56 CPU cores during that replay. This includes busy
waits and overlapped host work and excludes capture/client processes. It
does not measure the CPU share of request latency. One replay fixture is
historical JPEG and the other is raw PPM; the live default uses DVPP raw
RGB/PPM transport. These matrix/NZ changes do not remove JSON, base64,
host copies or image preprocessing.

The full candidate also passed `live-three-dense-all-raw`: 12 descriptions
from back-door, Driveway and open-bulb-cam in 37.44 seconds, including capture
startup and final responses. Individual inference requests took 4.13–5.50
seconds on these mostly static scenes. All 96 sampled frames came through
DVPP H.264/HEVC decode and resizing with zero JPEG bytes copied to the host.
There were zero inference errors, capture failures or budget drops. The
back-door source had one missed sampling slot and a 1.24-second input PTS
gap; the decoded timestamps retain it. This finite live check does not
establish gap-free camera transport.

The full candidate's bounded native profile confirms 1,584 custom matrix
calls and 396 NZ SwiGLU calls per chip, corresponding to 11 decode forwards
through all 36 layers. The recorded decode span was 331.93–331.96 ms; the
earlier native control diagnostic recorded 369.14–369.90 ms. These profiled
spans support the serving measurements but are not throughput benchmarks.
Custom matrices still represent about 72% of recorded kernel time, the
FP16 vocabulary head about 12%, and attention about 5%. Communication
metadata remains incomplete in native graph traces; these percentages do
not give a complete communication or CPU latency accounting. The phase
starts at the first custom projection, excluding the preceding norm, and
checks the native call counts against the single-request 12-token probe.
Receipts are in `npu-profile-dense-all/dense-summary.json`. The selected
service is left running in mode `all`, with profiling stopped.

### Eight-camera arrival tests

The current TP2 service was tested with eight simultaneous requests from the
same two saved motion fixtures. Four eight-request batches completed in
15.40–15.59 seconds each, averaging 0.516 descriptions per second. Eight
streams at an eight-second interval require 1.0 description per second,
about 1.94 times this measured rate. These inputs do not represent eight
distinct bodega cameras or an accuracy evaluation.

| Streams | Arrival interval | Scheduled | Completed | Dropped | Median response | Maximum response |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 8 s | 80 | 65 | 15 | 43.15 s | 61.27 s |
| 8 | 16 s | 96 | 96 | 0 | 14.65 s | 15.41 s |

The eight-second run's pending count grew to 28 before arrivals and hit the
32-request budget. It demonstrates overload, not a capacity increase from
buffering. The sixteen-second run had zero earlier pending requests at all
12 arrivals and zero failures; it completed in 191.39 seconds. Its modest
timing margin does not establish always-on capacity for busier scenes.
Receipts are `capacity-dense-eight-motion`, `paced-dense-eight-motion-8s`
and `paced-dense-eight-motion-16s`.

The replay changes arrival timing and still submits eight original frames
per request. To analyze consecutive sixteen-second live windows with eight
frames each would require sampling at 0.5 FPS and validating the resulting
action visibility. Submitting only every other existing eight-second window
would leave analysis gaps. Continuous recording, sampled observations and
continuous scene descriptions are separate coverage claims.

A larger-batch trial set `--max-sequences 16` and decode graph sizes `1 16`
without changing weights or image quality. The custom methods support at
most eight rows, so the 16-row graph uses the original CANN paths. It
successfully completed 32 natural descriptions, but delivered only 0.518
windows/s versus 0.527 windows/s for sixteen requests queued on the selected
eight-sequence service. Output lengths differ, so this is a bounded workload
comparison, not a kernel speedup measurement.

Matching 64-token diagnostics at eight concurrent requests took 14.07 seconds
per candidate batch versus 13.43 seconds for the control: 4.7% slower.
At sixteen concurrent requests, the candidate took 25.89 seconds per batch.
The first control took 33.20–39.18 seconds, but a fresh restored control
with the same body hashes and token counts took 26.05 seconds per batch.
The candidate's apparent 28.5% improvement shrank to 0.6% against that
repeat. It is not a demonstrated throughput gain. The selected eight-row
graph configuration was restored and passed a raw-frame scene smoke test.
Receipts include `sixteen-batch16-repeated-comparison.json`.

Further candidates include extending NZ activation to sixteen rows while
retaining the existing 16-row Cube tile, optimizing vision/text prefill,
and reducing redundant narration through motion/POS event prioritization.
These are unvalidated follow-up experiments. Larger decode batches alone
did not meet the eight-stream/eight-second target in this study; slower
sampling or event prioritization would change the analysis coverage and
must be evaluated explicitly.

## Start the model

Activate the Ascend environment and virtualenv first. On the test host:

```bash
source /srv/ai/bin/ascend-env.sh
source /srv/ai/venvs/qwen38-w4-test-ce1862/bin/activate
export PYTHONPATH="/srv/ai/src/vllm-ascend-rtsp-graph:$PYTHONPATH"
OMP_NUM_THREADS=4 python -m tools.rtsp_scene.launch \
  /srv/ai/storage/models/Qwen3-VL-8B-Instruct-w8a8-text-20261012 \
  --port 8003 --devices 0,1 --tensor-parallel-size 2 \
  --max-sequences 8 --max-batched-tokens 1024 --int8 --decode-graph-sizes 1 8
```

The API binds to loopback. This command starts the shared native-graph control
in the complete patched runtime. Use the prepared overlay with `--dense`
as shown above for the selected optimized service.
Omit `--decode-graph-sizes` for eager execution.
`--devices` must name exactly TP distinct devices. Optional
`--decode-graph-sizes` selects native
decode capture with at most two sizes on 310P. Graph use requires the tested
runtime fix described with the results; capture-only startup is not a valid
inference check. A missing optional request-timing plugin is nonfatal;
measurements use client wall time and native server metrics.

## Sample and describe cameras

Requires Python 3, FFmpeg, and FFprobe. The runner itself uses only the Python
standard library. Create a private JSON configuration outside the repository:

```json
[{"name": "test-camera", "url": "rtsp://camera.example/live", "enabled": true}]
```

Use authenticated camera URLs as needed and `chmod 600` the configuration.
Disabled cameras and cameras in privacy mode are excluded.

```bash
python tools/rtsp_scene/runner.py \
  --config /private/cameras.json --output /private/scene-results \
  --camera test-camera --iterations 2 --dvpp-binary /private/dvpp_decode
```

Omit `--camera` to sample all enabled cameras. Captures run concurrently;
model requests are submitted sequentially. Use `--capture-only` to save frames
without inference. Use `--clip /private/recording.mp4` with exactly one selected
camera to sample an existing recording, or `--reuse /private/capture-directory`
to analyze previously sampled frames. Clip sampling preserves relative PTS;
it does not invent a UTC anchor.

`--prompt-file /private/prompt.txt` supports controlled prompt experiments.
The runner records the exact prompt and its SHA-256 hash. Requests constrain
the response JSON schema; post-validation checks evidence frame order as well.
Schema conformance does not establish factual correctness. Invalid or truncated
responses remain in `response.json` for diagnosis and are rejected as results.

Each capture directory contains raw RGB PPMs (or explicitly selected JPEGs),
relative frame times, capture metadata,
the model response, and a validated `result.json` when successful. Error messages
redact RTSP URLs, including credentials and query tokens. Output is private by
default. No images or credentials are sent to Hugging Face; inference is local.

## Review and verification

```bash
python -m tools.rtsp_scene.review /private/scene-results
python -m unittest discover -s tests/ut/rtsp_scene -v
ruff check tools/rtsp_scene tests/ut/rtsp_scene
shellcheck tools/rtsp_scene/serve.sh
```

Open the generated `review.html` beside the artifacts. Its frame sliders allow
comparison of descriptions with their evidence. The tool does not start a web
server or publish footage. Keep the directory private.

Host artifacts are under `/srv/ai/storage/artifacts/rtsp-scene-20261012`.
The current shared TP2 API is at `http://127.0.0.1:8003`, with native decode
graphs `[1, 8]`, INT8 text projections and FP16 vision/head weights on devices
0 and 1. The earlier independent replicas are stopped; port 8004 is no longer
serving. Current logs are in the `rtsp-scene-model.service` and
`rtsp-scene-log.service` journals. The selected launch command and runtime
path are in `next-ops-20261013/selected-command.json`; the API PID is in
`serve-current.pid`. Profiling is configured but inactive outside explicit
diagnostic requests. Never stop the unrelated GLM processes on devices 2 and 3.
Measurements are retained in `projections-int8.json`, `capacity-int8-motion`,
`quality-int8-text`, `quality-int8-object-crop`, `jpege-clip`,
`jpege-four-cameras`, `live-two-fp16-jpege` and `live-four-mixed-jpege`
under the private host artifact directory. Raw input evidence is retained in
`raw-rgb-clip`, `transport-pixel-matched`, `media-input-bench.json` and
`live-three-raw-rgb`. The current tests cover credential redaction,
missing and extra filtered frames, invalid event evidence, static scenes, and
escaping model text in the review page, explicit TP device selection, native
graph/ACL updater dispatch, NZ trace classification and fixed-rate replay
with bounded drops. All 67 scoped host tests, two subtests and scoped pre-commit
checks pass.

## 16-row kernel follow-up

New builds can freeze `--max-decode-rows 16`. The NZ activation assigns a
second row to each of the eight cores, reusing its buffers after the first
row completes. Matrix kernels already operate on a padded 16-row Cube tile.
The runtime takes its row limit from the frozen build, retains the original
path above that limit, and validates loaded weights at rows 1 and the selected
maximum. Old manifests retain their eight-row limit. The initial staging
did not change the live service's eight-sequence scheduling or its existing
graph sizes.

Build `dense-i8-rows16-v14` passed real layer-0 shard comparisons on both chips
at rows 1, 8, 9 and 16. Each QKV/output/FFN case checks five input magnitudes
and three changed-input graph replays against CANN. Both frozen builds must
match exactly before the paired graph timing runs. The 9/16-row activation
probe on chip 1 also matched FP16 activation values, INT8 codes and scales
at all six tested magnitudes, including zero and very small values.

The subsequent check-only sweep covered all 36 layers on each chip, using
rows 1 and 16 at all five magnitudes: 1,080 candidate-to-CANN comparisons per
chip, all finite and bitwise equal. Its private receipts are
`kernel-rows16-all-layers-rank0.json` and
`kernel-rows16-all-layers-rank1.json`. These are local shard checks, excluding
TP collectives and full-model inference. Use `--layers 0 1 ... 35 --check-only`
to repeat them without timing claims.

The separate full-service candidate on port 8005 failed at HCCL communicator
initialization: the NPU socket on port 16666 was already occupied by the live
service. Both `auto` and an explicit range for the vendor port setting still
produced that conflict in this runtime. The
[CANN port-setting documentation](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0144.html)
lists applicability to later Atlas products, so support on 310P cannot be
assumed. At that stage, no candidate HTTP inference or serving performance
had been validated.
The failed candidate processes exited, and the original service remained
on port 8003 at that stage. The authorized sequential rollout below
subsequently cleared this conflict. Overlay creation now excludes inherited
`dense-gate-rank*` receipts so a candidate cannot reuse or overwrite earlier validation results.

These checks ran alongside the continuous four-camera logger. Timings are
contended diagnostics, not isolated performance gates or stream-capacity
measurements. The paired 16-row FFN medians were 2,686/2,035 microseconds on
chip 0 and 1,059/831 microseconds on chip 1 (old fallback/new NZ path), with
substantial trial variation. No whole-model or sustained-camera speedup is
established by these measurements. Private receipts are
`kernel-rows16-v14-rank*-shared.json` and `nz-rows16-rank1-shared.json`.

An optional `--nd-scatter` build removes the intermediate UB transpose and
writes NZ cells directly into ND rows using strided DMA. Its corrected build
`dense-i8-scatter16-v15` passed parity and mutable-graph checks on chip 1,
but did not show a consistent timing improvement at existing decode sizes.
It remains an experiment and is not selected for the serving candidate.

`kernel_compare.py` compares the old custom path (and its larger-batch CANN
fallback) against a candidate in one process. Separate frozen Torch
namespaces avoid duplicate operator registration. It refuses timings after
nonfinite outputs, mismatched values or stale graph outputs; CPU tests cover
these failure modes. The nightly hardware gate includes rows 9 and 16 for
builds with that frozen limit and still requires an isolated device for its
performance assertions.

```bash
python -m tools.rtsp_scene.build_dense \
  --output /private/rows16-build --columns 256 --inner-tile 256 --pipeline \
  --max-decode-rows 16 --namespace rtsp_dense_rows16
python -m tools.rtsp_scene.kernel_compare \
  --model /private/Qwen3-VL-8B-Instruct-w8a8-text \
  --control /private/previous-build --candidate /private/rows16-build \
  --output /private/paired-kernels.json --rank 0 --tokens 1 8 9 16
```

Add `--shared-device` when other inference work remains active; that option
labels the receipt and does not isolate the hardware. Devices 2 and 3 are
outside this experiment.

### Live mixed-workload profile

A new bounded profile used the continuous logger's existing requests, with no
replay request added. Both ranks recorded approximately 6.459 seconds of
device tasks; profiling was stopped before offline parsing. Receipt:
`live-mixed-profile-summary.json`, with raw directories recorded in
`live-profile-directory-state.json`. This includes vision, prefill and decode,
unlike the earlier decode-only diagnostic.

On rank 0, custom INT8 matrices account for 18.4% of recorded task duration,
prefill attention for 11.3%, large-batch quantized projections for 10.9%,
LayerNormV3 for 7.4% and vision encoder attention for 4.8%. Rank 1 shows the
same broad work distribution, with more recorded all-reduce time. Those are
task-duration fractions, not CPU/request latency attribution; tasks may
overlap and native traces omit some communication metadata.

The 65 vocabulary-head outputs on each rank have active row counts
`{1: 31, 2: 29, 3: 4, 4: 1}`. Quantized prefill projections have logical row
counts 227, 945, 950, 981 and 1024. The vision LayerNorm shapes include
hidden dimensions 1152 and 4608. This supports targeting prefill attention,
vision normalization and batching alongside narrow decode kernels. It does
not establish a cache hit rate or explain the entire replay/live difference.

The live four-camera run has reached its request budget and logged dropped
analysis windows. The earlier finite replay remains valid for its saved-frame
workload; it does not establish sustained four-camera live capacity. The
16-row kernels were still staged when this profile was captured. The
sequential rollout below validates the candidate without a competing TP2 server.

### Sequential rollout and isolated checks (October 13)

With the continuous logger stopped and its pending requests drained, the old
TP2 server was stopped before starting each trial on port 8003. Removing the
competing communicator cleared the HCCL bind failure. The 16-row candidate
loaded successfully, produced fresh regular validation receipts with all
648 loaded-weight comparisons per rank passing at rows 1 and 16, and
returned a valid description for an eight-frame raw RGB request. The pinned
runtime and unrelated workload on devices 2 and 3 were retained.

The isolated layer-0 paired graph benchmark used 100 iterations and eight
repeats on each chip. At 16 rows, QKV time fell 9.9–11.5%, output projection
time fell 5.1–5.3%, and FFN time fell 7.5–9.5%. Rows 1 and 8 changed by about
1% or less. All numerical and changed-input graph checks passed. These
isolated timings supersede the earlier contended speedup estimates; they do
not establish whole-model or camera throughput gains.

A separate matched serving comparison used the same saved frames and
prompts, verified request-body hashes and token usage, and generated exactly
64 tokens per request. Each cohort used two batches after two excluded
single-request warmups. The original control used eight sequences, graph
sizes 1/8 and a 1,024-token budget. The 16-row trial used 16 sequences and
graph sizes 1/16. A further trial raised the token budget to 4,096:

| Concurrent requests | Control batch mean | 16-sequence trial | 4,096-token trial |
| --- | --- | --- | --- |
| 1 | 3.145 s | 3.151 s | 3.148 s |
| 8 | 13.126 s | 13.587 s | 13.867 s |
| 16 | 26.023 s | 25.546 s | 25.406 s |

The wider graph slowed the eight-request cohort by 3.5%; its 16-request
batch reduction was only 1.8%. Raising the token budget reduced the
16-request batch by 2.4% but slowed the eight-request cohort by 5.6%.
That trial retained the original kernel
overlay, eight sequences, graph sizes 1/8 and the 1,024-token budget. The
validated 16-row overlay remains staged. No sustained stream-capacity
increase is claimed. These diagnostic outputs are truncated and are not
accepted scene reports.

The 16-row overlay was also tested with the original eight-sequence settings
under systemd. Matched batch means were 3.245, 13.259 and 26.160 seconds
for concurrency 1, 8 and 16, respectively: 3.2%, 1.0% and 0.5% slower than
the original control. This finite run does not establish a kernel-induced
regression, but gives no reason to replace the live kernel build. Receipt:
`fixed-selected-comparison.json`; its candidate command is retained as
`validated-candidate8-command.json`.

The pinned vLLM scheduler sets both its encoder compute and cache budgets
from `max_num_batched_tokens`, with a floor for the largest permitted image.
The 4,096-token trial changes vision budgeting and text prefill together;
its result cannot be attributed solely to either stage.

Private receipts are under `rows16-rollout-20261013`: isolated kernel
reports, exact command records, `fixed-comparison.json`,
`fixed-budget-comparison.json`, and the installed systemd unit definitions.
The API readiness helper retries refused connections and non-200 health
responses until a monotonic deadline. Scoped tests cover recovery, bounded
timeout and invalid timing arguments.

The coordinated systemd restart was hardware-validated: the logger stopped
at 04:19:19 UTC before the model stopped, and the model became ready at
04:21:19 UTC before the logger restarted. The first 37 seconds after the
logger initialized produced 13 valid descriptions across all four cameras,
with mean inference latency 5.15 seconds and maximum 6.06 seconds. There
were no inference errors or exhausted-budget drops in this short check.
FrontDoor retained six within-window capture PTS gaps. This is a restoration
check on the current scenes, not a sustained motion/capacity gate. Private
receipts: `restart-order.json`, `live-restored.json` and
`final-service-state.json`. Both units are enabled at boot. All 73 scoped
host tests, two subtests, manual pre-commit checks, launch-script ShellCheck
and systemd unit verification passed.

### Vision normalization follow-up

The October 13 live-only trace recorded vision `LayerNormV3` as about 5.5%
of summed device-task duration. The block inputs have shape `[N, 1, 1152]`;
merger normalization also uses two-dimensional inputs, including width 4608.
These are task-time fractions, not request-latency attribution.

`dense_setup.py --vision-zero-add-norm` freezes an optional normalization
path in a separate model overlay. For contiguous FP16 vision-block inputs
of shape `[N, 1, 1152]`, it calls `npu_add_layer_norm` with a shared zero
second input and requests the full sum output. The caller's FP16 input
rounding is retained. It neither fuses the preceding residual addition nor
modifies the two-dimensional merger path. Other shapes, dtypes, devices and
inputs exceeding the bounded zero buffer delegate to the original module.

The zero buffer is allocated once per worker and shared across all 54
vision-block normalizations. At the current 1,024-token budget it is
9,437,184 bytes per chip and is charged to model memory before KV-cache
profiling. No per-call zero allocation or host/device tensor transfer is
introduced in the normal inference path. A first request-activation check runs once per installed normalization
and is then removed. It is re-armed after startup warmup so dummy encoder
inputs cannot substitute for camera-activation validation.

Startup checks every loaded vision-block normalization and representative
larger row counts, requiring finite, bitwise-identical outputs. On both
chips, all 240 loaded-weight checks and all 54 first eligible model-activation
checks passed. Receipts are fresh overlay-local `vision-gate-rank*.json`
files; overlay creation excludes inherited text and vision gate reports.
The re-armed gate also passed all 54 checks per rank on the first actual
camera request, with shape `[3744, 1, 1152]`. Startup dummy checks are retained
separately and do not count as camera validation.

Rejected probes include inference-only normalization without a useful
speedup, genuine residual-add fusion that failed the FP16 parity gate, and
zero-add calls without a full sum output or on two-dimensional merger
inputs. Their timings do not support serving claims. In particular, the
first residual-fusion timing probe computed the reference addition twice;
those timing estimates are discarded. The selected path requires the
validated three-dimensional layout and full-output option.

`vision_norm_bench.py` loads layer-0 normalization parameters from the real
checkpoint and checks five activation magnitudes plus three changed-input
graph replays. It uses production's disabled-JIT setting; the default-JIT
baseline cannot be graph-captured on this runtime. The paired graph timing
alternates baseline/candidate order for 100 iterations and eight repeats.
Both chips passed all checks. Across the four observed row counts, isolated
normalization time fell 49.1–63.7%. This is a kernel-level result; serving
and sustained camera capacity require separate measurements.

```bash
python -m tools.rtsp_scene.dense_setup \
  --model /private/current-text-int8-overlay --build /private/current-build \
  --output /private/vision-zero-overlay --mode all --vision-zero-add-norm
python -m tools.rtsp_scene.vision_norm_bench \
  --model /private/text-int8-checkpoint --device 0 \
  --output /private/vision-norm-benchmark.json
```

The hardware nightly test is `tests/e2e/nightly/rtsp_scene/test_vision_norm.py`.
Pass `--scene-dense-model` and `--scene-dense-rank`; without a checkpoint the
test skips. It requires an isolated 310P and checks both parity and faster
paired graph execution. Private experiment receipts are under
`vision-followup-20261013` on the test host.

The matched raw-frame serving comparison used 16 captured windows with no
sampled image file shared between windows, frozen prompts, two excluded
warmups, two batches per cohort and exactly 64 completion tokens per request.
All 26 recorded request bodies and token usages matched between runs:

| Queued requests | Original batch mean | Zero-add norm batch mean | Time reduction |
| --- | --- | --- | --- |
| 1 | 3.217 s | 3.199 s | 0.6% |
| 4 | 7.315 s | 7.221 s | 1.3% |
| 8 | 12.403 s | 12.061 s | 2.8% |

Server prefill time fell 2.7–3.2% in these cohorts. This finite comparison
does not establish a sustained stream count or statistical significance.
Twenty-one of the 26 truncated output texts matched; generated text parity
is distinct from the strict normalization parity gate. Complete-description
probes produced valid JSON but varying captions, motivating a separate
identical-request repeatability control. Receipts:
`unique-zero-comparison.json`, `fixed-zero-text-check.json`,
`real-activation-gate-rank*.json` and the private natural-description runs.
The fresh original and zero-add servers each produced three distinct texts
for three sequential requests with the same body and temperature zero.
This establishes that variation also occurs on the original configuration;
it does not measure caption accuracy or prove statistical equivalence.
Receipt: `caption-repeatability.json`.

The launcher also accepts `--vision-parallel-mode data` to use upstream
Qwen3-VL image distribution across the two TP workers. Its default remains
`weights`. Text tensor parallelism, decode graph sizes and token budgets
remain independently configured. Replicating vision weights and distributing
images requires a separate memory, parity and serving check on this runtime;
the launcher option alone establishes no performance improvement.

Hardware trials retained the same text TP2 configuration and compared both
image distribution alone and image distribution with zero-add normalization:

| Queued requests | Original batch mean | Image distribution | Distribution plus zero-add |
| --- | --- | --- | --- |
| 1 | 3.217 s | 3.199 s | 3.192 s |
| 4 | 7.315 s | 7.088 s | 7.035 s |
| 8 | 12.403 s | 11.698 s | 11.631 s |

The combined trial reduced batch time by 3.8% at four queued requests and
6.2% at eight; mean server prefill time fell 6.7% and 8.0%, respectively.
The incremental batch improvement over distribution alone was below 1%,
so these short runs do not establish its statistical significance.
Every request body and 64-token workload matched the original control.
These remain truncated diagnostics, distinct from complete scene reports.
The natural-description trial started after the image-distribution timing
run completed; `dp-request-separation.json` records this separation.

The combined worker passed 648 loaded text-projection/FFN checks, 240 strict
vision-norm checks and 54 first-camera normalization checks per rank. The
camera normalization input was `[1872, 1, 1152]`, reflecting image
distribution. These gates check the custom operators, rather than bitwise
equivalence of the entire distributed vision encoder or caption accuracy.
Private receipts: `unique-dp-comparison.json`,
`unique-dp-zero-comparison.json`, `dp-zero-*-gate-rank*.json`.

The vision trial selected the overlay
`Qwen3-VL-8B-Instruct-w8a8-text-vision-dp-zero-20261013`, using the original
eight-row text kernels, image distribution and the gated zero-add norm.
The model remains TP2 on devices 0/1 with eight sequences, graph sizes 1/8
and a 1,024-token budget. Both original service units were restored, including
model readiness ordering and the logger restart dependency. Two complete
motion descriptions passed schema and frame-evidence validation before
live logging resumed. This is a serving compatibility check, not a labeled
accuracy evaluation. Private receipts include `selected-command.json`,
`natural-dp-zero`, `live-restored.json` and `final-service-state.json`.

The first 152 seconds of the restored logger produced 54 valid descriptions
across FrontDoor, back-door and Driveway, with mean inference latency
4.01 seconds and maximum 7.26 seconds. There were no inference errors or
exhausted-budget drops in this short check. Capture PTS gaps remained
explicit: six for FrontDoor and eight for back-door. `open-bulb-cam` was
unreachable on its RTSP TCP port from both the Ascend host and the local
workstation (errno 113); the logger continues retrying it. This check covers
three reachable cameras, not sustained four- or eight-camera capacity.
Both services are active and enabled at boot. All 81 scoped host tests,
two subtests, manual pre-commit checks and service-unit verification passed.

### Remaining-cost experiments

The next October 13 trace used the deployed image-distribution/zero-add model
and only live logger traffic. Profiling was stopped after six seconds. It
still recorded custom INT8 matrices at 25–26% of summed device-task time,
text attention at 16–18%, large-row quantized matrices at 9–11%, and generic
FP16 matrices at about 12%. All-reduce was 4.5–8.6%. These task fractions
include overlap and profiler overhead and are not request-latency shares.
Receipts are private under `next-ops-20261013`.

`input_quant.cpp` quantizes contiguous FP16 decode inputs directly to the
existing matrix kernel's INT8 NZ layout. This avoids repeating ND-to-NZ
packing in every matrix core. It preserves CANN's FP16 multiplication,
integer rounding and FP32 per-token scales. Each row owns an aligned scale
slot; consumers use stride eight. The original NZ FFN down-input path was
already fused and is retained.

The optional build is separate from the frozen matrix build. Overlay setup
freezes both its path and manifest, excludes inherited gate receipts and
rejects a matrix build with a different row contract. All binary hashes are
checked before loading. The input kernel supports up to eight rows and the
three validated hidden widths. Larger prefill inputs retain the original
model path. Logical contiguity does not prove ND device storage: non-base
formats are explicitly converted to ND for reference quantization and then
packed to NZ. The 310P runtime's reference quantizer rejects internal NZ
storage directly; the hardware test covers this fallback.

Both chips passed 75 exact code/scale checks, 15 non-ND fallback checks and
20 complete checkpoint-projection cases with five magnitudes and three
changed-input graph replays. The hardware nightly test passed on each chip.
The model worker also passed all 648 loaded text checks bit for bit per
rank. Initial isolated projection reductions ranged from about 0.5% to
10.3%. These kernel timings do not imply the same serving improvement.
The first matched serving run improved the eight-request batch by only
0.35%; the four-request control had different prefill timing despite this
being a decode-only change, so that apparent larger improvement needs a
fresh control before attribution.

The fresh control averaged 3.196, 7.038 and 11.568 seconds at concurrency
one, four and eight, compared with 3.195, 7.035 and 11.590 seconds for the
input-kernel candidate. It reproduced the apparent four-request gain and
slightly beat the candidate at eight. The input kernel therefore remains
staged despite its strict operator parity and isolated speedups. The live
selection retains the existing matrix/input path. Receipt:
`control-repeat-comparison.json`.

```bash
python -m tools.rtsp_scene.build_input_quant --output /private/input-build
python -m tools.rtsp_scene.dense_setup \
  --model /private/current-overlay --build /private/current-matrix-build \
  --output /private/input-overlay --mode all --vision-zero-add-norm \
  --input-quant-build /private/input-build
```

`--local-frame-root` in the model launcher and `--frame-transport file
--frame-root` in the runner/logger opt into upstream local-file image
loading. The logger owns only the `scene-live` child under the shared root.
Files remain present until inference completes. The client checks the
loopback endpoint, canonical path boundary, header, dimensions and payload
length; symlink escapes and malformed frames are rejected. Default base64
transport remains available for remote endpoints. This removes client pixel
reads/base64 encoding and large image payloads from JSON; server PPM loading,
image preparation and host-to-NPU transfer remain. It is not direct device
tensor transport or a zero-copy pipeline.

`file_transport_bench.py` alternates transport order on the same captures
and prompts, with two excluded warmups and four rounds per cohort. All 20
request pairs matched frame hashes, prompt hashes and token usage. At
concurrency one, mean batch time fell from 3.207 to 3.128 seconds (2.5%) and
client CPU time from 61.5 to 4.6 ms (92.5%). At concurrency four, mean batch
time fell from 7.110 to 6.894 seconds (3.0%) and total client CPU time per
batch from 193.0 to 19.2 ms (90.0%). Requests averaged 8–9.8 MB with base64
and about 3.4 KB with file references. Server process-group CPU time fell
only 1.5–2.1%; it includes overlapped work and busy waits and does not measure
the CPU fraction of request latency. Private receipt: `file-transport`.

The launcher adds opt-in `--async-scheduling` and `--kv-block-size` settings
without new environment variables. The default scheduling flag and cache
geometry are unchanged. Async scheduling passed the matched structured
requests and reduced batch time by 4.7%, 2.3% and 1.4% at concurrency one,
four and eight, respectively. A graph-size 1/4 trial with the same async
scheduler improved the four-request batch by only 1.0% but slowed the
eight-request batch by 12.4%; graph sizes 1/8 are retained. Receipts:
`async-comparison.json`, `graph4-comparison.json`.

A 64-token KV-block trial retained async scheduling and graph sizes 1/8.
Matched batch means were 3.092, 6.896 and 11.458 seconds for concurrency
one, four and eight: 1.6%, 0.3% and 0.2% slower than the 128-token async
control. This run gives no reason to change cache geometry. The selected
configuration retains 128-token blocks. Receipt: `block64-comparison.json`.

Native fused-prefill probes were kept separate from deployed changes.
`npu_quant_matmul_dequant` accepts contiguous output-major weights but failed
exact output parity at the observed prefill row counts. Incorrect-layout
attempts are retained as diagnostic failures and support no timing claims.
`npu_dequant_swiglu_quant` explicitly reports no operator configuration for
Ascend 310P in this CANN installation. Native INT8 matmul/all-reduce probes
were rejected by the current API's incompatible scale requirements:
FP32 dequant scales are rejected, while encoded INT64 dequant scales conflict
with FP32 per-token scales; encoding both scales is also rejected. This
does not establish that every fused-collective implementation is impossible.
The working custom matrix and HCCL paths remain available. Private receipts:
`native-prefill*.json`, `comm-*-rank*.json`, and their probe scripts/logs.

The final before/after transport comparison used the same original model
overlay in both cases. It compared synchronous scheduling/base64 with
asynchronous scheduling/local-file references, using two rounds per cohort
and matching all ten requests by frame hashes, prompts and token usage.
Single-request batch mean fell from 3.207 to 3.022 seconds (5.8%); four-request
batch mean fell from 7.022 to 6.773 seconds (3.5%). Client CPU per batch fell
91.3% and 89.3%, respectively. This directly measures the combined selected
changes; gains from earlier experiments are not added together. Receipt:
`combined-comparison.json`.

The live selection retains the original image-distribution/zero-add overlay
and eight-row matrix kernels, adds async scheduling and local-file frames,
and keeps graph sizes 1/8, eight sequences, the 1,024-token budget and
128-token KV blocks. The input-quantization overlay remains staged. Four
complete file-transport scene reports passed schema and frame-evidence
validation before the logger resumed; completions ranged from 34 to 112
tokens. This is a compatibility check, not a labeled accuracy evaluation.
Both original systemd units were restored, including readiness and restart
dependencies. Scene records now include transport mode, preparation wall
time and request bytes so the live path can be checked. Private receipts:
`selected-command.json`, `selected-natural`, `live-restored.json` and
`final-service-state.json`.

The restored logger's first 8 minutes 46 seconds produced 212 scenes across
all four cameras, with mean inference time 4.336 seconds and maximum 6.845
seconds. There were no inference errors or inference-budget drops in that
interval. FrontDoor produced only 17 scenes, with 109 sample timestamp gaps
and one capture error; the other cameras produced 65 scenes each. This
short observation does not prove four gap-free eight-second streams or
eight-camera capacity. All observed scene records used local-file transport.

Validation: 98 scoped host tests and two subtests passed. The input-kernel
hardware nightly test passed separately on logical devices zero and one.
The selected model passed 648 loaded text checks, 240 vision checks and 54
first-capture activation checks per rank. Manual pre-commit checks passed
on the changed source, tests and this README. Both services are active and
enabled, with no automatic restarts since restoration. Private receipts
are mirrored under `rtsp-scene-20261012/next-ops-20261013` beside this checkout.

### Prefill and resident image experiments

The following October 13 experiments are saved privately under
`pipeline-wave-20261013`. The serving control retains the selected model,
async scheduling, local-file transport, eight sequences, graph sizes 1/8
and the 1,024-token prefill budget. Diagnostic serving runs use the same
26 request bodies and fixed 64-token completions; truncated responses do
not establish scene accuracy or sustained camera capacity.

Increasing the budget to 2,048 or 4,096 tokens reduced queue time but gave
no useful four/eight-request batch improvement. A full-prefill attention
trial required an explicit 128-token KV block: the runtime otherwise chose
16, incompatible with this backend's 64/128 kernel blocks. With that fixed,
the full-prefill trial completed. Against a fresh 1,024-token control,
four/eight-request batches were about 0.3/0.4% slower. The single-request
trial was about 3.2% faster, which does not justify changing the shared
camera service. `--disable-chunked-prefill` remains an explicit experiment
requiring a context-sized token budget and supported KV block size.

Submission delays of 0.25 and 0.5 seconds per request were also tested with
the same bodies and token budgets. Batch timing includes those delays.
The 0.25-second trial improved four/eight-request batches by only 0.8/0.9%;
0.5 seconds improved four requests by 1.8% but gave no eight-request gain.
The latter nearly eliminated server queue time by moving waiting before
submission, without increasing throughput. This does not justify delaying
live windows. The launcher/logger selection keeps immediate submission.
Receipts: `stagger025-comparison.json`, `stagger050-comparison.json` and
`stagger-latency.json`, which includes completion time plus submission delay.

`prefill_i8.cpp` is a standalone 64-row, 128-column INT8 Cube prototype.
It loads bounded K tiles rather than allocating a full prefill activation
in each core, handles partial M tiles, and writes ND output directly.
Correcting the L1 NZ to L0A row-block ordering produced exact checkpoint
outputs in all 32 projection/row cases, each tested at five magnitudes.
At the observed large-row shapes it was roughly 2.8–4.5 times slower than
the native path, so it is not installed in serving. Prefill remains eager;
its reference ND-to-NZ Identity operation cannot be captured in a graph
in this runtime. Receipt: `prefill-v2-eager-rank0.json`.

The native `AddRmsNormDynamicQuant` operation has no binary in this pinned
CANN installation. Two custom residual/RMSNorm/NZ-quantization variants
preserved residual sums but failed the nonzero exact normalization/code
gates. They remain staged; zero-input passes are not evidence for a usable
fusion. Receipts: `fused-norm-rank1.json`, `custom-norm-rank0.json` and
`custom-norm-v2-rank0.json`.

The image path made concrete progress. `dvpp_tensor.cpp` reuses the existing
decoder inside an initialized Torch process. It borrows the worker's device
context, passes decoded RGB as a non-owning NPU tensor view, and synchronizes
before releasing each DVPP frame. The probe retains owned storage by a
device-to-device clone. Borrowed views must never escape the callback.
This path writes no images and performs no RGB/JPEG payload copies to the
host during capture. Validation deliberately copies retained frames to the
CPU afterward to construct the reference. Host demux, compressed-packet
upload, driver work and the device-to-device clone remain.

The pinned image processor's temporal expansion creates a nine-dimensional
copy unsupported by this NPU runtime. `tensor_preprocess.py` preserves its
patch order with an eight-dimensional spatial permutation followed by a
five-dimensional temporal expansion. Its override applies only to a copied
processor instance. The ordinary device resize still changed pixels, so
that path was rejected for exact serving substitution.

`image_prepare.cpp` and `image_horizontal.cpp` instead reproduce the CPU
byte resizer's fixed-point rounding. Cached four-tap coefficients use 14
fractional bits; products/sums fit exact FP32 integer arithmetic. Horizontal
and vertical passes round and clip separately. A cached normalization table
maps every channel/byte value to the original processor's FP16 model input.
Gather offsets use explicit integer rounding. The first build supports raw
RGB inputs 320×180 and 320×240; the model's minimum-pixel rule maps them to
352×192 and 320×256, respectively. The version-two build also handles the
live logger's existing 640×360 and 640×480 inputs, which map to 640×352 and
576×416. Downsampling uses 15 fractional bits and up to five active taps in
six allocated slots. These geometries were checked against the pinned CPU
byte resizer; other geometries/settings are rejected. Legacy frozen builds
remain accepted for their original input shapes.

Both chips passed the hardware nightly gate: two geometries, one/eight
frames, and random/zero/maximum/ramp pixel patterns, with all FP16 model
inputs bitwise identical. Isolated eight-frame timings were:

| Input | Device | CPU preparation | NPU preparation | Time reduction |
| --- | --- | --- | --- | --- |
| 320×180 | 0 | 6.53 ms | 1.45 ms | 77.8% |
| 320×180 | 1 | 6.43 ms | 1.47 ms | 77.2% |
| 320×240 | 0 | 6.38 ms | 0.93 ms | 85.5% |
| 320×240 | 1 | 6.26 ms | 0.93 ms | 85.2% |

Live H.264 back-door frames on device one matched the CPU model inputs
exactly, with preparation falling from 8.27 to 0.94 ms. Live H.265 Driveway
frames on device zero also matched exactly, falling from 7.53 to 1.46 ms.
Receipts: `nightly-image-rank0`, `nightly-image-rank1`,
`dvpp-exact-rank0.json` and `dvpp-exact-rank1.json`.

The full-resolution extension passed both nightly tests on each chip, covering
both input widths, all four geometries and 64 exact pixel-pattern checks in
total. Eight-frame preparation at the live capture resolution measured:

| Input | Device | CPU preparation | NPU preparation | Time reduction |
| --- | --- | --- | --- | --- |
| 640×360 | 0 | 17.52 ms | 2.20 ms | 87.4% |
| 640×360 | 1 | 25.31 ms | 2.21 ms | 91.3% |
| 640×480 | 0 | 31.71 ms | 5.92 ms | 81.3% |
| 640×480 | 1 | 29.68 ms | 5.89 ms | 80.1% |

Live back-door frames at 640×480 matched exactly and reduced preparation
from 33.37 to 5.91 ms (82.3%). Live Driveway frames at 640×360 matched
exactly and reduced preparation from 27.12 to 2.22 ms (91.8%). Both captures
wrote no images and copied no RGB/JPEG payload to the host during capture.
These gains preserve the live capture resolution. Receipts:
`nightly-image-full-rank0`, `nightly-image-full-rank1`,
`dvpp-exact640-rank0.json` and `dvpp-exact640-rank1.json`.

These are standalone device-pipeline results, not scene-serving speedups.
At this checkpoint, the live logger used local PPM references and the original
CPU image processor. Full resident integration needs worker-owned decoder/system lifetime,
bounded frame ownership, scheduler routing to the owning chip, and a metadata
contract that bypasses frontend pixel processing/serialization. Exporting
borrowed device pointers across processes is not a valid transport. Tables
and decoder buffers must be charged before KV-cache planning. No new upstream
model-runner behavior or shared OPP changes were deployed in this experiment.

To reproduce on an isolated chip with the pinned environment:

```bash
python -m tools.rtsp_scene.image_prepare \
  --compiler /private/matrix-build/compile-dense --output /private/image-build
python -m tools.rtsp_scene.image_prepare_bench \
  --build /private/image-build --matrix-build /private/matrix-build \
  --model /private/checkpoint --device 0 --input-width 640 --output /private/prepare.json
python -m tools.rtsp_scene.dvpp_tensor_probe --compile --build /private/tensor-build
python -m tools.rtsp_scene.dvpp_tensor_probe \
  --build /private/tensor-build --config /private/cameras.json --camera Driveway \
  --model /private/checkpoint --device 0 --max-side 640 --image-build /private/image-build \
  --matrix-build /private/matrix-build --output /private/live.json
```

`tests/e2e/nightly/rtsp_scene/test_image_prepare.py` requires exact model
inputs and at least a 10% isolated preparation improvement in every cohort.
It passed separately on both logical devices. The scoped host suite passed
123 tests plus two subtests; manual pre-commit checks passed for all changed
source and test files.

After the final restoration, both original systemd units were active and
enabled with zero automatic restarts. The first 2 minutes 9 seconds produced
48 scene records, all using file transport, with no inference errors or
inference-budget drops. FrontDoor produced three records and had one capture
error/reconnect plus timestamp gaps; the other three cameras produced 15
records each. This recovery check is not a sustained throughput measurement.
At this checkpoint, the live service retained the earlier selected configuration
and the exact device image pipeline was staged for worker integration. Receipt:
`pipeline-wave-20261013/final-service-state.json`.

## Worker image preparation

The opt-in `SceneDeviceQwen3VLForConditionalGeneration` architecture inherits
the existing Qwen3-VL model and registers its own processor. Ordinary Qwen
architectures retain their original behavior. The new processor packs CPU
RGB byte frames with CPU-only geometry and token-grid metadata. It skips CPU
resize, normalization and patch construction. `SceneDenseWorker` prepares the
FP16 patches on each NPU before calling the inherited vision/DeepStack path.
It uses the existing exact image kernels and TP2 vision data parallelism.

`scene_rgb` is flat uint8 HWC data in image order; `scene_geometry` holds
height/width for each frame and stays on CPU. `image_grid_thw` stays on CPU
and describes the resized model input, rather than the capture dimensions.
Flat field sizes use raw byte counts, independently of image-token counts.
The worker validates dimensions, grids, dtype and byte length, then groups
compatible shapes in batches of at most eight and restores the original order.
Only the four frozen geometries, patch/merge/temporal sizes 16/2/2, and pixel
bounds 65,536–262,144 use byte transport. Other request settings or image
dimensions retain the stock float processor. The selected runtime disables
encoder graphs and retains the existing native text decode graphs.

All static preparation tables are allocated and charged before KV-cache
planning. The helper attaches to `model_runner.get_model()`; attaching it to
the ACL graph wrapper passes startup but fails the first image request, since
the wrapper forwards attribute reads rather than writes. Worker processor
settings come from `multimodal_config.mm_processor_kwargs`, rather than the
checkpoint defaults. The first real serialized byte batch on each worker
checks exact FP16 pixels against the CPU processor and writes
`image-transport-gate-rank<N>.json`. This one-shot gate copies pixels to CPU;
subsequent supported requests perform no CPU model image preparation.

The hardware worker benchmark passed on both chips, including mixed shape
ordering and a nine-frame batch crossing the kernel limit. Eight-frame times:

| Input | Device | CPU reference including FP16 conversion | Worker NPU preparation |
| --- | --- | --- | --- |
| 640×360 | 0 | 25.61 ms | 2.44 ms |
| 640×360 | 1 | 26.25 ms | 2.44 ms |
| 640×480 | 0 | 31.00 ms | 6.21 ms |
| 640×480 | 1 | 31.24 ms | 6.18 ms |

The separate frontend benchmark, with eight CPU byte images, measured
34.72 → 1.65 ms for 640×360 and 43.45 → 1.92 ms for 640×480. It compares
the full stock CPU processor producing FP32 against byte packing; it excludes
file loading and worker inference. Transport sizes fell from 43.25 to 5.53 MB
and 46.01 to 7.37 MB, respectively (87.2% and 84.0%). These are preparation
and transport measurements, not complete scene latency reductions.

The real TP2 server passed concurrent byte-image, stock-image fallback and
text requests, with both worker pixel gates exact. A natural eight-frame
caption completed in 4.78 seconds with 122 output tokens and passed the scene
JSON schema. A scoped host run passed 136 tests plus two subtests. The worker
nightly benchmark checks exact pixels, smaller transport and at least a 10%
isolated preparation improvement; the serving regression specifically
exercises the graph wrapper, both field schemas and ordinary text inference.

Reproduce with the pinned runtime, a version-two frozen image build, and an
isolated TP2 server:

```bash
python -m tools.rtsp_scene.dense_setup \
  --model /private/checkpoint --build /private/matrix-build \
  --output /private/device-overlay --mode all --vision-zero-add-norm \
  --device-image-build /private/image-build
```

The serving processes must discover a private general-plugin entry point
named `rtsp_scene_device_image` pointing to
`tools.rtsp_scene.device_image_plugin:register`. The selected experiment
uses `/private/plugin/rtsp_scene_device_image-0.0.1.dist-info/METADATA`:

```text
Metadata-Version: 2.1
Name: rtsp-scene-device-image
Version: 0.0.1
```

And `entry_points.txt` in the same directory:

```ini
[vllm.general_plugins]
rtsp_scene_device_image = tools.rtsp_scene.device_image_plugin:register
```

Add `/private/plugin` to the existing runtime import path, then launch the
overlay with the selected `tools.rtsp_scene.launch` arguments. The plugin
registers the additional architecture without patching the stock model.

```bash
python -m pytest -q --confcutdir=tests/e2e/nightly/rtsp_scene \
  tests/e2e/nightly/rtsp_scene/test_worker_image_prepare.py \
  --scene-dense-model /private/device-overlay \
  --scene-dense-build /private/matrix-build \
  --scene-image-prepare-build /private/image-build --scene-dense-rank 0
python -m pytest -q --confcutdir=tests/e2e/nightly/rtsp_scene \
  tests/e2e/nightly/rtsp_scene/test_worker_image_serving.py \
  --scene-dense-model /private/device-overlay \
  --scene-image-serving-endpoint http://127.0.0.1:8003/v1/chat/completions \
  --scene-image-serving-root /private/shared-media
```

This bridge still reads PPM frames on the CPU and transfers RGB bytes to the
workers. JPEG encoding/decoding remains absent. Fully resident DVPP decode
still needs worker-owned system lifetime, bounded buffers and scheduler-aware
frame ownership. Borrowed device pointers cannot be sent through frontend
IPC. Receipts are under `worker-image-20261013`; two early timing runs are
explicitly invalid because the model's systemd dependency restarted live
camera logging during the benchmarks. The isolated comparison uses a temporary
logger condition and verifies the logger is inactive before and after each
benchmark. Restore that condition and any temporary mask before resuming
continuous logging.

The isolated 26-request fixed-64-token comparison (plus two warmups) used
identical request body hashes and token budgets. Results versus a fresh late
control were:

| Concurrent requests | Stock processor | Worker preparation | Latency reduction |
| --- | --- | --- | --- |
| 1 | 3.005 s | 3.100 s | −3.2% |
| 4 | 6.801 s | 6.780 s | 0.3% |
| 8 | 11.399 s | 11.441 s | −0.4% |

These small timing differences do not establish a scene throughput gain.
The API process consumed 4.53 CPU seconds for the stock path versus 1.64
seconds for the byte path across 28 requests including warmups: 161.8 versus
58.6 ms per request, a 63.8% reduction. CPU time comes from the API process's
`/proc/<pid>/stat`, excludes worker CPU time, and includes frontend file I/O,
processing, serialization and HTTP work. Worker launch/control CPU work
remains. Receipts: `baseline-fixed-control`, `candidate-fixed-isolated`, and
the corresponding `*-api-cpu.json` files. Fixed-token truncated outputs are
diagnostic workloads, rather than accepted scene reports. This experiment
does not increase the validated stream count.

The worker preparation overlay is selected in `scene-model-start.sh`. After
the final clean restart, the concurrent serving regression passed again,
both worker pixel gates were exact, and the original model/logger unit
definitions were restored without temporary conditions or masks. Both
services were active, enabled and had zero automatic restarts. The first
127 seconds produced 43 scene records across all four cameras, with no
inference errors or inference-budget drops. Median/max inference times were
4.30/5.58 seconds. FrontDoor reconnected after one capture error and retained
timestamp gaps; back-door also had two reported sample gaps. This short
recovery check does not establish sustained capacity. Selected-state receipt:
`worker-image-20261013/final-service-state.json` (also copied to the artifact
root's `final-service-state.json`).

## Decode profiling and projection-specific tiles

The next isolated pass is `decode-next-20261013`. `decode_profile.py` sends
text-only token-ID prompts of length 2,244, matching the saved scene request
context length, with 16 diagnostic output tokens and concurrency 1/4/8.
It warms each cohort, starts/stops the native profiler, and parses both worker
traces in separate non-daemon processes. Daemon workers cannot run the
profiler parsing process pool. The pinned parsing API is
`torch_npu.profiler.profiler.analyse`; the top-level package does not export
that function. No live logger runs during profiling or serving benchmarks.

`decode_profile_summary.py` isolates the first custom matrix after the final
native quantized prefill task. It checks complete groups of four projections
and one NZ activation across all 36 layers, and preserves sub-microsecond
precision by subtracting decimal timestamps before converting to floats.
This classifier requires the selected text-only execution path. It excludes
interleaved decode before the final prefill, the preceding normalization, and
communication metadata absent from the native graph trace. Fractions describe
summed recorded device tasks, rather than CPU time or request wall latency.

Both chips recorded 15 complete decode forwards per cohort. Matrix task
fractions were 72.7–73.0% at concurrency one and 58.2–59.3% at concurrency
four/eight. Attention increased from 5.1% to 21.2–21.5%; the FP16 vocabulary
head represented 12.1% and 9.6–9.8%, respectively. The concurrency-eight tail
contains active head batches seven through one after excluding prefill,
rather than a steady batch of eight. This makes attention a higher-priority
multi-stream target than the single-request trace suggests. Receipts:
`profiles-v3` and `decode-summary.json`.

An opt-in `--active-epilogue` matrix build converts and scales only the logical
NZ rows rather than all 16 physical rows. Both chips passed five input
magnitudes and three changed-input graph cases for QKV, output and whole FFN
at rows 1/2/4/8. The hardware nightly test passed on each chip. Timing gains
were small and inconsistent: whole FFN ranged from −0.5% to +2.0%. It remains
a staged candidate, with the flag disabled in selected kernels. Receipts:
`active-build-v1`, `nightly-rank0`, and `nightly-rank1`.

The 128-column/512-inner pipelined build balances the 3,072-column QKV shard
as 24 tiles over eight AI cores, versus 12 tiles with 256-column output tiles.
Paired real-weight graphs at rows 1/2/4/8 measured 9.3–11.0% faster QKV on
both chips, with bitwise identical reference values and changed-input replay
outputs. The same build slowed output projection and whole FFN, so it must
be selected per projection. These repeated layer-zero graph measurements do
not establish complete serving gains or account for full-model weight-cache
behavior. Receipts: `tile128-rank0.json` and `tile128-rank1.json`.

`dense_setup --qkv-build /private/qkv-build --mode all` freezes the optional
QKV-only build alongside the primary kernel manifest. The two builds must
have distinct native namespaces and identical decode row limits. At worker
load, every QKV projection uses its selected backend; output projections,
FFN, vision preparation, prefill fallback and graph sizes retain their
existing paths. Resident additions are charged before cache planning.
Loaded gates check all 36 QKV layers at rows one/eight and three magnitudes,
and the serving regression requires all 216 checks per chip to be bitwise
identical when a QKV override is configured. The private serving overlay
passed these gates and concurrent byte-image, fallback-image and text
requests. Smaller operator time alone does not justify a deployment claim.

The QKV-only serving candidate was compared using 26 matched file-image
requests, 64 output tokens each, and two excluded warmups. The initial control
was 2.989/6.781/11.405 seconds at concurrency 1/4/8. The candidate was
3.016/6.801/11.415 seconds; a separate isolated warm repeat measured
3.026/6.807/11.425 seconds. A fresh late control measured
2.995/6.749/11.390 seconds. These results do not show an end-to-end gain.
The original launcher and primary 256-column/256-inner build are retained,
including the previously selected worker image preparation. The QKV-only
selector and 128-column/512-inner build remain staged. Inference weights,
output length and frame sampling were not reduced to claim a speedup.
One other warm run overlapped the concurrent serving regression and is
explicitly excluded in `invalid-timings.json`.

Reproduction commands in the pinned environment:

```bash
python -m tools.rtsp_scene.decode_profile   --profile-root /private/native-profiles --output /private/decode-profiles
python -m tools.rtsp_scene.decode_profile_summary   --profiles /private/decode-profiles --output /private/decode-summary.json
python -m tools.rtsp_scene.build_dense   --output /private/qkv-build --columns 128 --inner-tile 512 --pipeline   --namespace rtsp_dense_qkv128
python -m tools.rtsp_scene.kernel_compare   --model /private/checkpoint --control /private/primary-build   --candidate /private/qkv-build --output /private/qkv-rank0.json   --rank 0 --device 0 --tokens 1 2 4 8 --iterations 100 --repeats 8
```

The kernel comparison now accepts an explicit device independently of the
TP checkpoint shard, so chip-one timing can use `--rank 1 --device 1`.
Run serving comparisons with camera logging paused and all profile recording
stopped. The scoped host suite passed 148 tests plus two subtests, both
active-epilogue hardware nightly gates passed, and the candidate serving gate
passed. Manual pre-commit checks passed on changed source, tests and this
runbook. The native vocabulary head and attention path remain unchanged.
The next attention investigation should use representative multi-stream
batches and KV lengths, where its recorded task fraction is about 21%.

After restoration, both services were active and enabled with zero automatic
restarts, no temporary unit conditions remained, and the original launcher
matched its backup exactly. The first 182 seconds produced 65 scene records
across all four cameras, with no inference errors or inference-budget drops.
Median/max inference times were 3.69/5.52 seconds. FrontDoor retained one
capture error and sample gaps; back-door had one reported sample gap. This
is a short recovery check, not a sustained capacity result. Current selected
state: `decode-next-20261013/final-service-state.json`, also copied to the
artifact root. Prior worker-image receipts retain their historical snapshot.
