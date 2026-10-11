# One Ascend card plus RTX: hardware qualification

October 11, 2026. These receipts describe the v10 overlap qualification at
`http://192.168.53.187:8001/v1`, model `glm-sidecard`. The later
[mHC optimization](../optimization/README.md) supersedes its API deployment.
[allocation.json](allocation.json) and [service-process.json](service-process.json)
identify the owned processes. GLM uses Ascend card 16409, PCIe 49:00.0,
physical devices 2 and 3, plus the RTX. Card 8217, devices 0 and 1, remains
available to the user's other workload. Its existing processes were not stopped.

The earlier TP4 sidecard API and CUDA workers were retired at the user's
request. The first smaller launch found a new Qwen service on devices 0/1;
GLM moved to the other physical pair. The final launcher accepts exactly
one pair and checks runtime visibility against that admission.

## Implementation and memory

CUDA owns experts 0–143 in all 42 main MoE layers. NPU rank 0 owns 144–215;
rank 1 owns 216–287. Attention, recurrent state and shared experts stay on
Ascend. Original integer codes and FP16 scales are preserved. CUDA banks use
lossless four-bit word planes. Native Cube conversion runs on CUDA; CPU
safetensors handles are reused per layer. Input packets are copied before
native work; host-only RPC runs on one session thread while NPU experts run
on the original caller stream.

[cuda-ready.json](cuda-ready.json) records **122.80 seconds** to prepare all
6,048 CUDA experts and **76,555,018,240 bytes** of admitted allocations
(71.30 GiB including projection buffers). Temporary Cube storage is freed.
[cuda-process.json](cuda-process.json) points to the v8 daemon whose banks
were reused for the v9 and v10 API deployments.
[cuda-compiled-source.cu](cuda-compiled-source.cu) retains the actual compiled
input; the library hash is pinned in the gate and service receipts.

The final API uses 3.5 GiB KV per NPU and advertises **260,352 context tokens**.
Kilo has 228,352 input / 32,000 output limits. The earlier 4 GiB serial
candidate advertised 300,672 tokens but left only 28 MiB free on the smaller
NPU after 8K and mixed-size requests. Its capacity setting was retired.
[npu-inventory.log](npu-inventory.log) records 571 MiB free on that chip after
the overlap checks and 8K benchmark. Both GLM chips report healthy.
[cuda-inventory.log](cuda-inventory.log) records the sole CUDA compute process.

## Correctness checks

- [direct-cube-gate.json](direct-cube-gate.json): nine CUDA conversion cases,
  W2/W3/W4 and multiple full tile geometries, all byte-equal to CPU packing.
- [packed-gate.json](packed-gate.json): 27 real-checkpoint cases, 144 CUDA
  experts in W4/W3/W2 layers, 128/512/640 tokens and 1/2/8 routes per token;
  all weighted outputs equal bitwise to the wide-address INT8 reference.
- [service-validation.json](service-validation.json): all seven operational
  API cases passed, and all 42 layers executed CUDA for actual requests.
  Three short calls and one bulk prefill per layer compared serialized and
  overlapped execution: **168 checks, zero absolute error and zero NRMSE**.
  Resident copies of CUDA experts on NPU are absent; these comparisons check
  scheduling parity, separately from the GPU code/weight gates.
- [context-validation.json](context-validation.json): 32,000-output-budget
  chat generated `4` and stopped normally; 260,351 input tokens plus one
  output token were admitted, then disconnected after headers; over-context
  output was rejected with HTTP 400. Full 260K prefill was not completed.

Greedy repeated completions still vary. The full service validation flag is
false; whole-model quality and long-context accuracy remain unqualified.

## Actual serving measurements

The 8K prompt is synthetic repeated Python-review text. Each case generated
four tokens and completed. Cache usage is reported explicitly.

| One-card profile | Input | Cached | First text | Uncached prompt rate |
| --- | --- | --- | --- | --- |
| Serial, first | 8,192 | 0 | 172.39 s | 47.52 tok/s |
| Serial, repeat | 8,192 | 8,064 | 3.29 s | 38.89 tok/s |
| Overlap, first | 8,192 | 0 | 143.23 s | 57.20 tok/s |
| Overlap, repeat | 8,192 | 8,064 | 2.73 s | 46.82 tok/s |

[serial-prefill-8192.json](serial-prefill-8192.json) and
[prefill-8192.json](prefill-8192.json) record the cases. The new profile's
cold prompt rate increased **20.4%** versus the serial one-card candidate;
cache allocation also changed from 4 to 3.5 GiB. Decode gaps were typically
0.36–0.40 seconds, approximately 2.6–2.8 tokens/s.

The identical 449-token uncached prompt with one output token fell from
**8.14 to 6.54 seconds**; all 42 layers used overlap, and shadow work was zero
for this steady measurement. See
[serial-steady-prefill-profile.json](serial-steady-prefill-profile.json) and
[steady-prefill-profile.json](steady-prefill-profile.json).

The retired three-card v6 profile measured 76.37 uncached tokens/s with 640
cached tokens in its 8K first case and 6.67 seconds for its repeat. Its bank
placement, TP width, cache size and cached-token count differ; these are
historical resource/performance points, not an isolated comparison:
[prior-three-card-prefill.json](prior-three-card-prefill.json).

## Repository checks

Focused CPU checks passed: 354 sidecard/pipeline tests and 33 grouped-bank
tests. Changed Python/Markdown files passed all applicable repository hooks;
the CUDA source passed explicit clang-format validation. The larger assembly
test cannot collect in this local environment because installed upstream
vLLM lacks `vllm.models.deepseek_v41`. Repository-wide formatting has existing
failures outside this change; the scoped checks cover the delivered files.
