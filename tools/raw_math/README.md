# Raw math experiments

Model-independent comparisons of resident GEMM, FP32 vector math, device copies,
and pageable/pinned host transfers. The hardware-qualified study and numerical
receipts are in [the October 11 report](../../artifacts/raw-math-20261011/README.md).

## CUDA

Build the standalone library with the installed CUDA toolkit. This host uses
SM120; set the architecture to match your GPU when adapting the experiment.
A CPU PyTorch installation supplies reproducible inputs and reference checks.

```bash
nvcc -O3 --fmad=false -std=c++17 -arch=sm_120 -shared -Xcompiler -fPIC \
  tools/raw_math/cuda_math.cu -lcublas -o /tmp/raw-math.so
python -m tools.raw_math.benchmark --backend cuda --library /tmp/raw-math.so \
  --suite matrix --precision fp16 fp32 int8 bf16 --output /tmp/cuda-matrix.json
python -m tools.raw_math.benchmark --backend cuda --library /tmp/raw-math.so \
  --suite vector --output /tmp/cuda-vector.json
python -m tools.raw_math.benchmark --backend cuda --library /tmp/raw-math.so \
  --suite copy --output /tmp/cuda-copy.json
```

The CUDA library and its dependencies must be on the normal library search path.
GEMM uses cuBLAS; add, exp, softmax, and RMSNorm use standalone CUDA kernels.
FP32 uses pedantic computation. FP16/BF16 request FP32 accumulation. INT8 uses
integer accumulation followed by an FP16 output conversion included in timing.

## Ascend

Use the installed torch_npu environment with the CANN environment sourced.
This experiment guards physical devices **2,3**, the two chips on the assigned
card. Logical device 0 uses physical 2; logical 1 uses physical 3. The guard
protects the separate workload on physical devices 0,1 on this host.

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m tools.raw_math.benchmark \
  --backend npu --suite matrix --precision fp16 fp32 int8 \
  --output /tmp/npu-matrix.json
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m tools.raw_math.benchmark \
  --backend npu --suite vector --output /tmp/npu-vector.json
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m tools.raw_math.benchmark \
  --backend npu --suite copy --output /tmp/npu-copy.json
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m tools.raw_math.card \
  --output /tmp/npu-card.json
```

Matrix cases measure ND and preformatted NZ weights independently. HF32 is
disabled; strict FP32 accuracy remains a separate check. Test unsupported
precisions in their own process:

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m tools.raw_math.benchmark \
  --backend npu --suite capability --output /tmp/npu-bf16.json
ASCEND_RT_VISIBLE_DEVICES=2,3 python -m tools.raw_math.benchmark \
  --backend npu --suite matrix --precision fp32 --reference-fp16-inputs \
  --output /tmp/npu-input-rounding.json
```

The rounding diagnostic explicitly changes the reference to FP16-rounded inputs
and records error against the original FP32 reference. It does not qualify
strict FP32. INT8 uses already quantized inputs and unit dequantization scales;
activation quantization and model weight reconstruction are outside this study.

## Timing and validation

Initialization, uploads, layout conversion, and validation readbacks are outside
resident arithmetic timing. Transfers include the requested copy. Single-call
and queued modes both store all device-event and synchronized wall samples.
Device events can include host submission gaps. CUDA queues through C++; Ascend
queues through Python/torch_npu. Ascend quantized GEMM and RMSNorm allocate their
returned output through the vendor API; CUDA reuses output buffers.

Matrix results are sampled against CPU FP64 dot products, row math checks all
columns of sampled rows, and vector math checks up to 1024 evenly spaced elements.
Arithmetic outputs also receive a full finite-value check. Copies check every
byte. Initial, changed-input, and final validation gates reject stale output.
Reports preserve unsupported or inaccurate cases without assigning them a speed.
`completed=true` means every case was attempted; inspect status and passed counts.

Both-chip throughput uses simultaneous independent jobs and a shared wall
completion interval, including coordination. It does not model a sharded GEMM
or a distributed model. No service, device clock, power limit, or ECC setting
is changed by these benchmark entry points.

## Export and tests

```bash
python -m tools.raw_math.report artifacts/raw-math-20261011/receipts --plots
pytest -q --confcutdir=tests/ut/raw_math tests/ut/raw_math
```

The exporter requires complete receipts with all passed numerical gates. Failed
cases remain visible in CSV, and FP32 rounding diagnostics carry a separate
column. Plot export additionally needs matplotlib.
