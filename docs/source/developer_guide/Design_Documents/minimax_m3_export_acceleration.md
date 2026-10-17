# MiniMax-M3 CPU export acceleration

The initial export used the canonical Torch float64 MSE implementation, one
tensor at a time with four CPU threads. The recent rate was about four seconds
per expert matrix, projecting roughly 26 more hours. Increasing Torch threads
alone gave limited improvement on the Ryzen 9 7950X host.

An optional `--mse-backend numba` compiles the same 48-candidate block search.
It keeps the float64 candidate grid, scale floor, signed W4 code range,
round-to-even, FP16 scale storage, and code packing. Fast-math is disabled.
The block loop avoids materializing a series of full-chunk temporary arrays
for each candidate. Its CPU thread count follows the existing `--threads`
setting. Numba was already installed; no package upgrade was needed.

Different float64 reduction orders can change the winner of nearly tied
candidates. The compiled search detects these cases using a conservative
relative error margin and recomputes those blocks with the canonical Torch
search. Code generation and packing still use the existing Torch path.

## Validation and measured improvement

- All 68 focused offline tests passed, including 21 new tests covering exact
  codes and float64 scales, FP16/BF16/FP32/FP64 inputs, zero/tiny/outlier blocks,
  candidate ties, invalid input, identical exported file hashes, and resuming
  across backends without rewriting completed tensors.
- Packed-code and FP16-scale SHA-256 values matched the previous Torch probe
  for all nine complete real matrices: expert 0, layers 3/31/59, w1/w2/w3.
- A complete real `[3072,6144]` matrix export took 6.267 seconds with Torch
  and 1.103 seconds with Numba, including its first-call overhead. The output
  safetensors SHA-256 was identical: approximately **5.7x faster** in that test.
  The original background converter was running during both measurements.
- After resume verification, the first 37 newly converted tensors averaged
  0.711 seconds each, projecting about 4.5 hours remaining at that rate. This
  initial live sample is recorded in `live-rate.json`; it is an estimate,
  not a completion guarantee.
- These are CPU conversion measurements, not NPU inference performance.
  Disk reads, resume verification, and other host workloads still affect the
  full-export completion time.

Receipts are in `artifacts/minimax-m3-export-speed/`: `real-parity.json`,
`export-benchmark.json`, and `unit-tests.txt`. Reproduce the real matrix gate:

```bash
python -m tools.minimax_m3.validate_mse_cpu \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  artifacts/minimax-m3-offline-20261016/w4-mse-probe.json \
  --output /tmp/minimax-m3-compiled-mse-parity.json
```

## Resumed full export

The original exporter was stopped and superseded after 499 completed tensor
receipts. Those files were retained. The replacement verifies their hashes
before continuing with the compiled search. An incomplete temporary tensor is
rewritten automatically. The backend is recorded in new tensor receipts and
the final manifest, but it does not change the quantization recipe or resume
identity. Existing receipts without a backend field represent Torch.

```bash
python -m tools.minimax_m3.export_w4 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3 \
  /run/media/matteius/20TB-drive/models/MiniMax-M3-W4 \
  --method mse --mse-backend numba --threads 4 --chunk-rows 256
```

The supervised replacement log is
`artifacts/minimax-m3-full-export-20261017T032703Z/export.log`; its `status.json`
records the running PID or final exit state. CPU and I/O priorities remain
reduced. Qwen and the NPUs were not modified or used. The full export remains
in progress; the command above is for reproducing/resuming, not for starting a
second writer alongside the supervised job.
