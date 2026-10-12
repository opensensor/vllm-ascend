# Measured comparisons

Queued device-event medians; three warmups, 15 samples. GEMM multiply and add each count as one operation. Ascend uses one chip with NZ weights unless stated.

| Resident GEMM | RTX ms | Ascend ms | RTX TFLOP/s or TOP/s | Ascend TFLOP/s or TOP/s | RTX speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 vector | 0.011349 | 0.183950 | 2.96 | 0.18 | 16.21× |
| FP16 batch128 | 0.019712 | 0.253314 | 217.89 | 16.96 | 12.85× |
| FP16 square4096 | 0.376992 | 2.873021 | 364.57 | 47.84 | 7.62× |
| FP16 square8192 | 2.682752 | 40.223202 | 409.84 | 27.34 | 14.99× |
| INT8 vector | 0.013290 | 0.078964 | 2.52 | 0.42 | 5.94× |
| INT8 batch128 | 0.018344 | 0.094486 | 234.13 | 45.46 | 5.15× |
| INT8 square4096 | 0.212416 | 1.801537 | 647.03 | 76.29 | 8.48× |
| INT8 square8192 | 1.465408 | 19.577240 | 750.31 | 56.16 | 13.36× |

| Two independent Ascend jobs | Both chips combined TFLOP/s or TOP/s | Wall ms for four jobs per chip |
| --- | ---: | ---: |
| FP16 square4096 | 84.54 | 13.006 |
| FP16 square8192 | 48.66 | 180.772 |
| INT8 square4096 | 153.46 | 7.165 |
| INT8 square8192 | 108.17 | 81.320 |

| FP32 vector operation | RTX µs | Ascend µs | RTX speed ratio |
| --- | ---: | ---: | ---: |
| add_67108864 | 523.776 | 4153.984 | 7.93× |
| exp_67108864 | 358.016 | 2777.578 | 7.76× |
| softmax_1x4096 | 5.265 | 12.415 | 2.36× |
| softmax_2048x4096 | 21.984 | 442.825 | 20.14× |
| rmsnorm_1x4096 | 3.760 | 20.636 | 5.49× |
| rmsnorm_2048x4096 | 16.048 | 387.988 | 24.18× |

| Copy | Size | RTX wall µs | Ascend wall µs | RTX wall payload GB/s | Ascend wall payload GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| copy D2D | 8 KiB | 10.830 | 121.801 | 0.756 | 0.067 |
| copy D2D | 1 GiB | 1474.194 | 11063.871 | 728.359 | 97.049 |
| h2d pinned | 8 KiB | 9.669 | 393.368 | 0.847 | 0.021 |
| h2d pinned | 1 GiB | 38033.872 | 80065.098 | 28.231 | 13.411 |
| d2h pinned | 8 KiB | 9.338 | 383.850 | 0.877 | 0.021 |
| d2h pinned | 1 GiB | 37605.338 | 81075.509 | 28.553 | 13.244 |
