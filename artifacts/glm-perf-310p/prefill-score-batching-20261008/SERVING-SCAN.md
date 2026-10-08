# GLM serving 扫描摘要

英文逐项说明见 [SERVING-SCAN.en.md](SERVING-SCAN.en.md)。四 rank live status
仍为合格的 kda_reuse_v2、native expert fallback 为零，selected state rows 生效。

- `dsa.py` 的 `.item()` / matmuls 属于 reference；实际使用 paged MLA / kpool。
- `kda.py` 是 CPU parity core；NPU 使用 `kda_310.py`。
- packing、CPU copies、stacking 与 indexer weight casts 主要是 load/preparation。
- KDA 的 layout copies、beta/gate 与 output RMSNorm gating 是实际候选。
- L2norm 已返回 FP16；A_log / dt_bias 已为 FP32。相同 dtype 转换通常无操作。
- prefill carry 已使用 v925 小规模 gather/scatter，但精度边界仍存在。
- `moe.py` mHC helpers 不是 bound implementation；实际 patch 已使用合格 norm。
- `.t()` / split 通常为 view；reshape / contiguous 是否复制取决于 stride。

最新 640-token profile 的 InplaceCopy/Cast family 合计 21.89 ms、Matmul/Cast
34.88 ms，不能逐行归因或当 wall critical path。更大成本仍为 experts、sparse
attention 与 KDA scores；下一候选批量化完整 KDA prefill kernel 内的 vector work。
