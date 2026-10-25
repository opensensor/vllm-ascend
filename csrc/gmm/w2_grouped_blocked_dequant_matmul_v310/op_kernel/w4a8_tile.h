/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Signed-W4 storage / INT8 Cube compute for the sealed MiniMax runtime.
#ifndef MINIMAX_W4A8_TILE_H
#define MINIMAX_W4A8_TILE_H
#include "w2_blocked_dequant_matmul_v310.h"

namespace NsW2 {
constexpr uint32_t INT8_WEIGHT_BYTES = W2_L1_WIDE_TILE_N * W2_L1_WIDE_K;
constexpr uint32_t INT8_STAGE_K = 256;
constexpr uint32_t INT8_A_BYTES = W2_TILE_M * INT8_STAGE_K;
constexpr uint32_t INT8_FULL_A_OFFSET = INT8_WEIGHT_BYTES + 2 * INT8_A_BYTES;
constexpr uint32_t INT8_FUSED_MAX_ROWS = 4;
constexpr uint32_t INT8_FUSED_MAX_K = 6144;
static_assert(INT8_FULL_A_OFFSET + 16 * INT8_FUSED_MAX_K <= Arch::AtlasA2::L1_SIZE,
              "INT8 decode activations and tile stages must fit L1");

class W4A8Tile : public W2BlockedDequantMatmulV310Cube {
public:
    using Int8TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, int8_t, layout::RowMajor, int8_t, layout::nZ, int32_t, layout::RowMajor>;
    using Int8Mmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, int8_t, typename Int8TileCopy::LayoutTagL1A>;
    GlobalTensor<int8_t> xInt8Gm_;
    GlobalTensor<int8_t> debugGm_;
    GlobalTensor<uint32_t> offsetsGm_;
    bool debugDecode_ = false;
    bool fusedActivation_ = false;
    float rowScales_[INT8_FUSED_MAX_ROWS];
    GlobalTensor<float> activationScaleGm_;
    GlobalTensor<float> weightScaleGm_;

    __aicore__ inline void FillInt8Offsets()
    {
#ifdef MINIMAX_INT8_ORIGINAL_LAYOUT
        DataCopy(gatherOffsetsUB_, offsetsGm_, 4096);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
#endif
        Duplicate(masksUB_, static_cast<int16_t>(0x0f0f), 1024);
        Duplicate(masksUB_[1024], static_cast<int16_t>(0x0808), 1024);
        Duplicate(masksUB_[2048], static_cast<int16_t>(0xf0f0), 1024);
        SetFlag<HardEvent::S_V>(EVENT_ID6);
        WaitFlag<HardEvent::S_V>(EVENT_ID6);
    }

    __aicore__ inline void DecodePackedInt8(int64_t codeOffset)
    {
        DataCopy(cU8_, codesGm_[codeOffset], 2048);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto words = cU8_.ReinterpretCast<int16_t>();
        And(fieldI16UB_, words, masksUB_, 1024);
        And(andTmp_, words, masksUB_[1024], 1024);
        PipeBarrier<PIPE_V>();
        // Each byte's sign bit 0x08 becomes 0xf0. No carry crosses bytes.
        Muls(andTmp_, andTmp_, static_cast<int16_t>(30), 1024);
        PipeBarrier<PIPE_V>();
        Or(fieldI16UB_, fieldI16UB_, andTmp_, 1024);
        // A masked high nibble is already a signed INT8 multiple of sixteen.
        // This avoids ShiftRight, which dav_m200 does not implement.
        And(c16_, words, masksUB_[2048], 1024);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        PipeBarrier<PIPE_V>();
        Cast(signedHalfUB_, fieldI16UB_.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, 2048);
        Cast(signedHalfUB_[2048], c16_.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, 2048);
        PipeBarrier<PIPE_V>();
        Muls(signedHalfUB_[2048], signedHalfUB_[2048], static_cast<half>(1.0f / 16.0f), 2048);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeInt8Tile(uint32_t n0, uint32_t nActual, uint32_t kStart)
    {
        auto weightL1 = resource.l1Buf.template GetBufferByByte<int8_t>(0);
        // These buffers are unused by the parent's NZ decoder.
        auto quantized = nzTileUB_.ReinterpretCast<int8_t>();
        for (uint32_t rowBase = 0; rowBase < nActual; rowBase += 16) {
#ifdef GLM_W2_GROUPED_L1_SCALE_CACHE
            const int64_t scaleBufferBase = rowBase / W2_BLOCK_SIZE * kbCount_;
#else
            const int64_t scaleRow = (n0 + rowBase) / 32;
            DataCopy(scaleUB_, scaleGm_[scaleRow * kbCount_], (int32_t)kbCount_);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            const int64_t scaleBufferBase = 0;
#endif
            for (uint32_t k0 = kStart; k0 < kStart + W2_L1_WIDE_K; k0 += 256) {
                const int64_t codeOffset = ((int64_t)(n0 + rowBase) / 16 * (K_ / 256) + k0 / 256) * 2048;
                DecodePackedInt8(codeOffset);
                for (uint32_t block = 0; block < 8; ++block) {
                    const half ratio = static_cast<half>(scaleUB_.GetValue(scaleBufferBase + k0 / 32 + block));
                    Muls(signedHalfUB_[block * 512], signedHalfUB_[block * 512], ratio, 512);
                    PipeBarrier<PIPE_V>();
                }
                // dav_m200 implements CAST_ROUND, but CAST_RINT emits no conversion.
#ifdef MINIMAX_INT8_ORIGINAL_LAYOUT
                Gather(fractalRowsUB_, signedHalfUB_, gatherOffsetsUB_, (uint32_t)0, 4096);
                PipeBarrier<PIPE_V>();
                Cast(quantized, fractalRowsUB_, RoundMode::CAST_ROUND, 4096);
#else
                Cast(quantized, signedHalfUB_, RoundMode::CAST_ROUND, 4096);
#endif
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                if (debugDecode_) {
                    const int64_t dest = ((int64_t)(n0 + rowBase) / 16 * (K_ / 256) + k0 / 256) * 4096;
                    DataCopy(debugGm_[dest], quantized, 4096);
                }
                DataCopyParams descriptor;
                descriptor.blockCount = 8;
                descriptor.blockLen = 16;
                descriptor.srcStride = 0;
                descriptor.dstStride = W2_L1_WIDE_TILE_N - 16;
                const uint32_t dest = (k0 - kStart) / 32 * W2_L1_WIDE_TILE_N * 32 + rowBase * 32;
                DataCopy(weightL1[dest], quantized, descriptor);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void PrepareActivation()
    {
        // Scratch is below the parent decoder's UB region and released before
        // the output epilogue. Full quantized activations remain in L1.
        auto input = resource.ubBuf.template GetBufferByByte<half>(0);
        auto values = resource.ubBuf.template GetBufferByByte<float>(16384);
        auto absolute = resource.ubBuf.template GetBufferByByte<half>(49152);
        auto quantHalf = resource.ubBuf.template GetBufferByByte<half>(65536);
        auto quantized = resource.ubBuf.template GetBufferByByte<int8_t>(81920);
        auto pairs = resource.ubBuf.template GetBufferByByte<half>(90112);
        auto offsets = resource.ubBuf.template GetBufferByByte<uint32_t>(91136);
        auto maxima = resource.ubBuf.template GetBufferByByte<half>(92160);
        auto reduced = resource.ubBuf.template GetBufferByByte<half>(93440);
        const uint32_t repeats = K_ / 128;
        for (uint32_t i = 0; i < repeats; ++i) offsets.SetValue(i, i * 2 * sizeof(half));
        SetFlag<HardEvent::S_V>(EVENT_ID6);
        WaitFlag<HardEvent::S_V>(EVENT_ID6);
        auto fullA = resource.l1Buf.template GetBufferByByte<int8_t>(INT8_FULL_A_OFFSET);
        for (uint32_t row = 0; row < (uint32_t)T_; ++row) {
            DataCopy(input, xGm_[row * K_], (uint32_t)K_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
            Abs(absolute, input, (uint32_t)K_);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(pairs, absolute, 128, repeats, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            Gather(maxima, pairs, offsets, (uint32_t)0, repeats);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced, maxima, repeats, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            SetFlag<HardEvent::V_S>(EVENT_ID6);
            WaitFlag<HardEvent::V_S>(EVENT_ID6);
            const float maximum = static_cast<float>(reduced.GetValue(0));
            const float scale = maximum > 0 ? maximum / 127.0f : 1.0f;
            rowScales_[row] = scale;
            Cast(values, input, RoundMode::CAST_NONE, (uint32_t)K_);
            PipeBarrier<PIPE_V>();
            Muls(values, values, 1.0f / scale, (uint32_t)K_);
            PipeBarrier<PIPE_V>();
            Cast(quantHalf, values, RoundMode::CAST_NONE, (uint32_t)K_);
            PipeBarrier<PIPE_V>();
            Cast(quantized, quantHalf, RoundMode::CAST_ROUND, (uint32_t)K_);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
            DataCopyParams descriptor;
            descriptor.blockCount = K_ / 32;
            descriptor.blockLen = 1;
            descriptor.srcStride = 0;
            descriptor.dstStride = 15;
            DataCopy(fullA[row * 32], quantized, descriptor);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        }
        SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID6);
    }

    __aicore__ inline void StoreInt8Accumulator(LocalTensor<int32_t> accumulator, uint32_t n0, uint32_t nActual)
    {
        const uint32_t alignedRows = AlignUpU<uint32_t>((uint32_t)T_, 16);
        const uint32_t elements = alignedRows * nActual;
        auto raw = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        auto values = raw.ReinterpretCast<float>();
        auto converted = resource.ubBuf.template GetBufferByByte<half>(elements * sizeof(float));
        SetFlag<HardEvent::M_V>(EVENT_ID7);
        WaitFlag<HardEvent::M_V>(EVENT_ID7);
        DataCopyParams fromCube;
        fromCube.blockCount = nActual / 16;
        fromCube.blockLen = alignedRows / 16;
        fromCube.srcStride = 0;
        fromCube.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(raw, accumulator, fromCube, enhanced);
        PipeBarrier<PIPE_V>();
        Cast(values, raw, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        for (uint32_t nf = 0; nf < nActual / 16; ++nf) {
            const float weightScale = weightScaleGm_.GetValue((n0 + nf * 16) / 32);
            for (uint32_t row = 0; row < (uint32_t)T_; ++row) {
                const uint32_t offset = nf * alignedRows * 16 + row * 16;
                const float scale = weightScale * (fusedActivation_ ? rowScales_[row] : activationScaleGm_.GetValue(row));
                Muls(values[offset], values[offset], scale, 16);
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(converted, values, RoundMode::CAST_NONE, elements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        for (uint32_t nf = 0; nf < nActual / 16; ++nf) {
            DataCopyParams toGm;
            toGm.blockCount = (uint32_t)T_;
            toGm.blockLen = 1;
            toGm.srcStride = 0;
            toGm.dstStride = (N_ - 16) * sizeof(half) / 32;
            DataCopy(yGm_[n0 + nf * 16], converted[nf * alignedRows * 16], toGm);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        SetFlag<HardEvent::V_M>(EVENT_ID7);
        WaitFlag<HardEvent::V_M>(EVENT_ID7);
    }
    __aicore__ inline void ProcessInt8(bool reuseDecodeTables)
    {
        if (!reuseDecodeTables) {
            AllocBuffers();
            FillTables();
            FillInt8Offsets();
        }
        if (debugDecode_) {
            for (uint32_t n0 = tileLane_ * W2_L1_WIDE_TILE_N; n0 < N_; n0 += tileLanes_ * W2_L1_WIDE_TILE_N) {
#ifdef GLM_W2_GROUPED_L1_SCALE_CACHE
                DataCopy(scaleUB_, scaleGm_[static_cast<int64_t>(n0) / W2_BLOCK_SIZE * kbCount_],
                         static_cast<int32_t>(W2_SCALE_BUFFER_ROWS * kbCount_));
                SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
#endif
                for (uint32_t k0 = 0; k0 < K_; k0 += W2_L1_WIDE_K) {
                    DecodeInt8Tile(n0, W2_L1_WIDE_TILE_N, k0);
                }
            }
            return;
        }
        if (fusedActivation_) PrepareActivation();
        const uint32_t rows = T_ == 1 ? W2_FRACTAL_SIZE : (uint32_t)T_;
        const uint32_t alignedRows = AlignUpU<uint32_t>(rows, W2_FRACTAL_SIZE);
        auto aLayout = tla::MakeLayout<int8_t, layout::RowMajor>((uint32_t)T_, (uint32_t)K_);
        auto tensorA = tla::MakeTensor(xInt8Gm_, aLayout, Arch::PositionGM{});
        using CopyA = typename Int8TileCopy::template CopyGmToL1A<decltype(tensorA)>;
        CopyA copyA;
        typename Int8TileCopy::CopyL1ToL0A copyL0A;
        typename Int8TileCopy::CopyL1ToL0B copyL0B;
        Int8Mmad tileMmad;
        auto aL1Layout = tla::MakeLayout<int8_t, typename Int8TileCopy::LayoutTagL1A>(
            alignedRows, INT8_STAGE_K);
        auto bL1Layout = tla::MakeLayout<int8_t, typename Int8TileCopy::LayoutTagL1B>(
            W2_L1_WIDE_K, W2_L1_WIDE_TILE_N);
        auto aL0Layout = tla::MakeLayout<int8_t, typename Int8TileCopy::LayoutTagL0A>(
            rows, INT8_STAGE_K);
        auto bL0Layout = tla::MakeLayout<int8_t, typename Int8TileCopy::LayoutTagL0B>(
            INT8_STAGE_K, W2_L1_WIDE_TILE_N);
        auto tensorL1B = tla::MakeTensor(
            resource.l1Buf.template GetBufferByByte<int8_t>(0), bL1Layout, Arch::PositionL1{});
#ifndef GLM_W2_GROUPED_L1_WIDE_PIPELINED
        auto tensorL1A = tla::MakeTensor(
            resource.l1Buf.template GetBufferByByte<int8_t>(INT8_WEIGHT_BYTES),
            aL1Layout, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            resource.l0ABuf.template GetBufferByByte<int8_t>(0), aL0Layout, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            resource.l0BBuf.template GetBufferByByte<int8_t>(0), bL0Layout, Arch::PositionL0B{});
#endif
        auto accumulator = resource.l0CBuf.template GetBufferByByte<int32_t>(0);
        auto tensorL0C = tla::MakeTensor(
            accumulator, tla::MakeLayoutL0C(rows, W2_L1_WIDE_TILE_N), Arch::PositionL0C{});

        const uint32_t nBlocks = CeilDivU<uint32_t>((uint32_t)N_, W2_L1_WIDE_TILE_N);
        for (uint32_t nb = tileLane_; nb < nBlocks; nb += tileLanes_) {
            const uint32_t n0 = nb * W2_L1_WIDE_TILE_N;
            const uint32_t nActual = MinU<uint32_t>(W2_L1_WIDE_TILE_N, (uint32_t)N_ - n0);
#ifdef GLM_W2_GROUPED_L1_SCALE_CACHE
            // The four scale rows cover every K slice of this N tile. Keep
            // them in UB until the tile completes instead of reloading the
            // same rows and waiting for MTE2 at each 1024-K boundary.
            DataCopy(scaleUB_, scaleGm_[static_cast<int64_t>(n0) / W2_BLOCK_SIZE * kbCount_],
                     static_cast<int32_t>(nActual / W2_BLOCK_SIZE * kbCount_));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
#endif
#ifdef GLM_W2_GROUPED_L1_WIDE_PIPELINED
            // Keep these stage handoffs alive across K tiles. The next tile
            // can dequantize while Cube consumes the previous tile's L0 data.
            for (uint32_t stage = 0; stage < W2_L1_WIDE_A_STAGES; ++stage) {
                SetFlag<HardEvent::MTE1_MTE2>(stage);
                SetFlag<HardEvent::M_MTE1>(stage);
            }
#endif
            for (uint32_t kBlock = 0; kBlock < K_; kBlock += W2_L1_WIDE_K) {
                DecodeInt8Tile(n0, nActual, kBlock);
                SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID4);
                WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID4);
                for (uint32_t kLocal = 0; kLocal < W2_L1_WIDE_K; kLocal += INT8_STAGE_K) {
                    const uint32_t kGlobal = kBlock + kLocal;
                    auto tileA = GetTile(tensorA, tla::MakeCoord((uint32_t)0, kGlobal),
                                         tla::MakeShape((uint32_t)T_, INT8_STAGE_K));
#ifdef GLM_W2_GROUPED_L1_WIDE_PIPELINED
                    const uint32_t stage = (kLocal / INT8_STAGE_K) % W2_L1_WIDE_A_STAGES;
                    auto tensorL1A = tla::MakeTensor(
                        resource.l1Buf.template GetBufferByByte<int8_t>(
                            INT8_WEIGHT_BYTES + stage * INT8_A_BYTES),
                        aL1Layout, Arch::PositionL1{});
                    auto tensorL0A = tla::MakeTensor(
                        resource.l0ABuf.template GetBufferByByte<int8_t>(stage * INT8_A_BYTES),
                        aL0Layout, Arch::PositionL0A{});
                    auto tensorL0B = tla::MakeTensor(
                        resource.l0BBuf.template GetBufferByByte<int8_t>(
                            stage * W2_L1_WIDE_TILE_N * INT8_STAGE_K * sizeof(int8_t)),
                        bL0Layout, Arch::PositionL0B{});
                    WaitFlag<HardEvent::MTE1_MTE2>(stage);
#endif
                    // dav_m200's generic INT8 ND-to-NZ fallback uses an unsupported
                    // vector Adds<int8>. Copy aligned K32 strips directly into L1 zN.
                    for (uint32_t strip = 0; !fusedActivation_ && strip < INT8_STAGE_K / 32; ++strip) {
                        DataCopyParams descriptor;
                        descriptor.blockCount = (uint32_t)T_;
                        descriptor.blockLen = 1;
                        descriptor.srcStride = K_ / 32 - 1;
                        descriptor.dstStride = 0;
                        DataCopy(tensorL1A.data()[strip * alignedRows * 32],
                                 xInt8Gm_[kGlobal + strip * 32], descriptor);
                    }
#ifdef GLM_W2_GROUPED_L1_WIDE_PIPELINED
                    SetFlag<HardEvent::MTE2_MTE1>(stage);
                    WaitFlag<HardEvent::M_MTE1>(stage);
                    WaitFlag<HardEvent::MTE2_MTE1>(stage);
#else
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
#endif
                    auto tileL1A = GetTile(tensorL1A, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                                            tla::MakeShape(rows, INT8_STAGE_K));
                    auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(kLocal, (uint32_t)0),
                                            tla::MakeShape(INT8_STAGE_K, nActual));
                    if (fusedActivation_) {
                        auto fullTensorA = tla::MakeTensor(
                            resource.l1Buf.template GetBufferByByte<int8_t>(INT8_FULL_A_OFFSET),
                            tla::MakeLayout<int8_t, typename Int8TileCopy::LayoutTagL1A>(alignedRows, (uint32_t)K_),
                            Arch::PositionL1{});
                        auto fullTileA = GetTile(fullTensorA, tla::MakeCoord((uint32_t)0, kGlobal),
                                                tla::MakeShape(rows, INT8_STAGE_K));
                        copyL0A(tensorL0A, fullTileA);
                    } else {
                        copyL0A(tensorL0A, tileL1A);
                    }
                    copyL0B(tensorL0B, tileL1B);
#ifdef GLM_W2_GROUPED_L1_WIDE_PIPELINED
                    SetFlag<HardEvent::MTE1_MTE2>(stage);
#endif
                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                    const uint8_t unitFlag = kGlobal + INT8_STAGE_K == K_ ? 0b11 : 0b10;
                    tileMmad(tensorL0C, tensorL0A, tensorL0B, rows, nActual,
                             INT8_STAGE_K, kGlobal == 0, unitFlag);
#ifdef GLM_W2_GROUPED_L1_WIDE_PIPELINED
                    SetFlag<HardEvent::M_MTE1>(stage);
#else
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
#endif
                }
                // All B reads must finish before dequant overwrites L1. Cube
                // uses separate L0 buffers and does not need to drain here.
                SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID4);
                WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID4);
            }
#ifdef GLM_W2_GROUPED_L1_WIDE_PIPELINED
            for (uint32_t stage = 0; stage < W2_L1_WIDE_A_STAGES; ++stage) {
                WaitFlag<HardEvent::M_MTE1>(stage);
                WaitFlag<HardEvent::MTE1_MTE2>(stage);
            }
#endif
            StoreInt8Accumulator(accumulator, n0, nActual);
        }
    }

};
} // namespace NsW2
#endif
