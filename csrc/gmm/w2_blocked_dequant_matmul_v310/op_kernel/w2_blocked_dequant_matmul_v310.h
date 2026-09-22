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

/*!
 * \file w2_blocked_dequant_matmul_v310.h
 * \brief Cube (AIC) 310P kernel accepting packed signed W2/W4 codes.
 *
 *   out[t, n] = sum_k x[t, k] * (code[n, k] * block_scale[n/32, k/32])
 *   codes: uint8 [N, K/codes_per_byte], little-endian by field. Packed width
 *   selects W2 (4 codes/byte) or W4 (2 codes/byte); both use two's-complement
 *   sign extension and match tools/deepseek_w2/w2_format.py.
 *
 * Interleave strategy (perf): the on-chip unpack naturally produces the weight
 * in FIELD-MAJOR K order (p = j*packed_k + i; k = codes_per_byte*i + j).
 * Rather than
 * gather each of the N weight rows back to true K order (a per-row random-access
 * gather, the dominant cost on m200), we store the weight field-major and
 * de-interleave the (small) activation x into the SAME field-major K order once
 * per core. The Cube contracts x_fm against w_fm over the permuted K, giving the
 * identical result with only T gathers instead of N.
 */

#ifndef W2_BLOCKED_DEQUANT_MATMUL_V310_H
#define W2_BLOCKED_DEQUANT_MATMUL_V310_H

#define CATLASS_ARCH 2201
#define CATLASS_UNIFIED_CORE 1

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "kernel_operator.h"

namespace NsW2 {

using namespace AscendC;
using namespace Catlass;
using namespace tla;

constexpr int64_t W2_BLK = 32;
constexpr uint32_t W2_TILE_N = 128;

template <typename T>
__aicore__ inline T CeilDivU(T a, T b) { return (b == 0) ? 0 : (a + b - 1) / b; }
template <typename T>
__aicore__ inline T AlignUpU(T a, T b) { return (b == 0) ? 0 : (a + b - 1) / b * b; }
template <typename T>
__aicore__ inline T MinU(T a, T b) { return (a < b) ? a : b; }

class W2BlockedDequantMatmulV310Cube {
public:
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeTla = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShapeTla = L1TileShapeTla;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, half, layout::RowMajor, half, layout::ColumnMajor, float, layout::RowMajor>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, half, half, float, void, TileCopy>;

    __aicore__ inline W2BlockedDequantMatmulV310Cube() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR codes, GM_ADDR blockScale, GM_ADDR y,
                                GM_ADDR user, GM_ADDR tiling)
    {
        __gm__ W2BlockedDequantMatmulTilingData *__restrict td =
            reinterpret_cast<__gm__ W2BlockedDequantMatmulTilingData *__restrict>(tiling);
        T_ = td->numTokens;
        N_ = td->nDim;
        K_ = td->kDim;
        codesPerByte_ = td->codesPerByte;
        bitsPerCode_ = 8 / codesPerByte_;
        packedK_ = K_ / codesPerByte_;
        blockPackedCols_ = W2_BLK / codesPerByte_;
        fieldMask_ = (1 << bitsPerCode_) - 1;
        signHalf_ = 1 << (bitsPerCode_ - 1);
        kbCount_ = K_ / W2_BLK;
        mAligned_ = AlignUpU<int64_t>(T_, 16);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        codesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(codes));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(blockScale));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));
        wdqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(user));
        yfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + N_ * K_ * sizeof(half)));
        // xfm workspace follows [Wdq][yF]; per-core slice of mAligned*K halfs.
        xfmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(
            user + N_ * K_ * sizeof(half) + mAligned_ * N_ * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        BlockMmad blockMmad(resource);

        AllocBuffers();
        FillTables();

        // De-interleave x into field-major K order, once, into this core slice.
        const int64_t xfmBase = static_cast<int64_t>(coreId) * mAligned_ * K_;
        DeinterleaveX(xfmBase);
        AscendC::PipeBarrier<PIPE_ALL>();

        auto aLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)K_);
        auto bLayout = tla::MakeLayout<half, layout::ColumnMajor>((uint32_t)K_, (uint32_t)N_);
        auto cLayout = tla::MakeLayout<float, layout::RowMajor>((uint32_t)mAligned_, (uint32_t)N_);
        auto tensorA = tla::MakeTensor(xfmGm_[xfmBase], aLayout, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(wdqGm_, bLayout, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(yfGm_, cLayout, Arch::PositionGM{});

        const uint32_t nBlocks = CeilDivU<uint32_t>((uint32_t)N_, W2_TILE_N);

        for (uint32_t nb = coreId; nb < nBlocks; nb += coreNum) {
            uint32_t n0 = nb * W2_TILE_N;
            uint32_t nActual = MinU<uint32_t>(W2_TILE_N, (uint32_t)N_ - n0);

            DequantBlock(n0, nActual);
            AscendC::PipeBarrier<PIPE_ALL>();

            GemmCoord shape{(uint32_t)T_, nActual, (uint32_t)K_};
            auto tA = GetTile(tensorA, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                              tla::MakeShape((uint32_t)T_, (uint32_t)K_));
            auto tB = GetTile(tensorB, tla::MakeCoord((uint32_t)0, n0),
                              tla::MakeShape((uint32_t)K_, nActual));
            auto tC = GetTile(tensorC, tla::MakeCoord((uint32_t)0, n0),
                              tla::MakeShape((uint32_t)T_, nActual));
            blockMmad.preSetFlags();
            blockMmad(tA, tB, tC, shape);
            blockMmad.finalWaitFlags();
            AscendC::PipeBarrier<PIPE_ALL>();

            CastOut(n0, nActual);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void AllocBuffers()
    {
        const int64_t K = K_;
        const int64_t packedK = packedK_;
        uint32_t off = 0;
        scaleUB_ = resource.ubBuf.template GetBufferByByte<float>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)kbCount_ * sizeof(float), 512);
        cU8_ = resource.ubBuf.template GetBufferByByte<uint8_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedK * sizeof(uint8_t), 512);
        cH_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedK * sizeof(half), 512);
        c16_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedK * sizeof(int16_t), 512);
        andTmp_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedK * sizeof(int16_t), 512);
        mjhUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedK * sizeof(half), 512);
        f16UB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(int16_t), 512);
        fhUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(half), 512);
        wHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(half), 512);
        castFUB_ = resource.ubBuf.template GetBufferByByte<float>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_TILE_N * sizeof(float), 512);
        castHUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_TILE_N * sizeof(half), 512);
        deintOffUB_ = resource.ubBuf.template GetBufferByByte<uint32_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(uint32_t), 512);
        threeUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(int16_t), 512);
        twoUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(int16_t), 512);
        masksUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(int16_t), 512);
        rowScaleUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)K * sizeof(half), 512);
    }

    __aicore__ inline void FillTables()
    {
        const int64_t K = K_;
        const int64_t packedK = packedK_;
        Duplicate(threeUB_, static_cast<int16_t>(fieldMask_), (int32_t)K);
        Duplicate(twoUB_, static_cast<int16_t>(signHalf_), (int32_t)K);
        for (int64_t j = 0; j < codesPerByte_; ++j) {
            Duplicate(masksUB_[j * packedK],
                      static_cast<int16_t>(fieldMask_ << (bitsPerCode_ * j)),
                      (int32_t)packedK);
        }
        // x de-interleave gather (byte offsets): xfm[p] follows packed fields.
        for (int64_t i = 0; i < packedK; ++i) {
            for (int64_t j = 0; j < codesPerByte_; ++j) {
                int64_t p = j * packedK + i;
                uint32_t src = (uint32_t)(codesPerByte_ * i + j);
                deintOffUB_.SetValue(p, src * (uint32_t)sizeof(half));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // Build this core field-major copy of x[T, K] at xfmGm_[xfmBase ..].
    __aicore__ inline void DeinterleaveX(int64_t xfmBase)
    {
        const int32_t K = (int32_t)K_;
        for (int64_t t = 0; t < T_; ++t) {
            DataCopy(fhUB_, xGm_[t * K_], K);           // reuse fhUB_ as x-in
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Gather(wHalfUB_, fhUB_, deintOffUB_, (uint32_t)0, (uint32_t)K);  // x-out
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopy(xfmGm_[xfmBase + t * K_], wHalfUB_, K);
            PipeBarrier<PIPE_ALL>();  // fully fence: next row reuses fhUB_/wHalfUB_
        }
    }

    __aicore__ inline void DequantBlock(uint32_t n0, uint32_t nActual)
    {
        const int32_t K = (int32_t)K_;
        const int32_t packedK = (int32_t)packedK_;

        for (uint32_t rBase = 0; rBase < nActual; rBase += W2_BLK) {
            uint32_t rows = MinU<uint32_t>((uint32_t)W2_BLK, nActual - rBase);
            int64_t nb = ((int64_t)n0 + rBase) / W2_BLK;

            // Field-major row scale: block m (kb) covers field-major positions
            // { j*K4 + i : i in [8m, 8m+8), j in 0..3 }.
            DataCopy(scaleUB_, scaleGm_[nb * kbCount_], kbCount_);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            for (int64_t kb = 0; kb < kbCount_; ++kb) {
                half sv = static_cast<half>(scaleUB_.GetValue(kb));
                for (int64_t j = 0; j < codesPerByte_; ++j) {
                    Duplicate(rowScaleUB_[j * packedK + kb * blockPackedCols_], sv,
                              (int32_t)blockPackedCols_);
                }
            }
            PipeBarrier<PIPE_V>();

            for (uint32_t rr = 0; rr < rows; ++rr) {
                int64_t n = (int64_t)n0 + rBase + rr;

                if (rr != 0 || rBase != 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                DataCopy(cU8_, codesGm_[n * packedK_], packedK);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                Cast(cH_, cU8_, RoundMode::CAST_NONE, packedK);
                PipeBarrier<PIPE_V>();
                Cast(c16_, cH_, RoundMode::CAST_RINT, packedK);
                PipeBarrier<PIPE_V>();

                for (int32_t j = 0; j < codesPerByte_; ++j) {
                    And(andTmp_, c16_, masksUB_[j * packedK], packedK);
                    PipeBarrier<PIPE_V>();
                    Cast(mjhUB_, andTmp_, RoundMode::CAST_NONE, packedK);
                    PipeBarrier<PIPE_V>();
                    half fieldRecip = static_cast<half>(
                        1.0f / static_cast<float>(1 << (bitsPerCode_ * j)));
                    Muls(mjhUB_, mjhUB_, fieldRecip, packedK);
                    PipeBarrier<PIPE_V>();
                    Cast(f16UB_[j * packedK], mjhUB_, RoundMode::CAST_RINT, packedK);
                    PipeBarrier<PIPE_V>();
                }

                Add(f16UB_, f16UB_, twoUB_, K);
                PipeBarrier<PIPE_V>();
                And(f16UB_, f16UB_, threeUB_, K);
                PipeBarrier<PIPE_V>();
                Sub(f16UB_, f16UB_, twoUB_, K);
                PipeBarrier<PIPE_V>();

                Cast(fhUB_, f16UB_, RoundMode::CAST_NONE, K);
                PipeBarrier<PIPE_V>();
                Mul(wHalfUB_, fhUB_, rowScaleUB_, K);

                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(wdqGm_[n * K_], wHalfUB_, K);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void CastOut(uint32_t n0, uint32_t nActual)
    {
        for (int64_t t = 0; t < T_; ++t) {
            DataCopy(castFUB_, yfGm_[t * N_ + n0], nActual);
            PipeBarrier<PIPE_ALL>();
            Cast(castHUB_, castFUB_, RoundMode::CAST_NONE, nActual);
            PipeBarrier<PIPE_ALL>();
            DataCopy(yGm_[t * N_ + n0], castHUB_, nActual);
            PipeBarrier<PIPE_ALL>();
        }
    }

    Arch::Resource<ArchTag> resource;

    GlobalTensor<half> xGm_;
    GlobalTensor<uint8_t> codesGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<half> yGm_;
    GlobalTensor<half> wdqGm_;
    GlobalTensor<float> yfGm_;
    GlobalTensor<half> xfmGm_;

    LocalTensor<uint32_t> deintOffUB_;
    LocalTensor<int16_t> threeUB_;
    LocalTensor<int16_t> twoUB_;
    LocalTensor<int16_t> masksUB_;
    LocalTensor<half> rowScaleUB_;
    LocalTensor<float> scaleUB_;
    LocalTensor<uint8_t> cU8_;
    LocalTensor<half> cH_;
    LocalTensor<int16_t> c16_;
    LocalTensor<int16_t> andTmp_;
    LocalTensor<half> mjhUB_;
    LocalTensor<int16_t> f16UB_;
    LocalTensor<half> fhUB_;
    LocalTensor<half> wHalfUB_;
    LocalTensor<float> castFUB_;
    LocalTensor<half> castHUB_;

    int64_t T_;
    int64_t N_;
    int64_t K_;
    int64_t codesPerByte_;
    int64_t bitsPerCode_;
    int64_t packedK_;
    int64_t blockPackedCols_;
    int64_t fieldMask_;
    int64_t signHalf_;
    int64_t kbCount_;
    int64_t mAligned_;
};

}  // namespace NsW2
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_H
