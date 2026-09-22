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
 * \brief 310P packed W2/W4 block-dequant Cube matmul.
 *
 * The checkpoint remains byte-packed in canonical row-major order. Each AI
 * core decodes only its current 128-output-channel tile, restores logical K
 * order in UB, and writes 16x16 fragments directly in NZ order to a reusable
 * per-core workspace. CATLASS therefore consumes an already-NZ B operand:
 * there is no full [N,K] fp16 materialization, no activation de-interleave,
 * and no ND-to-NZ conversion in the matmul path.
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

constexpr int64_t W2_BLOCK_SIZE = 32;
constexpr uint32_t W2_FRACTAL_SIZE = 16;
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
        ArchTag, half, layout::RowMajor, half, layout::zN, float, layout::RowMajor>;
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
        blockPackedCols_ = W2_BLOCK_SIZE / codesPerByte_;
        fieldMask_ = (1 << bitsPerCode_) - 1;
        signHalf_ = 1 << (bitsPerCode_ - 1);
        kbCount_ = K_ / W2_BLOCK_SIZE;
        mAligned_ = AlignUpU<int64_t>(T_, W2_FRACTAL_SIZE);

        const int64_t coreCount = GetBlockNum();
        const int64_t nzElementsPerCore = W2_TILE_N * K_;
        const int64_t outputElementsPerCore = mAligned_ * W2_TILE_N;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        codesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(codes));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(blockScale));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));
        wdqNzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(user));
        yfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            user + coreCount * nzElementsPerCore * sizeof(half)));

        coreNzBase_ = static_cast<int64_t>(GetBlockIdx()) * nzElementsPerCore;
        coreYfBase_ = static_cast<int64_t>(GetBlockIdx()) * outputElementsPerCore;
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        const uint32_t nBlocks = CeilDivU<uint32_t>((uint32_t)N_, W2_TILE_N);
        BlockMmad blockMmad(resource);

        AllocBuffers();
        FillTables();

        auto aLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)K_);
        auto bLayout = tla::MakeLayout<half, layout::zN>((uint32_t)K_, W2_TILE_N);
        auto cLayout = tla::MakeLayout<float, layout::RowMajor>((uint32_t)mAligned_, W2_TILE_N);
        auto tensorA = tla::MakeTensor(xGm_, aLayout, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(wdqNzGm_[coreNzBase_], bLayout, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(yfGm_[coreYfBase_], cLayout, Arch::PositionGM{});

        for (uint32_t nb = coreId; nb < nBlocks; nb += coreNum) {
            const uint32_t n0 = nb * W2_TILE_N;
            const uint32_t nActual = MinU<uint32_t>(W2_TILE_N, (uint32_t)N_ - n0);

            DequantTileToNz(n0, nActual);
            AscendC::PipeBarrier<PIPE_ALL>();

            GemmCoord shape{(uint32_t)T_, nActual, (uint32_t)K_};
            auto tA = GetTile(tensorA, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                              tla::MakeShape((uint32_t)T_, (uint32_t)K_));
            auto tB = GetTile(tensorB, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                              tla::MakeShape((uint32_t)K_, nActual));
            auto tC = GetTile(tensorC, tla::MakeCoord((uint32_t)0, (uint32_t)0),
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
        uint32_t off = 0;
        scaleUB_ = resource.ubBuf.template GetBufferByByte<float>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)kbCount_ * sizeof(float), 512);
        cU8_ = resource.ubBuf.template GetBufferByByte<uint8_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(uint8_t), 512);
        cH_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(half), 512);
        c16_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(int16_t), 512);
        andTmp_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(int16_t), 512);
        fieldHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(half), 512);
        fieldI16UB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(int16_t), 512);
        signedHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(half), 512);
        logicalHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(half), 512);
        gatherOffsetsUB_ = resource.ubBuf.template GetBufferByByte<uint32_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(uint32_t), 512);
        threeUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(int16_t), 512);
        signHalfUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(int16_t), 512);
        masksUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)W2_BLOCK_SIZE * sizeof(int16_t), 512);
        tileLoUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + W2_FRACTAL_SIZE * W2_FRACTAL_SIZE * sizeof(half), 512);
        tileHiUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + W2_FRACTAL_SIZE * W2_FRACTAL_SIZE * sizeof(half), 512);
        nzFractalUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + W2_FRACTAL_SIZE * W2_FRACTAL_SIZE * sizeof(half), 512);
        castFUB_ = resource.ubBuf.template GetBufferByByte<float>(off);
        off = AlignUpU<uint32_t>(off + W2_TILE_N * sizeof(float), 512);
        castHUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
    }

    __aicore__ inline void FillTables()
    {
        Duplicate(threeUB_, static_cast<int16_t>(fieldMask_), (int32_t)W2_BLOCK_SIZE);
        Duplicate(signHalfUB_, static_cast<int16_t>(signHalf_), (int32_t)W2_BLOCK_SIZE);
        for (int64_t field = 0; field < codesPerByte_; ++field) {
            Duplicate(masksUB_[field * blockPackedCols_],
                      static_cast<int16_t>(fieldMask_ << (bitsPerCode_ * field)),
                      (int32_t)blockPackedCols_);
        }

        // Packed bytes decode field-major: all low fields, then all next fields.
        // This gather restores the checkpoint's logical adjacent-code order.
        for (int64_t byte = 0; byte < blockPackedCols_; ++byte) {
            for (int64_t field = 0; field < codesPerByte_; ++field) {
                const int64_t logical = byte * codesPerByte_ + field;
                const int64_t decoded = field * blockPackedCols_ + byte;
                gatherOffsetsUB_.SetValue(logical, (uint32_t)(decoded * sizeof(half)));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void DecodeRowBlock(int64_t codeOffset, half scale)
    {
        const int32_t packedCount = (int32_t)blockPackedCols_;
        const int32_t decodedCount = (int32_t)W2_BLOCK_SIZE;

        DataCopy(cU8_, codesGm_[codeOffset], packedCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(cH_, cU8_, RoundMode::CAST_NONE, packedCount);
        PipeBarrier<PIPE_V>();
        Cast(c16_, cH_, RoundMode::CAST_RINT, packedCount);
        PipeBarrier<PIPE_V>();

        for (int32_t field = 0; field < codesPerByte_; ++field) {
            And(andTmp_, c16_, masksUB_[field * blockPackedCols_], packedCount);
            PipeBarrier<PIPE_V>();
            Cast(fieldHalfUB_, andTmp_, RoundMode::CAST_NONE, packedCount);
            PipeBarrier<PIPE_V>();
            const half fieldRecip = static_cast<half>(
                1.0f / static_cast<float>(1 << (bitsPerCode_ * field)));
            Muls(fieldHalfUB_, fieldHalfUB_, fieldRecip, packedCount);
            PipeBarrier<PIPE_V>();
            Cast(fieldI16UB_[field * blockPackedCols_], fieldHalfUB_,
                 RoundMode::CAST_RINT, packedCount);
            PipeBarrier<PIPE_V>();
        }

        // Sign-extend the two's-complement W2/W4 field in int16.
        Add(fieldI16UB_, fieldI16UB_, signHalfUB_, decodedCount);
        PipeBarrier<PIPE_V>();
        And(fieldI16UB_, fieldI16UB_, threeUB_, decodedCount);
        PipeBarrier<PIPE_V>();
        Sub(fieldI16UB_, fieldI16UB_, signHalfUB_, decodedCount);
        PipeBarrier<PIPE_V>();
        Cast(signedHalfUB_, fieldI16UB_, RoundMode::CAST_NONE, decodedCount);
        PipeBarrier<PIPE_V>();

        Gather(logicalHalfUB_, signedHalfUB_, gatherOffsetsUB_, (uint32_t)0,
               (uint32_t)decodedCount);
        PipeBarrier<PIPE_V>();
        Muls(logicalHalfUB_, logicalHalfUB_, scale, decodedCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DequantTileToNz(uint32_t n0, uint32_t nActual)
    {
        for (uint32_t rowBase = 0; rowBase < nActual; rowBase += W2_FRACTAL_SIZE) {
            const uint32_t rows = MinU<uint32_t>(W2_FRACTAL_SIZE, nActual - rowBase);
            const int64_t scaleRow = (static_cast<int64_t>(n0) + rowBase) / W2_BLOCK_SIZE;
            DataCopy(scaleUB_, scaleGm_[scaleRow * kbCount_], (int32_t)kbCount_);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);

            for (int64_t kb = 0; kb < kbCount_; ++kb) {
                const half scale = static_cast<half>(scaleUB_.GetValue(kb));
                for (uint32_t rr = 0; rr < rows; ++rr) {
                    const int64_t n = static_cast<int64_t>(n0) + rowBase + rr;
                    const int64_t codeOffset = n * packedK_ + kb * blockPackedCols_;
                    DecodeRowBlock(codeOffset, scale);
                    DataCopy(tileLoUB_[rr * W2_FRACTAL_SIZE], logicalHalfUB_, W2_FRACTAL_SIZE);
                    DataCopy(tileHiUB_[rr * W2_FRACTAL_SIZE],
                             logicalHalfUB_[W2_FRACTAL_SIZE], W2_FRACTAL_SIZE);
                    PipeBarrier<PIPE_V>();
                }

                // W is [N,K], while Cube consumes B=W^T [K,N]. Transposing
                // each 16x16 half-block produces exactly the zN fractal bytes.
                const int64_t nFractal = rowBase / W2_FRACTAL_SIZE;
                const int64_t nzColumnBlockStride = K_ * W2_FRACTAL_SIZE;
                const int64_t nzBase = coreNzBase_ + nFractal * nzColumnBlockStride;
                AscendC::Transpose(nzFractalUB_, tileLoUB_);
                PipeBarrier<PIPE_V>();
                DataCopy(wdqNzGm_[nzBase + (2 * kb) * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE],
                         nzFractalUB_, W2_FRACTAL_SIZE * W2_FRACTAL_SIZE);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::Transpose(nzFractalUB_, tileHiUB_);
                PipeBarrier<PIPE_V>();
                DataCopy(wdqNzGm_[nzBase + (2 * kb + 1) * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE],
                         nzFractalUB_, W2_FRACTAL_SIZE * W2_FRACTAL_SIZE);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void CastOut(uint32_t n0, uint32_t nActual)
    {
        for (int64_t t = 0; t < T_; ++t) {
            DataCopy(castFUB_, yfGm_[coreYfBase_ + t * W2_TILE_N], nActual);
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
    GlobalTensor<half> wdqNzGm_;
    GlobalTensor<float> yfGm_;

    LocalTensor<float> scaleUB_;
    LocalTensor<uint8_t> cU8_;
    LocalTensor<half> cH_;
    LocalTensor<int16_t> c16_;
    LocalTensor<int16_t> andTmp_;
    LocalTensor<half> fieldHalfUB_;
    LocalTensor<int16_t> fieldI16UB_;
    LocalTensor<half> signedHalfUB_;
    LocalTensor<half> logicalHalfUB_;
    LocalTensor<uint32_t> gatherOffsetsUB_;
    LocalTensor<int16_t> threeUB_;
    LocalTensor<int16_t> signHalfUB_;
    LocalTensor<int16_t> masksUB_;
    LocalTensor<half> tileLoUB_;
    LocalTensor<half> tileHiUB_;
    LocalTensor<half> nzFractalUB_;
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
    int64_t coreNzBase_;
    int64_t coreYfBase_;
};

}  // namespace NsW2
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_H
