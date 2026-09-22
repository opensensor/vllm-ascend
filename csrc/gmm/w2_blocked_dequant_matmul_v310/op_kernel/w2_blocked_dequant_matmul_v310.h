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
constexpr uint32_t W2_TILE_K = 128;
constexpr uint32_t W2_K_FRACTALS_PER_TILE = W2_TILE_K / W2_FRACTAL_SIZE;

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
        ArchTag, half, layout::RowMajor, half, layout::zN, half, layout::RowMajor>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, half, half, half, void, TileCopy>;

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
        packedTileCols_ = W2_TILE_K / codesPerByte_;
        packedTileCount_ = W2_FRACTAL_SIZE * packedTileCols_;
        decodedTileCount_ = W2_FRACTAL_SIZE * W2_TILE_K;
        fieldMask_ = (1 << bitsPerCode_) - 1;
        signHalf_ = 1 << (bitsPerCode_ - 1);
        kbCount_ = K_ / W2_BLOCK_SIZE;

        const int64_t nzElementsPerCore = W2_TILE_N * K_;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        codesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(codes));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(blockScale));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));
        wdqNzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(user));

        coreNzBase_ = static_cast<int64_t>(GetBlockIdx()) * nzElementsPerCore;
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        const uint32_t nBlocks = CeilDivU<uint32_t>((uint32_t)N_, W2_TILE_N);

        AllocBuffers();
        FillTables();

        auto aLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)K_);
        auto bLayout = tla::MakeLayout<half, layout::zN>((uint32_t)K_, W2_TILE_N);
        auto cLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)N_);
        auto tensorA = tla::MakeTensor(xGm_, aLayout, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(wdqNzGm_[coreNzBase_], bLayout, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(yGm_, cLayout, Arch::PositionGM{});

        for (uint32_t nb = coreId; nb < nBlocks; nb += coreNum) {
            const uint32_t n0 = nb * W2_TILE_N;
            const uint32_t nActual = MinU<uint32_t>(W2_TILE_N, (uint32_t)N_ - n0);

            DequantTileToNz(n0, nActual);
            // The dequantizer writes this core's packed-NZ workspace through
            // MTE3, while BlockMmad consumes it from GM through MTE2.  A pipe
            // barrier does not order those engines on 310P.
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);

            GemmCoord shape{(uint32_t)T_, nActual, (uint32_t)K_};
            auto tA = GetTile(tensorA, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                              tla::MakeShape((uint32_t)T_, (uint32_t)K_));
            auto tB = GetTile(tensorB, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                              tla::MakeShape((uint32_t)K_, nActual));
            auto tC = GetTile(tensorC, tla::MakeCoord((uint32_t)0, n0),
                              tla::MakeShape((uint32_t)T_, nActual));
            // BlockMmad carries ping-pong stage indices. Reconstruct it for
            // every N tile so cores that process multiple tiles restart from
            // the same synchronized stage state.
            BlockMmad blockMmad(resource);
            blockMmad.preSetFlags();
            blockMmad(tA, tB, tC, shape);
            blockMmad.finalWaitFlags();
            // A core can process several N tiles and reuse the same GM
            // workspace.  Finish BlockMmad's MTE2 reads before the next tile
            // overwrites that workspace through MTE3.
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID4);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID4);
        }
    }

private:
    __aicore__ inline void AllocBuffers()
    {
        uint32_t off = 0;
        scaleUB_ = resource.ubBuf.template GetBufferByByte<float>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)kbCount_ * sizeof(float), 512);
        cU8_ = resource.ubBuf.template GetBufferByByte<uint8_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(uint8_t), 512);
        cH_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(half), 512);
        c16_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(int16_t), 512);
        andTmp_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(int16_t), 512);
        fieldHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)packedTileCount_ * sizeof(half), 512);
        fieldI16UB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
        signedHalfUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(half), 512);
        fractalRowsUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(half), 512);
        gatherOffsetsUB_ = resource.ubBuf.template GetBufferByByte<uint32_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(uint32_t), 512);
        threeUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
        signHalfUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
        masksUB_ = resource.ubBuf.template GetBufferByByte<int16_t>(off);
        off = AlignUpU<uint32_t>(off + (uint32_t)decodedTileCount_ * sizeof(int16_t), 512);
        nzFractalUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + W2_FRACTAL_SIZE * W2_FRACTAL_SIZE * sizeof(half), 512);
    }

    __aicore__ inline void FillTables()
    {
        Duplicate(threeUB_, static_cast<int16_t>(fieldMask_), (int32_t)decodedTileCount_);
        Duplicate(signHalfUB_, static_cast<int16_t>(signHalf_), (int32_t)decodedTileCount_);
        for (int64_t field = 0; field < codesPerByte_; ++field) {
            Duplicate(masksUB_[field * packedTileCount_],
                      static_cast<int16_t>(fieldMask_ << (bitsPerCode_ * field)),
                      (int32_t)packedTileCount_);
        }

        // The vector unpack is field-major across the complete packed tile.
        // Gather directly into eight [N=16,K=16] row-major fragments, avoiding
        // an intermediate logical [16,128] tensor and 128 small UB copies.
        for (int64_t row = 0; row < W2_FRACTAL_SIZE; ++row) {
            for (int64_t k = 0; k < W2_TILE_K; ++k) {
                const int64_t byte = k / codesPerByte_;
                const int64_t field = k % codesPerByte_;
                const int64_t decoded = field * packedTileCount_ + row * packedTileCols_ + byte;
                const int64_t kFractal = k / W2_FRACTAL_SIZE;
                const int64_t kWithinFractal = k % W2_FRACTAL_SIZE;
                const int64_t reordered =
                    kFractal * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE + row * W2_FRACTAL_SIZE + kWithinFractal;
                gatherOffsetsUB_.SetValue(reordered, (uint32_t)(decoded * sizeof(half)));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void DecodeTile(int64_t codeOffset)
    {
        // Copy one packed row at a time. A strided 2-D GM-to-UB DataCopy looks
        // attractive here, but dav_m200 rejects that descriptor at runtime
        // with an MTE "burst num" exception even when every row and stride is
        // 32-byte aligned. The scalar overload is hardware-proven on 310P and
        // still batches all 128 logical K values for the vector decode below.
        for (uint32_t row = 0; row < W2_FRACTAL_SIZE; ++row) {
            DataCopy(cU8_[row * packedTileCols_],
                     codesGm_[codeOffset + static_cast<int64_t>(row) * packedK_],
                     static_cast<int32_t>(packedTileCols_));
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(cH_, cU8_, RoundMode::CAST_NONE, (int32_t)packedTileCount_);
        PipeBarrier<PIPE_V>();
        Cast(c16_, cH_, RoundMode::CAST_RINT, (int32_t)packedTileCount_);
        PipeBarrier<PIPE_V>();

        for (int32_t field = 0; field < codesPerByte_; ++field) {
            And(andTmp_, c16_, masksUB_[field * packedTileCount_], (int32_t)packedTileCount_);
            PipeBarrier<PIPE_V>();
            Cast(fieldHalfUB_, andTmp_, RoundMode::CAST_NONE, (int32_t)packedTileCount_);
            PipeBarrier<PIPE_V>();
            const half fieldRecip = static_cast<half>(
                1.0f / static_cast<float>(1 << (bitsPerCode_ * field)));
            Muls(fieldHalfUB_, fieldHalfUB_, fieldRecip, (int32_t)packedTileCount_);
            PipeBarrier<PIPE_V>();
            Cast(fieldI16UB_[field * packedTileCount_], fieldHalfUB_,
                 RoundMode::CAST_RINT, (int32_t)packedTileCount_);
            PipeBarrier<PIPE_V>();
        }

        // Sign-extend the two's-complement W2/W4 field in int16.
        Add(fieldI16UB_, fieldI16UB_, signHalfUB_, (int32_t)decodedTileCount_);
        PipeBarrier<PIPE_V>();
        And(fieldI16UB_, fieldI16UB_, threeUB_, (int32_t)decodedTileCount_);
        PipeBarrier<PIPE_V>();
        Sub(fieldI16UB_, fieldI16UB_, signHalfUB_, (int32_t)decodedTileCount_);
        PipeBarrier<PIPE_V>();
        Cast(signedHalfUB_, fieldI16UB_, RoundMode::CAST_NONE, (int32_t)decodedTileCount_);
        PipeBarrier<PIPE_V>();

        Gather(fractalRowsUB_, signedHalfUB_, gatherOffsetsUB_, (uint32_t)0,
               (uint32_t)decodedTileCount_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DequantTileToNz(uint32_t n0, uint32_t nActual)
    {
        for (uint32_t scaleRowBase = 0; scaleRowBase < nActual; scaleRowBase += W2_BLOCK_SIZE) {
            const int64_t scaleRow = (static_cast<int64_t>(n0) + scaleRowBase) / W2_BLOCK_SIZE;
            DataCopy(scaleUB_, scaleGm_[scaleRow * kbCount_], (int32_t)kbCount_);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);

            for (uint32_t rowInScale = 0; rowInScale < W2_BLOCK_SIZE; rowInScale += W2_FRACTAL_SIZE) {
                const uint32_t rowBase = scaleRowBase + rowInScale;
                for (int64_t k0 = 0; k0 < K_; k0 += W2_TILE_K) {
                    const int64_t codeOffset =
                        (static_cast<int64_t>(n0) + rowBase) * packedK_ + k0 / codesPerByte_;
                    DecodeTile(codeOffset);

                    // W is [N,K], while Cube consumes B=W^T [K,N]. Transpose each
                    // 16x16 fragment, then apply its [32,32] scale once to all 256
                    // values before the already-NZ GM store.
                    const int64_t nFractal = rowBase / W2_FRACTAL_SIZE;
                    const int64_t nzColumnBlockStride = K_ * W2_FRACTAL_SIZE;
                    const int64_t nzBase = coreNzBase_ + nFractal * nzColumnBlockStride;
                    for (int64_t kFractal = 0; kFractal < W2_K_FRACTALS_PER_TILE; ++kFractal) {
                        const int64_t localFractalOffset =
                            kFractal * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE;
                        const int64_t globalKFractal = k0 / W2_FRACTAL_SIZE + kFractal;
                        const int64_t scaleIndex = k0 / W2_BLOCK_SIZE + kFractal / 2;
                        const half scale = static_cast<half>(scaleUB_.GetValue(scaleIndex));
                        AscendC::Transpose(nzFractalUB_, fractalRowsUB_[localFractalOffset]);
                        PipeBarrier<PIPE_V>();
                        Muls(nzFractalUB_, nzFractalUB_, scale,
                             W2_FRACTAL_SIZE * W2_FRACTAL_SIZE);
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                        DataCopy(wdqNzGm_[nzBase + globalKFractal * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE],
                                 nzFractalUB_, W2_FRACTAL_SIZE * W2_FRACTAL_SIZE);
                        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                    }
                }
            }
        }
    }

    Arch::Resource<ArchTag> resource;

    GlobalTensor<half> xGm_;
    GlobalTensor<uint8_t> codesGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<half> yGm_;
    GlobalTensor<half> wdqNzGm_;

    LocalTensor<float> scaleUB_;
    LocalTensor<uint8_t> cU8_;
    LocalTensor<half> cH_;
    LocalTensor<int16_t> c16_;
    LocalTensor<int16_t> andTmp_;
    LocalTensor<half> fieldHalfUB_;
    LocalTensor<int16_t> fieldI16UB_;
    LocalTensor<half> signedHalfUB_;
    LocalTensor<half> fractalRowsUB_;
    LocalTensor<uint32_t> gatherOffsetsUB_;
    LocalTensor<int16_t> threeUB_;
    LocalTensor<int16_t> signHalfUB_;
    LocalTensor<int16_t> masksUB_;
    LocalTensor<half> nzFractalUB_;

    int64_t T_;
    int64_t N_;
    int64_t K_;
    int64_t codesPerByte_;
    int64_t bitsPerCode_;
    int64_t packedK_;
    int64_t packedTileCols_;
    int64_t packedTileCount_;
    int64_t decodedTileCount_;
    int64_t fieldMask_;
    int64_t signHalf_;
    int64_t kbCount_;
    int64_t coreNzBase_;
};

}  // namespace NsW2
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_H
