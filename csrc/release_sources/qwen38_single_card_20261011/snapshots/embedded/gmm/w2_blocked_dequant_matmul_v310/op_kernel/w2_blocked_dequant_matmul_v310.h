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
 * order in UB, and writes 16x16 fragments directly in NZ order. The grouped
 * NZ-packed path retains one decoded B tile in L1 and feeds Cube directly;
 * the canonical and single-expert paths use a reusable GM workspace.
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
constexpr uint32_t W2_TILE_M = 128;
constexpr uint32_t W2_TILE_N = 128;
constexpr uint32_t W2_TILE_K = 128;
constexpr uint32_t W2_K_FRACTALS_PER_TILE = W2_TILE_K / W2_FRACTAL_SIZE;
constexpr uint32_t W2_L1_TILE_N = 32;
constexpr uint32_t W2_L1_MAX_K = 4096;
constexpr uint32_t W2_L1_WEIGHT_BYTES = W2_L1_TILE_N * W2_L1_MAX_K * sizeof(half);
constexpr uint32_t W2_L1_A_STAGES = 2;
constexpr uint32_t W2_L1_A_STAGE_BYTES = W2_TILE_M * W2_TILE_K * sizeof(half);
constexpr uint32_t W2_L1_MAX_CUBE_K = 512;
static_assert(W2_BLOCK_SIZE == 2 * W2_FRACTAL_SIZE);
// The unified-core CATLASS epilogue uses UB [0, 96 KiB) for a maximum-size
// 128x128 FP32 accumulator followed by its FP16 output.  Keep dequant scratch
// and, critically, the reusable masks/gather table above that region so a
// logical block can safely process more than one output-channel tile.
constexpr uint32_t W2_MMAD_EPILOGUE_UB_BYTES =
    W2_TILE_M * W2_TILE_N * (sizeof(float) + sizeof(half));

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
    using ResidentTileShapeTla = tla::Shape<tla::Int<128>, tla::Int<W2_L1_TILE_N>, tla::Int<128>>;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, half, layout::RowMajor, half, layout::zN, half, layout::RowMajor>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, half, half, half, void, TileCopy>;
    using ResidentBlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicyTla, ResidentTileShapeTla, ResidentTileShapeTla, half, half, half, void, TileCopy>;
    static_assert(W2_L1_WEIGHT_BYTES + W2_L1_A_STAGES * W2_L1_A_STAGE_BYTES <= ArchTag::L1_SIZE,
                  "GLM W2/W4 decoded tile and activation stages must fit L1");
    static_assert(W2_L1_A_STAGES * W2_L1_A_STAGE_BYTES <= ArchTag::L0A_SIZE &&
                      W2_L1_A_STAGES * W2_L1_TILE_N * W2_L1_MAX_CUBE_K * sizeof(half) <= ArchTag::L0B_SIZE,
                  "GLM W2/W4 Cube stages must fit L0");

    __aicore__ inline W2BlockedDequantMatmulV310Cube() {}

    // Shared by the single-expert and grouped-expert entry points. Keeping the
    // geometry explicit lets a grouped kernel advance through device-resident
    // expert banks without manufacturing host tiling data or synchronizing
    // route counts back to Python.
    __aicore__ inline void InitGeometry(GM_ADDR x, GM_ADDR codes, GM_ADDR blockScale,
                                        GM_ADDR y, GM_ADDR user, int64_t numTokens,
                                        int64_t nDim, int64_t kDim,
                                        int64_t codesPerByte, bool nzPacked = false,
                                        bool useL1 = false)
    {
        T_ = numTokens;
        N_ = nDim;
        K_ = kDim;
        codesPerByte_ = codesPerByte;
        nzPacked_ = nzPacked;
        useL1_ = useL1 && nzPacked && K_ <= W2_L1_MAX_K;
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
        if (useL1_) {
            ProcessFromL1();
            return;
        }
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
    __aicore__ inline void ProcessFromL1()
    {
        const uint32_t nBlocks = CeilDivU<uint32_t>((uint32_t)N_, W2_L1_TILE_N);
        const int64_t rows = T_;
        const auto input = xGm_;
        const auto output = yGm_;
        AllocBuffers();
        FillTables();
        for (uint32_t nb = GetBlockIdx(); nb < nBlocks; nb += GetBlockNum()) {
            const uint32_t n0 = nb * W2_L1_TILE_N;
            const uint32_t nActual = MinU<uint32_t>(W2_L1_TILE_N, (uint32_t)N_ - n0);
            DequantTileToNz(n0, nActual, true);
            SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID4);
            for (int64_t row = 0; row < rows; row += W2_TILE_M) {
                T_ = MinU<int64_t>(W2_TILE_M, rows - row);
                xGm_ = input[row * K_];
                yGm_ = output[row * N_];
                MatmulFromL1(n0, nActual, row == 0);
            }
            SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID4);
            WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID4);
        }
        T_ = rows;
        xGm_ = input;
        yGm_ = output;
    }

    __aicore__ inline void MatmulFromL1(uint32_t n0, uint32_t nActual, bool firstRow)
    {
        auto aLayout = tla::MakeLayout<half, layout::RowMajor>((uint32_t)T_, (uint32_t)K_);
        auto tensorA = tla::MakeTensor(xGm_, aLayout, Arch::PositionGM{});
        using CopyA = typename TileCopy::template CopyGmToL1A<decltype(tensorA)>;
        CopyA copyA;
        typename TileCopy::CopyL1ToL0A copyL0A;
        typename TileCopy::CopyL1ToL0B copyL0B;
        typename ResidentBlockMmad::TileMmad tileMmad;
        const uint32_t mActual = T_ == 1 ? W2_FRACTAL_SIZE : (uint32_t)T_;
        const uint32_t mAligned = AlignUpU<uint32_t>(mActual, W2_FRACTAL_SIZE);
        uint32_t cubeK = W2_L1_MAX_CUBE_K;
        while (cubeK > W2_TILE_K &&
               (K_ % cubeK != 0 || mAligned * cubeK * sizeof(half) > W2_L1_A_STAGE_BYTES)) {
            cubeK -= W2_FRACTAL_SIZE;
        }
        auto aL1Layout = tla::MakeLayout<half, typename TileCopy::LayoutTagL1A>(mAligned, cubeK);
        auto bL1Layout = tla::MakeLayout<half, typename TileCopy::LayoutTagL1B>((uint32_t)K_, nActual);
        auto tensorL1B = tla::MakeTensor(resource.l1Buf.template GetBufferByByte<half>(0), bL1Layout,
                                        Arch::PositionL1{});
        auto aL0Layout = tla::MakeLayout<half, typename TileCopy::LayoutTagL0A>(mActual, cubeK);
        auto bL0Layout = tla::MakeLayout<half, typename TileCopy::LayoutTagL0B>(cubeK, nActual);
        auto l0C = resource.l0CBuf.template GetBufferByByte<float>(0);
        auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(mActual, nActual), Arch::PositionL0C{});
        for (uint32_t stage = 0; stage < W2_L1_A_STAGES; ++stage) {
            SetFlag<HardEvent::MTE1_MTE2>(stage);
            SetFlag<HardEvent::M_MTE1>(stage);
        }
        if (firstRow) {
            WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID4);
        }
        for (uint32_t k0 = 0; k0 < K_; k0 += cubeK) {
            const uint32_t stage = (k0 / cubeK) % W2_L1_A_STAGES;
            auto l1A = resource.l1Buf.template GetBufferByByte<half>(
                W2_L1_WEIGHT_BYTES + stage * W2_L1_A_STAGE_BYTES);
            auto tensorL1A = tla::MakeTensor(l1A, aL1Layout, Arch::PositionL1{});
            auto tileA = GetTile(tensorA, tla::MakeCoord((uint32_t)0, k0),
                                 tla::MakeShape((uint32_t)T_, cubeK));
            WaitFlag<HardEvent::MTE1_MTE2>(stage);
            copyA(tensorL1A, tileA);
            SetFlag<HardEvent::MTE2_MTE1>(stage);
            auto l0A = resource.l0ABuf.template GetBufferByByte<half>(stage * W2_L1_A_STAGE_BYTES);
            auto l0B = resource.l0BBuf.template GetBufferByByte<half>(
                stage * W2_L1_TILE_N * cubeK * sizeof(half));
            auto tensorL0A = tla::MakeTensor(l0A, aL0Layout, Arch::PositionL0A{});
            auto tensorL0B = tla::MakeTensor(l0B, bL0Layout, Arch::PositionL0B{});
            auto tileL1A = GetTile(tensorL1A, tla::MakeCoord((uint32_t)0, (uint32_t)0),
                                   tla::MakeShape(mActual, cubeK));
            auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(k0, (uint32_t)0),
                                   tla::MakeShape(cubeK, nActual));
            WaitFlag<HardEvent::M_MTE1>(stage);
            WaitFlag<HardEvent::MTE2_MTE1>(stage);
            copyL0A(tensorL0A, tileL1A);
            copyL0B(tensorL0B, tileL1B);
            SetFlag<HardEvent::MTE1_MTE2>(stage);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            const uint8_t unitFlag = k0 + cubeK == K_ ? 0b11 : 0b10;
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, nActual, cubeK, k0 == 0, unitFlag);
            SetFlag<HardEvent::M_MTE1>(stage);
        }
        for (uint32_t stage = 0; stage < W2_L1_A_STAGES; ++stage) {
            WaitFlag<HardEvent::M_MTE1>(stage);
            WaitFlag<HardEvent::MTE1_MTE2>(stage);
        }
        StoreL1Accumulator(l0C, n0, nActual);
    }

    __aicore__ inline void StoreL1Accumulator(LocalTensor<float> accumulator, uint32_t n0,
                                               uint32_t nActual)
    {
        const uint32_t mAligned = AlignUpU<uint32_t>((uint32_t)T_, W2_FRACTAL_SIZE);
        const uint32_t nAligned = AlignUpU<uint32_t>(nActual, W2_FRACTAL_SIZE);
        const uint32_t elements = mAligned * nAligned;
        auto result = resource.ubBuf.template GetBufferByByte<float>(0);
        auto converted = resource.ubBuf.template GetBufferByByte<half>(elements * sizeof(float));
        SetFlag<HardEvent::M_V>(EVENT_ID7);
        WaitFlag<HardEvent::M_V>(EVENT_ID7);
        DataCopyParams fromCube;
        fromCube.blockCount = nAligned / W2_FRACTAL_SIZE;
        fromCube.blockLen = mAligned / W2_FRACTAL_SIZE;
        fromCube.srcStride = 0;
        fromCube.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(result, accumulator, fromCube, enhanced);
        Cast(converted, result, RoundMode::CAST_NONE, elements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        for (uint32_t nf = 0; nf < nAligned / W2_FRACTAL_SIZE; ++nf) {
            for (uint32_t mf = 0; mf < mAligned / W2_FRACTAL_SIZE; ++mf) {
                const uint32_t row = mf * W2_FRACTAL_SIZE;
                const uint32_t ubOffset =
                    (nf * (mAligned / W2_FRACTAL_SIZE) + mf) * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE;
                DataCopyParams toGm;
                toGm.blockCount = MinU<uint32_t>(W2_FRACTAL_SIZE, (uint32_t)T_ - row);
                toGm.blockLen = 1;
                toGm.srcStride = 0;
                toGm.dstStride = (N_ - W2_FRACTAL_SIZE) * sizeof(half) / 32;
                DataCopy(yGm_[static_cast<int64_t>(row) * N_ + n0 + nf * W2_FRACTAL_SIZE],
                         converted[ubOffset], toGm);
            }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        SetFlag<HardEvent::V_M>(EVENT_ID7);
        WaitFlag<HardEvent::V_M>(EVENT_ID7);
    }

    __aicore__ inline void AllocBuffers()
    {
        uint32_t off = W2_MMAD_EPILOGUE_UB_BYTES;
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
        nzTileUB_ = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + decodedTileCount_ * sizeof(half), 512);
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
        if (!nzPacked_) {
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
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void DecodeTile(int64_t codeOffset)
    {
        // The resident-L1 path reuses cU8_ promptly after a bulk DMA.
        // Complete the previous vector read before that DMA reuses it.
        if (useL1_) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        // Copy one packed row at a time. A strided 2-D GM-to-UB DataCopy looks
        // attractive here, but dav_m200 rejects that descriptor at runtime
        // with an MTE "burst num" exception even when every row and stride is
        // 32-byte aligned. The scalar overload is hardware-proven on 310P and
        // still batches all 128 logical K values for the vector decode below.
        if (nzPacked_) {
            DataCopy(cU8_, codesGm_[codeOffset], (int32_t)packedTileCount_);
        } else {
            for (uint32_t row = 0; row < W2_FRACTAL_SIZE; ++row) {
                DataCopy(cU8_[row * packedTileCols_],
                         codesGm_[codeOffset + static_cast<int64_t>(row) * packedK_],
                         static_cast<int32_t>(packedTileCols_));
            }
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

        if (!nzPacked_) {
            Gather(fractalRowsUB_, signedHalfUB_, gatherOffsetsUB_, (uint32_t)0,
                   (uint32_t)decodedTileCount_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DequantTileToNz(uint32_t n0, uint32_t nActual, bool toL1 = false)
    {
        for (uint32_t scaleRowBase = 0; scaleRowBase < nActual; scaleRowBase += W2_BLOCK_SIZE) {
            const int64_t scaleRow = (static_cast<int64_t>(n0) + scaleRowBase) / W2_BLOCK_SIZE;
            DataCopy(scaleUB_, scaleGm_[scaleRow * kbCount_], (int32_t)kbCount_);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);

            for (uint32_t rowInScale = 0; rowInScale < W2_BLOCK_SIZE; rowInScale += W2_FRACTAL_SIZE) {
                const uint32_t rowBase = scaleRowBase + rowInScale;
                for (int64_t k0 = 0; k0 < K_; k0 += W2_TILE_K) {
                    const int64_t codeOffset = nzPacked_
                        ? ((static_cast<int64_t>(n0) + rowBase) / W2_FRACTAL_SIZE *
                               (K_ / W2_TILE_K) + k0 / W2_TILE_K) * packedTileCount_
                        : (static_cast<int64_t>(n0) + rowBase) * packedK_ + k0 / codesPerByte_;
                    DecodeTile(codeOffset);

                    // W is [N,K], while Cube consumes B=W^T [K,N]. Transpose
                    // each 16x16 fragment into NZ order first.
                    const int64_t nFractal = rowBase / W2_FRACTAL_SIZE;
                    const int64_t nzColumnBlockStride = K_ * W2_FRACTAL_SIZE;
                    const int64_t nzBase = coreNzBase_ + nFractal * nzColumnBlockStride;
                    if (!nzPacked_) {
                        for (int64_t kFractal = 0; kFractal < W2_K_FRACTALS_PER_TILE; ++kFractal) {
                            const int64_t offset = kFractal * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE;
                            AscendC::Transpose(nzTileUB_[offset], fractalRowsUB_[offset]);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    // One [32,32] block scale covers two adjacent 16x16 NZ
                    // fragments. Multiplying both at once halves vector Muls
                    // and its barriers without changing any FP16 products.
                    for (int64_t kFractal = 0; kFractal < W2_K_FRACTALS_PER_TILE; kFractal += 2) {
                        const int64_t localFractalOffset =
                            kFractal * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE;
                        const int64_t scaleIndex = k0 / W2_BLOCK_SIZE + kFractal / 2;
                        const half scale = static_cast<half>(scaleUB_.GetValue(scaleIndex));
                        auto nzFractal = nzPacked_ ? signedHalfUB_[localFractalOffset]
                                                   : nzTileUB_[localFractalOffset];
                        Muls(nzFractal, nzFractal, scale,
                             2 * W2_FRACTAL_SIZE * W2_FRACTAL_SIZE);
                        PipeBarrier<PIPE_V>();
                    }
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                    if (toL1) {
                        auto weightL1 = resource.l1Buf.template GetBufferByByte<half>(0);
                        DataCopy(weightL1[nFractal * nzColumnBlockStride +
                                          k0 * W2_FRACTAL_SIZE], signedHalfUB_,
                                 (int32_t)decodedTileCount_);
                    } else if (nzPacked_) {
                        DataCopy(wdqNzGm_[nzBase + k0 * W2_FRACTAL_SIZE],
                                 signedHalfUB_, decodedTileCount_);
                    } else {
                        DataCopy(wdqNzGm_[nzBase + k0 * W2_FRACTAL_SIZE],
                                 nzTileUB_, decodedTileCount_);
                    }
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
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
    LocalTensor<half> nzTileUB_;

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
    bool nzPacked_;
    bool useL1_;
};

}  // namespace NsW2
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_H
