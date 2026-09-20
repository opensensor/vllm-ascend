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
 * \brief
 *
 * v2: Cube (AIC) implementation on 310P via the arch20 catlass block MMAD.
 *   out[t, n] = sum_k x[t, k] * (codes[n, k] * block_scale[n/32, k/32])
 *
 * MIX_AIC unified-core kernel. N is split into 128-column blocks distributed
 * across cube cores. For each block a core:
 *   1. VECTOR: dequant codes[nBlk, K] int8 -> fp16 into a GM weight workspace
 *      (Wdq), applying the per-[32,32] block scale (cast int8->fp32, Muls by
 *      block_scale, cast fp32->fp16).
 *   2. PIPE_ALL fence (commit MTE3 writes before the cube's MTE2 reads).
 *   3. CUBE: catlass BlockMmadTla computes yF[T, nBlk] (fp32) = x @ Wdq^T,
 *      where Wdq[nBlk, K] rowmajor is viewed ColumnMajor [K, nBlk] == W^T.
 *      Because the scale is applied to the weight before the MMAD, plain
 *      full-K accumulation is exact.
 *   4. VECTOR: cast yF[T, nBlk] fp32 -> y[T, nBlk] fp16.
 * Each core reads back only the Wdq slice it produced, so no cross-core
 * barrier is required (only within-core PIPE_ALL fences), matching the
 * proven chunk_fwd_o unified-core vec->cube handoff pattern.
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
// The op framework auto-generates and force-includes the kernel-side tiling
// data class (class W2BlockedDequantMatmulTilingData) from the registered
// BEGIN_TILING_DATA_DEF, so we must NOT include the hand-written plain mirror
// here (that would be a redefinition). Same mechanism chunk_fwd_o relies on.

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
        kbCount_ = K_ / W2_BLK;
        mAligned_ = AlignUpU<int64_t>(T_, 16);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        codesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(codes));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(blockScale));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));

        // user workspace layout: [ Wdq (N*K half) ][ yF (mAligned*N float) ]
        wdqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(user));
        yfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(user + N_ * K_ * sizeof(half)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();

        BlockMmad blockMmad(resource);

        auto aLayout = tla::MakeLayout<half, layout::RowMajor>(static_cast<uint32_t>(T_), static_cast<uint32_t>(K_));
        auto bLayout = tla::MakeLayout<half, layout::ColumnMajor>(static_cast<uint32_t>(K_), static_cast<uint32_t>(N_));
        auto cLayout = tla::MakeLayout<float, layout::RowMajor>(static_cast<uint32_t>(mAligned_), static_cast<uint32_t>(N_));
        auto tensorA = tla::MakeTensor(xGm_, aLayout, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(wdqGm_, bLayout, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(yfGm_, cLayout, Arch::PositionGM{});

        const uint32_t nBlocks = CeilDivU<uint32_t>(static_cast<uint32_t>(N_), W2_TILE_N);

        for (uint32_t nb = coreId; nb < nBlocks; nb += coreNum) {
            uint32_t n0 = nb * W2_TILE_N;
            uint32_t nActual = MinU<uint32_t>(W2_TILE_N, static_cast<uint32_t>(N_) - n0);

            // 1) VECTOR: dequant this N-slice of the weight into Wdq (fp16, GM).
            DequantBlock(n0, nActual);
            AscendC::PipeBarrier<PIPE_ALL>();

            // 2) CUBE: yF[0:mAligned, n0:n0+nActual] = x @ Wdq[n0..]^T
            GemmCoord shape{static_cast<uint32_t>(T_), nActual, static_cast<uint32_t>(K_)};
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

            // 3) VECTOR: cast yF (fp32) -> y (fp16)
            CastOut(n0, nActual);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    // Dequant codes[n0:n0+nActual, :] -> wdqGm[n0.., :] (fp16), block scale applied.
    // Fused fp16 path: rows are grouped in 32s that share one block-scale row; a
    // per-column half rowScale[K] is built once per group (each of the K/32 block
    // scales expanded across its 32 columns), then each row is cast int8->half and
    // multiplied by rowScale in a single vector op. Double-buffered codes staging
    // with targeted event sync keeps MTE2/V/MTE3 overlapped.
    __aicore__ inline void DequantBlock(uint32_t n0, uint32_t nActual)
    {
        uint32_t off = 0;
        LocalTensor<int8_t> codesUB0 = resource.ubBuf.template GetBufferByByte<int8_t>(off);
        off = AlignUpU<uint32_t>(off + K_ * sizeof(int8_t), 512);
        LocalTensor<int8_t> codesUB1 = resource.ubBuf.template GetBufferByByte<int8_t>(off);
        off = AlignUpU<uint32_t>(off + K_ * sizeof(int8_t), 512);
        LocalTensor<half> wHalf0 = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + K_ * sizeof(half), 512);
        LocalTensor<half> wHalf1 = resource.ubBuf.template GetBufferByByte<half>(off);
        off = AlignUpU<uint32_t>(off + K_ * sizeof(half), 512);
        LocalTensor<float> scaleUB = resource.ubBuf.template GetBufferByByte<float>(off);
        off = AlignUpU<uint32_t>(off + kbCount_ * sizeof(float), 512);
        LocalTensor<half> rowScale = resource.ubBuf.template GetBufferByByte<half>(off);

        LocalTensor<int8_t> codesUB[2] = {codesUB0, codesUB1};
        LocalTensor<half> wHalf[2] = {wHalf0, wHalf1};

        // Prime WAR flags so the first reload of each buffer proceeds.
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));

        for (uint32_t rBase = 0; rBase < nActual; rBase += W2_BLK) {
            uint32_t rows = MinU<uint32_t>(static_cast<uint32_t>(W2_BLK), nActual - rBase);
            int64_t nb = (static_cast<int64_t>(n0) + rBase) / W2_BLK;  // shared scale row

            // Load this group's block-scale row and build the half rowScale[K].
            DataCopy(scaleUB, scaleGm_[nb * kbCount_], kbCount_);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            for (int64_t kb = 0; kb < kbCount_; ++kb) {
                Duplicate(rowScale[kb * W2_BLK], static_cast<half>(scaleUB.GetValue(kb)),
                          static_cast<int32_t>(W2_BLK));
            }
            PipeBarrier<PIPE_V>();  // rowScale ready before the Muls read it

            for (uint32_t rr = 0; rr < rows; ++rr) {
                int64_t n = static_cast<int64_t>(n0) + rBase + rr;
                uint32_t b = rr & 1u;
                // WAR: previous store from this buffer must finish before reload.
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(b));
                DataCopy(codesUB[b], codesGm_[n * K_], K_);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(b));
                WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(b));
                Cast(wHalf[b], codesUB[b], RoundMode::CAST_NONE, K_);
                PipeBarrier<PIPE_V>();
                Mul(wHalf[b], wHalf[b], rowScale, K_);
                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(b));
                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(b));
                DataCopy(wdqGm_[n * K_], wHalf[b], K_);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(b));
            }
        }
        // Drain outstanding WAR flags for both buffers.
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    }

    // Cast yF[0:T, n0:n0+nActual] (fp32) -> y[0:T, n0:n0+nActual] (fp16).
    __aicore__ inline void CastOut(uint32_t n0, uint32_t nActual)
    {
        LocalTensor<float> fUB = resource.ubBuf.template GetBufferByByte<float>(0);
        LocalTensor<half> hUB = resource.ubBuf.template GetBufferByByte<half>(
            AlignUpU<uint32_t>(nActual * sizeof(float), 512));
        for (int64_t t = 0; t < T_; ++t) {
            DataCopy(fUB, yfGm_[t * N_ + n0], nActual);
            PipeBarrier<PIPE_ALL>();  // MTE2 -> V
            Cast(hUB, fUB, RoundMode::CAST_NONE, nActual);
            PipeBarrier<PIPE_ALL>();  // V -> MTE3
            DataCopy(yGm_[t * N_ + n0], hUB, nActual);
            PipeBarrier<PIPE_ALL>();  // MTE3 done before buffers reused next row
        }
    }

    Arch::Resource<ArchTag> resource;

    GlobalTensor<half> xGm_;
    GlobalTensor<int8_t> codesGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<half> yGm_;
    GlobalTensor<half> wdqGm_;
    GlobalTensor<float> yfGm_;

    int64_t T_;
    int64_t N_;
    int64_t K_;
    int64_t kbCount_;
    int64_t mAligned_;
};

}  // namespace NsW2
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_H
