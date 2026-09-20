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
 * \file gdn_gating_v310.h
 * \brief
 *
 *   g[t, h]       = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h], beta)
 *   betaOut[t, h] = fp16(sigmoid(b[t, h]))
 *
 * softplus uses the numerically stable identity
 *     softplus_beta(x) = (1/beta) * ( max(bx, 0) + log1p(exp(-|bx|)) ),  bx = beta*x
 * rather than PyTorch's guarded log1p(exp(bx)) / linear-tail branch. The two
 * agree to <= 2e-9 at the default threshold of 20 (far below fp16 resolution
 * downstream) and this form needs no CompareScalar/Select, using only vector
 * primitives that are certain to exist on 310P.
 */
#ifndef GDN_GATING_V310_H
#define GDN_GATING_V310_H

#include "kernel_operator.h"
#include "gdn_gating_v310_tiling_data.h"

namespace NsGdnGating {

using namespace AscendC;

template <typename T>
class GdnGatingV310 {
public:
    __aicore__ inline GdnGatingV310() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR negExpALogTiled, GM_ADDR dtBiasTiled, GM_ADDR g,
                                GM_ADDR betaOut, const GdnGatingTilingData *tilingData, TPipe *pipe)
    {
        H_ = tilingData->numHeads;
        tileCount_ = tilingData->tileCount;
        tilesPerCore_ = tilingData->tilesPerCore;
        beta_ = tilingData->beta;
        invBeta_ = tilingData->invBeta;
        tileElems_ = tilingData->tileRows * H_;
        totalElems_ = tilingData->numRows * H_;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b));
        negExpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(negExpALogTiled));
        dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBiasTiled));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        betaOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(betaOut));

        pipe_ = pipe;
        pipe_->InitBuffer(halfInBuf_, tileElems_ * sizeof(T));
        pipe_->InitBuffer(xBuf_, tileElems_ * sizeof(float));
        pipe_->InitBuffer(t1Buf_, tileElems_ * sizeof(float));
        pipe_->InitBuffer(t2Buf_, tileElems_ * sizeof(float));
        pipe_->InitBuffer(negExpBuf_, tileElems_ * sizeof(float));
        pipe_->InitBuffer(dtBiasBuf_, tileElems_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> halfIn = halfInBuf_.template Get<T>();
        LocalTensor<float> x = xBuf_.template Get<float>();
        LocalTensor<float> t1 = t1Buf_.template Get<float>();
        LocalTensor<float> t2 = t2Buf_.template Get<float>();
        LocalTensor<float> negExp = negExpBuf_.template Get<float>();
        LocalTensor<float> dtBias = dtBiasBuf_.template Get<float>();

        // Weight-only constants, already broadcast to [TILE_ROWS, H] by the
        // caller: load once per core, reuse for every tile.
        DataCopy(negExp, negExpGm_, tileElems_);
        DataCopy(dtBias, dtBiasGm_, tileElems_);
        PipeBarrier<PIPE_ALL>();

        const int64_t firstTile = GetBlockIdx() * tilesPerCore_;
        for (int64_t i = 0; i < tilesPerCore_; ++i) {
            const int64_t tile = firstTile + i;
            if (tile >= tileCount_) {
                break;
            }
            const int64_t off = tile * tileElems_;
            // The final input tile may be short. Vector ops use the true
            // element count, while outputs are allocated to a TILE_ROWS
            // multiple so every UB->GM transfer remains fully aligned.
            const int64_t elems = (off + tileElems_ <= totalElems_) ? tileElems_ : (totalElems_ - off);

            // g = -exp(A_log) * softplus(a + dt_bias)
            CopyInT(halfIn, aGm_, off, elems);
            PipeBarrier<PIPE_ALL>();
            Cast(x, halfIn, RoundMode::CAST_NONE, elems);
            PipeBarrier<PIPE_V>();
            Add(x, x, dtBias, elems);
            PipeBarrier<PIPE_V>();
            if (beta_ != 1.0f) {
                Muls(x, x, beta_, elems);
                PipeBarrier<PIPE_V>();
            }
            Abs(t1, x, elems);
            PipeBarrier<PIPE_V>();
            Muls(t1, t1, -1.0f, elems);
            PipeBarrier<PIPE_V>();
            Exp(t1, t1, elems);
            PipeBarrier<PIPE_V>();
            Adds(t1, t1, 1.0f, elems);
            PipeBarrier<PIPE_V>();
            Ln(t1, t1, elems);              // log1p(exp(-|bx|))
            PipeBarrier<PIPE_V>();
            Maxs(t2, x, 0.0f, elems);       // max(bx, 0)
            PipeBarrier<PIPE_V>();
            Add(t1, t1, t2, elems);
            PipeBarrier<PIPE_V>();
            if (beta_ != 1.0f) {
                Muls(t1, t1, invBeta_, elems);
                PipeBarrier<PIPE_V>();
            }
            Mul(t1, t1, negExp, elems);
            PipeBarrier<PIPE_ALL>();
            CopyOutF(gGm_, off, t1);

            // betaOut = fp16(sigmoid(b)), sigmoid in fp32 to match the
            // reference implementation this operator replaces.
            CopyInT(halfIn, bGm_, off, elems);
            PipeBarrier<PIPE_ALL>();
            Cast(x, halfIn, RoundMode::CAST_NONE, elems);
            PipeBarrier<PIPE_V>();
            Muls(x, x, -1.0f, elems);
            PipeBarrier<PIPE_V>();
            Exp(x, x, elems);
            PipeBarrier<PIPE_V>();
            Adds(x, x, 1.0f, elems);
            PipeBarrier<PIPE_V>();
            Reciprocal(x, x, elems);
            PipeBarrier<PIPE_V>();
            Cast(halfIn, x, RoundMode::CAST_NONE, elems);
            PipeBarrier<PIPE_ALL>();
            CopyOutT(betaOutGm_, off, halfIn);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void CopyInT(const LocalTensor<T> &dst, const GlobalTensor<T> &src, int64_t off, int64_t n)
    {
        if (n == tileElems_) {
            DataCopy(dst, src[off], tileElems_);
            return;
        }
        DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, src[off], p, pad);
    }

    __aicore__ inline void CopyOutT(const GlobalTensor<T> &dst, int64_t off, const LocalTensor<T> &src)
    {
        DataCopy(dst[off], src, tileElems_);
    }

    __aicore__ inline void CopyOutF(const GlobalTensor<float> &dst, int64_t off, const LocalTensor<float> &src)
    {
        DataCopy(dst[off], src, tileElems_);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> halfInBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> t1Buf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    TBuf<TPosition::VECCALC> negExpBuf_;
    TBuf<TPosition::VECCALC> dtBiasBuf_;
    GlobalTensor<T> aGm_;
    GlobalTensor<T> bGm_;
    GlobalTensor<float> negExpGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<T> betaOutGm_;

    int64_t H_ = 0;
    int64_t tileElems_ = 0;
    int64_t totalElems_ = 0;
    int64_t tileCount_ = 0;
    int64_t tilesPerCore_ = 0;
    float beta_ = 1.0f;
    float invBeta_ = 1.0f;
};

}  // namespace NsGdnGating
#endif  // GDN_GATING_V310_H
