// SPDX-License-Identifier: Apache-2.0
// Isolated fixed-shape row batching experiment. Not a serving binding.
#include "kernel_operator.h"
using namespace AscendC;
constexpr unsigned K = 128;
constexpr unsigned ROWS = 64;
constexpr unsigned HEADS = 16;
constexpr unsigned BANK = K * ROWS;
constexpr unsigned ARENA_BYTES = 104 * 1024;
template<class T> class ScoreProbe {
public:
    __aicore__ inline void Run(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR aqk, GM_ADDR akk) {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g));
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aqk));
        akk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(akk));
        pipe_.InitBuffer(arena_, ARENA_BYTES);
        for (unsigned head = GetBlockIdx(); head < HEADS; head += GetBlockNum()) {
            Compute(head);
        }
    }
private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<T> q_, k_, g_;
    GlobalTensor<float> aqk_, akk_;
    __aicore__ inline void Compute(unsigned head) {
        auto halfArena = arena_.Get<T>();
        auto keys = halfArena;
        auto gates = halfArena[BANK];
        auto gate = halfArena[2 * BANK];
        auto product = halfArena[3 * BANK];
        auto floats = arena_.Get<float>();
        auto productFp32 = floats[64 * 1024 / sizeof(float)];
        auto low = floats[96 * 1024 / sizeof(float)];
        auto high = floats[98 * 1024 / sizeof(float)];
        auto rowVector = halfArena[100 * 1024 / sizeof(T)];
        auto gateRow = rowVector[K];
        auto score = floats[(100 * 1024 + 2 * K * sizeof(T)) / sizeof(float)];
        DataCopy(keys, k_[head * BANK], BANK);
        DataCopy(gates, g_[head * BANK], BANK);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        // Keep K*K and Q*K separate as in the qualified implementation.
        for (unsigned pass = 0; pass < 2; ++pass) {
            for (unsigned row = 0; row < ROWS; ++row) {
                const unsigned columns = row + 1;
                const unsigned count = columns * K;
                Duplicate(score, 0.0f, ROWS);
                Duplicate(low, 12345.0f, ROWS * 8);
                Duplicate(high, 12345.0f, ROWS * 8);
                PipeBarrier<PIPE_V>();
                if (pass == 0) {
                    DataCopy(rowVector, k_[head * BANK + row * K], K);
                } else {
                    DataCopy(rowVector, q_[head * BANK + row * K], K);
                }
                DataCopy(gateRow, g_[head * BANK + row * K], K);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Sub(gate, gateRow, gates, K, static_cast<uint8_t>(columns), {1, 1, 1, 8, 0, 8});
                PipeBarrier<PIPE_V>();
                Muls(gate, gate, static_cast<T>(0.69314718055994530942f), count);
                PipeBarrier<PIPE_V>();
                Mins(gate, gate, static_cast<T>(11.089866f), count);
                PipeBarrier<PIPE_V>();
                Maxs(gate, gate, static_cast<T>(-17.0f), count);
                PipeBarrier<PIPE_V>();
                Exp(gate, gate, count);
                PipeBarrier<PIPE_V>();
                Mul(product, keys, gate, count);
                PipeBarrier<PIPE_V>();
                Mul(product, rowVector, product, K, static_cast<uint8_t>(columns), {1, 1, 1, 8, 0, 8});
                PipeBarrier<PIPE_V>();
                Cast(productFp32, product, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                WholeReduceSum(low, productFp32, 64, static_cast<uint8_t>(columns), 1, 1, 16);
                WholeReduceSum(high, productFp32[64], 64, static_cast<uint8_t>(columns), 1, 1, 16);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                for (unsigned col = 0; col < columns; ++col) {
                    float sum = 0.0f;
                    sum += low.GetValue(col * 8);
                    sum += high.GetValue(col * 8);
                    score.SetValue(col, sum);
                }
                SetFlag<HardEvent::S_V>(0);
                WaitFlag<HardEvent::S_V>(0);
                SetFlag<HardEvent::V_MTE3>(1);
                WaitFlag<HardEvent::V_MTE3>(1);
                if (pass == 0) {
                    DataCopy(akk_[head * ROWS * ROWS + row * ROWS], score, ROWS);
                } else {
                    DataCopy(aqk_[head * ROWS * ROWS + row * ROWS], score, ROWS);
                }
                SetFlag<HardEvent::MTE3_MTE2>(3);
                WaitFlag<HardEvent::MTE3_MTE2>(3);
                SetFlag<HardEvent::MTE3_V>(2);
                WaitFlag<HardEvent::MTE3_V>(2);
            }
        }
    }
};
extern "C" __global__ __aicore__ void glm_kda_score_probe_v1(
    GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR aqk, GM_ADDR akk) {
    InitSocState();
    ScoreProbe<half> probe;
    probe.Run(q, k, g, aqk, akk);
}
