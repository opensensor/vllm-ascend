#include "kernel_operator.h"
#include "sidecard_kda_v310_tiling_data.h"

namespace {
using namespace AscendC;
constexpr uint32_t DIM = 128;
constexpr uint32_t VALUE_GROUP = 16;
constexpr uint32_t GROUPS_PER_HEAD = DIM / VALUE_GROUP;
constexpr uint32_t FLOAT_ELEMENTS = 960;
constexpr uint32_t HALF_ELEMENTS = 160;
constexpr float EPSILON = 1e-6f;
constexpr float QUERY_SCALE = 0.08838834764831845f;  // 128**-0.5

class SidecardKda {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate,
                                GM_ADDR beta, GM_ADDR state, GM_ADDR output, GM_ADDR stateOut,
                                const SidecardKdaV310TilingData& data, TPipe& pipe)
    {
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(query));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(key));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(value));
        gate_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gate));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        state_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        stateOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stateOut));
        heads_ = data.heads;
        blocks_ = data.blocks;
        pipe.InitBuffer(floatBuffer_, FLOAT_ELEMENTS * sizeof(float));
        pipe.InitBuffer(halfBuffer_, HALF_ELEMENTS * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        auto local = floatBuffer_.Get<float>();
        auto halfLocal = halfBuffer_.Get<half>();
        auto q = local[0];
        auto k = local[128];
        auto decay = local[256];
        auto row = local[384];
        auto temporary = local[512];
        auto betas = local[800];
        auto values = local[928];
        auto outputs = local[944];
        DataCopy(betas, beta_, heads_);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t group = GetBlockIdx(); group < heads_ * GROUPS_PER_HEAD; group += blocks_) {
            const uint32_t head = group / GROUPS_PER_HEAD;
            const uint32_t firstValue = (group % GROUPS_PER_HEAD) * VALUE_GROUP;
            const uint32_t headOffset = head * DIM;
            DataCopy(halfLocal, query_[headOffset], DIM);
            PipeBarrier<PIPE_ALL>();
            Cast(q, halfLocal, RoundMode::CAST_NONE, DIM);
            PipeBarrier<PIPE_ALL>();
            DataCopy(halfLocal, key_[headOffset], DIM);
            PipeBarrier<PIPE_ALL>();
            Cast(k, halfLocal, RoundMode::CAST_NONE, DIM);
            DataCopy(decay, gate_[headOffset], DIM);
            DataCopy(halfLocal[128], value_[headOffset + firstValue], VALUE_GROUP);
            PipeBarrier<PIPE_ALL>();
            Cast(values, halfLocal[128], RoundMode::CAST_NONE, VALUE_GROUP);
            Exp(decay, decay, DIM);
            PipeBarrier<PIPE_ALL>();
            Normalize(q, temporary, local);
            Muls(q, q, QUERY_SCALE, DIM);
            PipeBarrier<PIPE_ALL>();
            Normalize(k, temporary, local);
            const float beta = betas.GetValue(head);
            for (uint32_t index = 0; index < VALUE_GROUP; ++index) {
                const uint32_t offset = (head * DIM + firstValue + index) * DIM;
                DataCopy(row, state_[offset], DIM);
                PipeBarrier<PIPE_ALL>();
                // Keep each FP32 multiply/add as a separate vector operation:
                // decay -> prediction -> innovation -> update -> output.
                Mul(row, row, decay, DIM);
                PipeBarrier<PIPE_ALL>();
                Mul(temporary, row, k, DIM);
                PipeBarrier<PIPE_ALL>();
                const float prediction = Sum(temporary, local);
                const float innovation = (values.GetValue(index) - prediction) * beta;
                Muls(temporary, k, innovation, DIM);
                PipeBarrier<PIPE_ALL>();
                Add(row, row, temporary, DIM);
                PipeBarrier<PIPE_ALL>();
                DataCopy(stateOut_[offset], row, DIM);
                Mul(temporary, row, q, DIM);
                PipeBarrier<PIPE_ALL>();
                outputs.SetValue(index, Sum(temporary, local));
                PipeBarrier<PIPE_ALL>();
            }
            Cast(halfLocal[144], outputs, RoundMode::CAST_RINT, VALUE_GROUP);
            PipeBarrier<PIPE_ALL>();
            DataCopy(output_[headOffset + firstValue], halfLocal[144], VALUE_GROUP);
            // Includes output and carry completion before buffer/group reuse.
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline float Sum(LocalTensor<float> input, LocalTensor<float> local)
    {
        ReduceSum(local[768], input, local[640], DIM);
        PipeBarrier<PIPE_ALL>();
        return local.GetValue(768);
    }
    __aicore__ inline void Normalize(LocalTensor<float> input, LocalTensor<float> temporary,
                                     LocalTensor<float> local)
    {
        Mul(temporary, input, input, DIM);
        PipeBarrier<PIPE_ALL>();
        const float squareSum = Sum(temporary, local);
        Duplicate(local[768], squareSum + EPSILON, 8);
        PipeBarrier<PIPE_ALL>();
        Rsqrt(local[768], local[768], 8);
        PipeBarrier<PIPE_ALL>();
        Muls(input, input, local.GetValue(768), DIM);
        PipeBarrier<PIPE_ALL>();
    }
    TBuf<TPosition::VECCALC> floatBuffer_;
    TBuf<TPosition::VECCALC> halfBuffer_;
    GlobalTensor<half> query_, key_, value_, output_;
    GlobalTensor<float> gate_, beta_, state_, stateOut_;
    uint32_t heads_ = 0;
    uint32_t blocks_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void sidecard_kda_v310(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta, GM_ADDR state,
    GM_ADDR output, GM_ADDR stateOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SidecardKdaV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(SidecardKdaV310TilingData, data, tiling);
    AscendC::TPipe pipe;
    SidecardKda operation;
    operation.Init(query, key, value, gate, beta, state, output, stateOut, data, pipe);
    operation.Process();
}
