// SPDX-License-Identifier: Apache-2.0
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "log.h"
#include "status.h"

#include "../exact_topk.h"

#include <algorithm>
#include <atomic>

namespace aicpu {
constexpr int64_t QSA_TOPK_TARGET_SHARDS = 8;

class QsaExactTopkAicpuV310Kernel : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext& context) override {
        auto* scores = context.Input(0);
        auto* indices = context.Input(1);
        if (scores == nullptr || indices == nullptr || scores->GetData() == nullptr ||
            indices->GetData() == nullptr || scores->GetTensorShape() == nullptr ||
            indices->GetTensorShape() == nullptr) {
            KERNEL_LOG_ERROR("QSA selector missing tensor or data");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        const auto score_shape = scores->GetTensorShape();
        const auto index_shape = indices->GetTensorShape();
        const auto* topk_attr = context.GetAttr("top_k");
        if (topk_attr == nullptr) {
            KERNEL_LOG_ERROR("QSA selector missing top_k attribute");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        qsa_exact_topk::SelectionShape logical_shape{};
        if (!qsa_exact_topk::ResolveSelectionShape(
                score_shape->NumElements(), index_shape->NumElements(),
                topk_attr->GetInt(), &logical_shape)) {
            KERNEL_LOG_ERROR("QSA selector incompatible score and index buffers");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        const auto queries = logical_shape.queries;
        const auto groups = logical_shape.groups;
        const auto topk = logical_shape.topk;
        const auto* score_data = static_cast<const float*>(scores->GetData());
        auto* index_data = static_cast<int32_t*>(indices->GetData());
        if (queries < QSA_TOPK_TARGET_SHARDS) {
            if (!qsa_exact_topk::SelectRows(score_data, queries, groups, topk, index_data)) {
                KERNEL_LOG_ERROR("QSA selector rejected scores or dimensions");
                return KERNEL_STATUS_PARAM_INVALID;
            }
            return KERNEL_STATUS_OK;
        }
        std::atomic<bool> valid{true};
        const auto shard = [&](int64_t first, int64_t last) {
            if (!qsa_exact_topk::SelectRows(score_data + first * groups, last - first,
                                            groups, topk, index_data + first * topk)) {
                valid.store(false, std::memory_order_relaxed);
            }
        };
        // ParallelFor uses at most the workers available to the device. Eight
        // row shards improved the host feasibility gate; device timing decides
        // whether this grain is useful on the 310P AI CPU.
        const auto grain = std::max<int64_t>(1, queries / QSA_TOPK_TARGET_SHARDS);
        const auto status = CpuKernelUtils::ParallelFor(context, queries, grain, shard);
        if (status != KERNEL_STATUS_OK || !valid.load(std::memory_order_relaxed)) {
            KERNEL_LOG_ERROR("QSA selector parallel row selection failed");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return KERNEL_STATUS_OK;
    }
};

namespace {
static const char* kernel_type = "QsaExactTopkAicpuV310";
REGISTER_CPU_KERNEL(kernel_type, QsaExactTopkAicpuV310Kernel);
}  // namespace
}  // namespace aicpu
