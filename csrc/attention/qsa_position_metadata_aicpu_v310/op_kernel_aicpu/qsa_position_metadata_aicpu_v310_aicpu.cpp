// SPDX-License-Identifier: Apache-2.0
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "log.h"
#include "status.h"

#include "../position_metadata.h"

namespace aicpu {
class QsaPositionMetadataAicpuV310Kernel : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext& context) override {
        auto* positions = context.Input(0);
        auto* metadata = context.Input(1);
        const auto* ratio = context.GetAttr("compress_ratio");
        const auto* capacity = context.GetAttr("capacity");
        const auto* selected_width = context.GetAttr("selected_width");
        if (positions == nullptr || metadata == nullptr || positions->GetData() == nullptr ||
            metadata->GetData() == nullptr || positions->GetTensorShape() == nullptr ||
            metadata->GetTensorShape() == nullptr || ratio == nullptr || capacity == nullptr ||
            selected_width == nullptr) {
            KERNEL_LOG_ERROR("QSA position metadata missing tensor or attribute");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        const int64_t rows = positions->GetTensorShape()->NumElements();
        if (rows < 0 || rows > INT64_MAX / 4 || metadata->GetTensorShape()->NumElements() != rows * 4 ||
            !qsa_position_metadata::Compute(
                static_cast<const int32_t*>(positions->GetData()), rows,
                ratio->GetInt(), capacity->GetInt(), selected_width->GetInt(),
                static_cast<int32_t*>(metadata->GetData()))) {
            KERNEL_LOG_ERROR("QSA position metadata rejected geometry or positions");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return KERNEL_STATUS_OK;
    }
};

namespace {
static const char* kernel_type = "QsaPositionMetadataAicpuV310";
REGISTER_CPU_KERNEL(kernel_type, QsaPositionMetadataAicpuV310Kernel);
}  // namespace
}  // namespace aicpu
