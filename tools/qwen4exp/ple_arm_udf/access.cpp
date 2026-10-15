// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <memory>
#include <vector>
#include "flow_func/meta_flow_func.h"
#include "../ple_device_access.h"

namespace FlowFunc {
class PleAccess : public MetaFlowFunc {
public:
    int32_t Init() override { return FLOW_FUNC_SUCCESS; }
    int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>>& inputs) override {
        if (inputs.size() != 1 || !inputs[0] || inputs[0]->GetRetCode() != 0 ||
            inputs[0]->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA) return FLOW_FUNC_FAILED;
        constexpr int64_t kReceiptBytes = 16384;
        auto message = context_->AllocTensorMsg({kReceiptBytes}, TensorDataType::DT_UINT8);
        if (!message || !message->GetTensor()) return FLOW_FUNC_FAILED;
        auto tensor = message->GetTensor();
        if (tensor->GetDataSize() != kReceiptBytes || !tensor->GetData()) return FLOW_FUNC_FAILED;
        char* bytes = static_cast<char*>(tensor->GetData());
        std::memset(bytes, 0, kReceiptBytes);
        PleProbeText receipt(bytes, kReceiptBytes);
        ple_device_access_report(receipt);
        if (receipt.truncated()) return FLOW_FUNC_FAILED;
        return context_->SetOutput(0, message);
    }
};
REGISTER_FLOW_FUNC("PleAccess", PleAccess);
}  // namespace FlowFunc
