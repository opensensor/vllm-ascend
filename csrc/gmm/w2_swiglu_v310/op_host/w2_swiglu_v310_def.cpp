// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class W2SwigluV310 : public OpDef {
 public:
  explicit W2SwigluV310(const char* name) : OpDef(name) {
    this->Input("gate_up").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(W2SwigluV310);
}  // namespace ops
