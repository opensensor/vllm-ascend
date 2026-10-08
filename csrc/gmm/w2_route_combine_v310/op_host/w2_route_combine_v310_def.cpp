// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class W2RouteCombineV310 : public OpDef {
 public:
  explicit W2RouteCombineV310(const char* name) : OpDef(name) {
    this->Input("routed").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("inverse_order").ParamType(REQUIRED).DataType({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("route_weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("group_ends").ParamType(REQUIRED).DataType({ge::DT_INT64})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND}).AutoContiguous();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(W2RouteCombineV310);
}  // namespace ops
