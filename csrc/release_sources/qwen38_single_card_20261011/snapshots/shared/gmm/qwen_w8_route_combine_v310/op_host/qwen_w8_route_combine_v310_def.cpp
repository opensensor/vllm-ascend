// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_def_registry.h"

namespace ops {
class QwenW8RouteCombineV310 : public OpDef {
 public:
  explicit QwenW8RouteCombineV310(const char* name) : OpDef(name) {
    this->Input("sortedRoutes")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("inverseOrder")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("routeWeights")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("localRouteMask")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(QwenW8RouteCombineV310);
}  // namespace ops
