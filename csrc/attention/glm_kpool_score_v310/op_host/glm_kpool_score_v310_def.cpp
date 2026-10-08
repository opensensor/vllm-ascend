// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class GlmKpoolScoreV310 : public OpDef {
 public:
  explicit GlmKpoolScoreV310(const char* name) : OpDef(name) {
    this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    for (const auto name : {"table", "boundaries", "positions"}) {
      this->Input(name).ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const auto name : {"pools", "blocks", "block_rows", "block_stride", "row_stride", "cache_offset"}) {
      this->Attr(name).AttrType(REQUIRED).Int();
    }
    this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false).DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(GlmKpoolScoreV310);
}
