#include "register/op_def_registry.h"
namespace ops {
class SidecardMhcPreV310 : public OpDef {
 public:
  explicit SidecardMhcPreV310(const char* name) : OpDef(name) {
    this->Input("residual").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("fn").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("base").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("normWeight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Output("post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("hidden").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Attr("normEps").AttrType(REQUIRED).Float(1e-6f);
    this->Attr("hcEps").AttrType(REQUIRED).Float(1e-6f);
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(SidecardMhcPreV310);
}  // namespace ops
