#include "register/op_def_registry.h"

namespace ops {

class QsaIndexerScoreV310 : public OpDef {
public:
    explicit QsaIndexerScoreV310(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("compressedKeyCache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("blockTable").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("queryStartLoc").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("positions").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Attr("compressRatio").AttrType(OPTIONAL).Int(4);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", config);
    }
};

OP_ADD(QsaIndexerScoreV310);

}  // namespace ops
