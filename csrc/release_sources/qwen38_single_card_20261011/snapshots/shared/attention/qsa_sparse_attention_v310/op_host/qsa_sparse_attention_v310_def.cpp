#include "register/op_def_registry.h"

namespace ops {

class QsaSparseAttentionV310 : public OpDef {
public:
    explicit QsaSparseAttentionV310(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("keyCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_FRACTAL_NZ});
        this->Input("valueCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_FRACTAL_NZ});
        this->Input("groupIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("groupCounts").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("tailStarts").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("tailCounts").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("blockTable").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("queryStartLoc").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Attr("scaleQ24").AttrType(REQUIRED).Int();
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

OP_ADD(QsaSparseAttentionV310);

}  // namespace ops
