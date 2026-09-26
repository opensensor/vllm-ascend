// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "register/op_def_registry.h"

namespace ops {

class QsaGatherValueNzV310 : public OpDef {
public:
    explicit QsaGatherValueNzV310(const char *name) : OpDef(name)
    {
        this->Input("valueCache").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_FRACTAL_NZ});
        this->Input("groupIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("groupCounts").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("tailStarts").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("tailCounts").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("blockTable").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("valueNz").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_FRACTAL_NZ});
        this->Attr("numKvHeads").AttrType(REQUIRED).Int();
        this->Attr("headDim").AttrType(REQUIRED).Int();
        this->Attr("transposeOutput").AttrType(REQUIRED).Bool(false);
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

OP_ADD(QsaGatherValueNzV310);

}  // namespace ops
