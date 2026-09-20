/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gdn_gating_v310_def.cpp
 * \brief GDN gate for Qwen3.5/3.8 linear-attention layers on Ascend 310P.
 *
 * Replaces a 7-op framework chain that runs in 48 of 64 layers every decoded
 * token. 310P decode is bound by kernel count (~14.2 us per launch measured),
 * so collapsing those launches into one is the whole point of this operator.
 *
 * negExpALogTiled / dtBiasTiled are the weight-only constants -exp(A_log) and
 * dt_bias, pre-broadcast by the caller to [TILE_ROWS, H]. They are passed
 * pre-tiled because a GDN head count of H = 48/TP gives unaligned 4*H byte
 * rows, which DataCopy cannot move; a [TILE_ROWS, H] block is always a
 * multiple of 32 bytes. They are constants, so the caller builds them once.
 */
#include "register/op_def_registry.h"

namespace ops {

class GdnGatingV310 : public OpDef {
public:
    explicit GdnGatingV310(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("negExpALogTiled")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dtBiasTiled")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("g")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("betaOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("beta").AttrType(OPTIONAL).Float(1.0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", aicoreConfig);
    }
};
OP_ADD(GdnGatingV310);

}  // namespace ops
