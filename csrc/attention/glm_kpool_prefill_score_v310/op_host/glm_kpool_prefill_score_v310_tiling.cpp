// SPDX-License-Identifier: Apache-2.0
#include "glm_kpool_prefill_score_v310_tiling.h"
#include "../score_geometry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
namespace optiling {
static ge::graphStatus Tile(gert::TilingContext* c) {
    OP_CHECK_NULL_WITH_CONTEXT(c, c->GetPlatformInfo());
    for (int i = 0; i < 6; ++i) {
        OP_CHECK_NULL_WITH_CONTEXT(c, c->GetInputShape(i));
        OP_CHECK_NULL_WITH_CONTEXT(c, c->GetAttrs()->GetInt(i));
    }
    const auto q=c->GetInputShape(0)->GetStorageShape(), w=c->GetInputShape(1)->GetStorageShape();
    const auto cache=c->GetInputShape(2)->GetStorageShape(), table=c->GetInputShape(3)->GetStorageShape();
    const auto bounds=c->GetInputShape(4)->GetStorageShape(), pos=c->GetInputShape(5)->GetStorageShape();
    const auto pools=*c->GetAttrs()->GetInt(0), blocks=*c->GetAttrs()->GetInt(1), br=*c->GetAttrs()->GetInt(2);
    const auto bs=*c->GetAttrs()->GetInt(3), rs=*c->GetAttrs()->GetInt(4), off=*c->GetAttrs()->GetInt(5);
    OP_CHECK_IF(q.GetDimNum()!=3 || q.GetDim(1)!=32 || q.GetDim(2)!=128 || w.GetDimNum()!=2 ||
        w.GetDim(0)!=q.GetDim(0) || w.GetDim(1)!=32 || cache.GetDimNum()!=1 ||
        table.GetDimNum()!=2 || table.GetDim(0)!=1 || bounds.GetDimNum()!=1 || bounds.GetDim(0)!=table.GetDim(0) ||
        pos.GetDimNum()!=1 || pos.GetDim(0)!=q.GetDim(0), OP_LOGE(c,"invalid score input shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!NsGlmKpoolPrefillScore::ValidLayout(q.GetDim(0),pools,blocks,br,bs,rs,off,cache.GetDim(0)) ||
        pools>table.GetDim(1)*br, OP_LOGE(c,"invalid cache geometry"), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
    OP_CHECK_IF(platform.GetCoreNumAic()==0, OP_LOGE(c,"no Cube cores"), return ge::GRAPH_FAILED);
    GlmKpoolPrefillScoreTilingData d;
    d.set_rows(q.GetDim(0)); d.set_pools(pools); d.set_requests(table.GetDim(0)); d.set_columns(table.GetDim(1));
    d.set_blocks(blocks); d.set_block_rows(br); d.set_block_stride(bs); d.set_row_stride(rs); d.set_cache_offset(off);
    c->SetBlockDim(platform.GetCoreNumAic()); c->SetTilingKey(0);
    auto ws=c->GetWorkspaceSizes(1); OP_CHECK_NULL_WITH_CONTEXT(c,ws); ws[0]=platform.GetLibApiWorkSpaceSize();
    d.SaveToBuffer(c->GetRawTilingData()->GetData(),c->GetRawTilingData()->GetCapacity());
    c->GetRawTilingData()->SetDataSize(d.GetDataSize()); return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) {return ge::GRAPH_SUCCESS;}
IMPL_OP_OPTILING(GlmKpoolPrefillScoreV310).Tiling(Tile).TilingParse<CompileInfo>(Parse);
}
