// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t HEADS=32, DIM=128, TILE=64, INNER=16, ALIGN=8;
constexpr IsResetLoad3dConfig LOAD_CONFIG={true,true};
struct ScoreTiling {int64_t rows,pools,requests,columns,blocks,block_rows,block_stride,row_stride,cache_offset;};
class Score {
 public:
  __aicore__ inline void Run(GM_ADDR query, GM_ADDR weights, GM_ADDR cache, GM_ADDR table,
      GM_ADDR boundaries, GM_ADDR positions, GM_ADDR output, const ScoreTiling& d) {
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(query));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cache));
    table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(boundaries));
    pos_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(positions));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(ql1_,HEADS*DIM*sizeof(half));
    pipe_.InitBuffer(kl1_,TILE*DIM*sizeof(half));
    pipe_.InitBuffer(a_,HEADS*DIM*sizeof(half));
    pipe_.InitBuffer(b_,TILE*DIM*sizeof(half));
    pipe_.InitBuffer(c_,HEADS*TILE*sizeof(float));
    pipe_.InitBuffer(gather_,TILE*DIM*sizeof(half));
    pipe_.InitBuffer(scores_,HEADS*TILE*sizeof(float));
    pipe_.InitBuffer(rows_,HEADS*TILE*sizeof(float));
    pipe_.InitBuffer(weights_,HEADS*sizeof(float));
    SetMMLayoutTransform(true);
    for (int64_t row=0;row<d.rows;++row) {
      const int64_t position=pos_.GetValue(row);
      int64_t complete=position<0?0:(position+1)/4;
      complete=complete<d.pools?complete:d.pools;
      if (complete==0) continue;
      int64_t request=0;
      while (request+1<d.requests && row>=ends_.GetValue(request)) ++request;
      LoadQuery(row);
      auto weights=weights_.Get<float>();
      DataCopy(weights,w_[row*HEADS],HEADS);
      PipeBarrier<PIPE_ALL>();
      for (int64_t start=GetBlockIdx()*TILE;start<complete;start+=GetBlockNum()*TILE) {
        const int64_t count=complete-start<TILE?complete-start:TILE;
        const bool invalidPages=Gather(d,request,start,count);
        Compute();
        auto scores=scores_.Get<float>(); auto rows=rows_.Get<float>();
        const UnaryRepeatParams repack{1,1,TILE/ALIGN,INNER/ALIGN};
        for (int64_t n=0;n<TILE/INNER;++n) {
          Adds(rows[n*INNER],scores[n*HEADS*INNER],0.0f,INNER,HEADS,repack);
        }
        PipeBarrier<PIPE_V>();
        Maxs(rows,rows,0.0f,HEADS*TILE);
        PipeBarrier<PIPE_V>();
        for (int64_t head=0;head<HEADS;++head) {
          Muls(rows[head*TILE],rows[head*TILE],weights.GetValue(head),TILE);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t heads=HEADS/2;heads>0;heads/=2) {
          Add(rows,rows,rows[heads*TILE],heads*TILE);
          PipeBarrier<PIPE_V>();
        }
        // A partial final tile writes full DMA blocks. Its inactive lanes
        // retain -inf, including invalid or recycled page-table entries.
        const int64_t stored=(count+ALIGN-1)/ALIGN*ALIGN;
        uint64_t mask[2]={count<TILE?~((uint64_t(1)<<count)-1):uint64_t(0),0};
        if (invalidPages) {
          for (int64_t lane=0;lane<count;++lane) {
            const int64_t page=table_.GetValue(request*d.columns+(start+lane)/d.block_rows);
            if (page<0 || page>=d.blocks) mask[0]|=uint64_t(1)<<lane;
          }
        }
        // Vector masking avoids scalar writes racing the vector reduction.
        if (mask[0]) Duplicate(rows,-__builtin_inff(),mask,1,1,8);
        PipeBarrier<PIPE_ALL>();
        DataCopy(out_[row*d.pools+start],rows,stored);
        PipeBarrier<PIPE_ALL>();
      }
    }
    SetMMLayoutTransform(false);
  }
 private:
  __aicore__ inline void LoadQuery(int64_t row) {
    Nd2NzParams p;
    p.ndNum=1; p.nValue=HEADS; p.dValue=DIM; p.srcDValue=DIM; p.dstNzC0Stride=HEADS;
    p.dstNzNStride=1; p.srcNdMatrixStride=0; p.dstNzMatrixStride=0;
    DataCopy(ql1_.Get<half>(),q_[row*HEADS*DIM],p);
    PipeBarrier<PIPE_ALL>();
    LoadData3DParamsV2<half> v;
    v.l1H=HEADS/INNER; v.l1W=INNER; v.channelSize=DIM;
    v.padList[0]=v.padList[1]=v.padList[2]=0; v.padList[3]=255;
    v.mExtension=HEADS; v.kExtension=DIM; v.mStartPt=v.kStartPt=0;
    v.strideW=v.strideH=v.filterW=v.filterH=v.dilationFilterW=v.dilationFilterH=1;
    v.filterSizeW=v.filterSizeH=false; v.enTranspose=0; v.fMatrixCtrl=0;
    LoadData<half,LOAD_CONFIG>(a_.Get<half>(),ql1_.Get<half>(),v);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline bool Gather(const ScoreTiling& d,int64_t request,int64_t start,int64_t count) {
    auto gathered=gather_.Get<half>();
    Duplicate(gathered,static_cast<half>(0),TILE*DIM);
    PipeBarrier<PIPE_ALL>();
    bool invalid=false;
    int64_t lane=0;
    while (lane<count) {
      const int64_t logical=(start+lane)/d.block_rows, inpage=(start+lane)%d.block_rows;
      const int64_t page=table_.GetValue(request*d.columns+logical);
      const int64_t size=count-lane<d.block_rows-inpage?count-lane:d.block_rows-inpage;
      if (page>=0 && page<d.blocks) {
        const DataCopyParams copy{static_cast<uint16_t>(size),1,static_cast<uint16_t>(d.row_stride/INNER-1),0};
        const int64_t source=d.cache_offset+page*d.block_stride+inpage*d.row_stride;
        for (int64_t dim=0;dim<DIM/INNER;++dim) {
          DataCopy(gathered[dim*TILE*INNER+lane*INNER],k_[source+dim*INNER],copy);
        }
      } else { invalid=true; }
      lane+=size;
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(kl1_.Get<half>(),gathered,TILE*DIM);
    PipeBarrier<PIPE_ALL>();
    return invalid;
  }
  __aicore__ inline void Compute() {
    LoadData2DParams load;
    load.startIndex=0; load.repeatTimes=(DIM/INNER)*(TILE/INNER); load.srcStride=1;
    load.dstGap=0; load.ifTranspose=false;
    LoadData(b_.Get<half>(),kl1_.Get<half>(),load);
    PipeBarrier<PIPE_ALL>();
    MmadParams mm; mm.m=HEADS; mm.n=TILE; mm.k=DIM; mm.cmatrixInitVal=true;
    Mmad(c_.Get<float>(),a_.Get<half>(),b_.Get<half>(),mm);
    PipeBarrier<PIPE_ALL>();
    const DataCopyParams copy{TILE/INNER,HEADS/INNER,0,0};
    DataCopyEnhancedParams enhanced; enhanced.blockMode=BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(scores_.Get<float>(),c_.Get<float>(),copy,enhanced);
    PipeBarrier<PIPE_ALL>();
  }
  TPipe pipe_;
  TBuf<TPosition::A1> ql1_;
  TBuf<TPosition::B1> kl1_;
  TBuf<TPosition::A2> a_;
  TBuf<TPosition::B2> b_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> gather_,scores_,rows_,weights_;
  GlobalTensor<half> q_,k_;
  GlobalTensor<float> w_,out_;
  GlobalTensor<int32_t> table_,ends_,pos_;
};
}
extern "C" __global__ __aicore__ void glm_kpool_score_v310(GM_ADDR query,GM_ADDR weights,GM_ADDR cache,
    GM_ADDR table,GM_ADDR boundaries,GM_ADDR positions,GM_ADDR output,GM_ADDR workspace,GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  ScoreTiling d;
  auto src=reinterpret_cast<__gm__ int64_t*>(tiling);
  auto dst=reinterpret_cast<int64_t*>(&d);
  for (int i=0;i<9;++i) dst[i]=src[i];
  Score op; op.Run(query,weights,cache,table,boundaries,positions,output,d);
}
