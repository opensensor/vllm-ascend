# SPDX-License-Identifier: Apache-2.0
"""Execute native staging with CPU pipe/layout stubs; never import NPU runtime."""

import ast
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import streaming_memory as memory
from tools.qwen4exp import streaming_operands as operands

ROOT = Path(__file__).resolve().parents[3]


def fixtures(tokens, top_k, local_rows, experts=4):
    inputs = torch.arange(tokens * 256, dtype=torch.float16).reshape(tokens, 256)
    weights = torch.ones(tokens, top_k)
    ids = torch.zeros(tokens, top_k, dtype=torch.int64)
    calls = []

    def pack(x):
        calls.append(("pack", x.shape[0]))
        low = torch.arange(x.shape[0] * 128).reshape(x.shape[0], 128).to(torch.int8)
        high = -low
        scale = torch.arange(x.shape[0] * 16).reshape(x.shape[0], 2, 8).float()
        return low, high, scale, scale + 100

    def dispatch(w, i, **kwargs):
        assert kwargs == {"num_local_experts": experts, "expert_offset": 16}
        rows = tokens * top_k
        order = torch.arange(rows - 1, -1, -1)
        ends = torch.full((experts,), local_rows, dtype=torch.int64)
        return SimpleNamespace(order=order, token_indices=torch.arange(rows) // top_k, group_list=ends)

    def gather(prepared, sorted_tokens, ends):
        # CPU fixture uses declared local_rows to simulate device end-controlled work.
        # Poison stale capacity; native projections must never consume peer tails.
        calls.append(("gather", local_rows))
        result = [torch.full((sorted_tokens.numel(), *x.shape[1:]), 37, dtype=x.dtype) for x in prepared]
        for x, out in zip(prepared, result):
            out[:local_rows] = x.index_select(0, sorted_tokens[:local_rows].long())
        return result

    return inputs, weights, ids, pack, dispatch, gather, calls


@pytest.mark.parametrize("tokens,top_k,local_rows", [(1, 10, 0), (16, 10, 1), (16, 10, 159), (16, 10, 160), (3, 2, 4)])
def test_quantized_once_and_only_stable_local_prefix_consumed(tokens, top_k, local_rows):
    inputs, weights, ids, pack, dispatch, gather, calls = fixtures(tokens, top_k, local_rows)
    result = operands.prepare_grouped_operands(
        inputs, weights, ids, pack=pack, dispatch=dispatch, gather=gather, num_local_experts=4, expert_offset=16
    )
    assert calls == [("pack", tokens), ("gather", local_rows)]
    assert result.physical_rows == tokens * top_k
    assert result.sorted_tokens.dtype == torch.int32
    for original, output in zip(result.token_operands, result.local_operands):
        assert torch.equal(output[:local_rows], original.index_select(0, result.sorted_tokens[:local_rows].long()))
        assert torch.all(output[local_rows:] == 37)


def test_route_shrink_grow_all_peer_keep_physical_capacity():
    for live in (16, 1, 0, 16):
        values = fixtures(4, 4, live)
        inputs, weights, ids, pack, dispatch, gather, calls = values
        result = operands.prepare_grouped_operands(
            inputs, weights, ids, pack=pack, dispatch=dispatch, gather=gather, num_local_experts=4, expert_offset=16
        )
        assert result.local_operands[0].shape == (16, 128)
        assert calls[1] == ("gather", live)


def test_empty_token_batch_does_not_dispatch_quantize_or_gather():
    values = fixtures(0, 10, 0)
    inputs, weights, ids, pack, dispatch, gather, calls = values
    result = operands.prepare_grouped_operands(
        inputs, weights, ids, pack=pack, dispatch=dispatch, gather=gather, num_local_experts=4
    )
    assert result.physical_rows == 0 and result.local_operands == () and calls == []


@pytest.mark.parametrize("field", ["tokens", "top_k", "width", "experts", "offset"])
def test_geometry_bounds(field):
    inputs, weights, ids, pack, dispatch, gather, _ = fixtures(2, 2, 0)
    experts, offset = 4, 16
    if field == "tokens":
        inputs = torch.empty(2561, 256)
        weights, ids = torch.empty(2561, 2), torch.empty(2561, 2, dtype=torch.long)
    elif field == "top_k":
        weights, ids = torch.empty(2, 11), torch.empty(2, 11, dtype=torch.long)
    elif field == "width":
        inputs = torch.empty(2, 255)
    elif field == "experts":
        experts = 129
    else:
        offset = -1
    with pytest.raises(ValueError, match="geometry"):
        operands.prepare_grouped_operands(
            inputs,
            weights,
            ids,
            pack=pack,
            dispatch=dispatch,
            gather=gather,
            num_local_experts=experts,
            expert_offset=offset,
        )


@pytest.mark.parametrize("broken", ["ends", "order", "quant_dtype", "quant_shape", "gather_dtype", "gather_shape"])
def test_callback_contract_failures(broken):
    inputs, weights, ids, pack, dispatch, gather, _ = fixtures(2, 2, 3)
    original_pack, original_dispatch, original_gather = pack, dispatch, gather
    if broken in ("ends", "order"):

        def dispatch(*args, **kwargs):
            result = original_dispatch(*args, **kwargs)
            if broken == "ends":
                result.group_list = result.group_list.int()
            else:
                result.order = result.order[:-1]
            return result

    elif broken.startswith("quant"):

        def pack(x):
            result = list(original_pack(x))
            result[0] = result[0].float() if broken == "quant_dtype" else result[0][:, :-1]
            return result

    else:

        def gather(*args):
            result = list(original_gather(*args))
            result[0] = result[0].float() if broken == "gather_dtype" else result[0][:-1]
            return result

    with pytest.raises(ValueError):
        operands.prepare_grouped_operands(
            inputs, weights, ids, pack=pack, dispatch=dispatch, gather=gather, num_local_experts=4, expert_offset=16
        )


def test_no_device_value_controls_or_npu_imports():
    tree = ast.parse((ROOT / "tools/qwen4exp/streaming_operands.py").read_text())
    forbidden = {"item", "cpu", "tolist", "nonzero", "numpy", "synchronize"}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in forbidden]
    assert "torch_npu" not in (ROOT / "tools/qwen4exp/streaming_operands.py").read_text()


@pytest.mark.parametrize("local_rows", [0, 1, 25600])
def test_transfer_cost_complete_and_does_not_claim_speed(local_rows):
    cost = operands.operand_transfer_cost(2560, 10, local_rows, 2560, 10)
    row = 2560 + 2 * 20 * 8 * 4
    assert cost["packed_bytes_per_row"] == row
    assert cost["compact_operand_storage_bytes"] == 25600 * row
    assert cost["indexed_total_operand_bytes"] == local_rows * row * 10
    assert cost["compact_total_operand_bytes"] == local_rows * row * 12
    assert cost["quantized_tokens"] == 2560
    assert cost["selected_backend"] is None and not cost["measured_bus_bytes"]


def test_independent_event_model_rejects_premature_slot_reuse():
    proof = memory.Ownership((memory.Region("operand", "UB", 0, 2048, "byte", "packed", "cube"),))
    proof.acquire("operand", 0)
    with pytest.raises(ValueError, match="reuse"):
        proof.acquire("operand", 2)
    proof.signal("M_MTE1", 0, 0)
    token = proof.wait("M_MTE1", 0, 0)
    proof.release("operand", 0, token)
    proof.acquire("operand", 2)


CPP_STUB = r"""
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
namespace AscendC {
using int4b_t = int8_t;
enum class HardEvent { M_MTE1, V_MTE2, MTE2_V, MTE2_MTE3, V_MTE3, MTE3_MTE1,
                       MTE3_MTE2, MTE1_MTE3, MTE1_M };
inline std::vector<int> pending;
inline std::vector<int> events;
template<HardEvent E> void SetFlag(uint32_t slot) {
  int key = static_cast<int>(E)*2+slot;
  if (std::find(pending.begin(),pending.end(),key)!=pending.end()) throw std::runtime_error("double event");
  pending.push_back(key); events.push_back(key);
}
template<HardEvent E> void WaitFlag(uint32_t slot) {
  int key = static_cast<int>(E)*2+slot;
  auto found=std::find(pending.begin(),pending.end(),key);
  if (found==pending.end()) throw std::runtime_error("missing event");
  pending.erase(found);
}
template<class T> struct Tensor {
  T *data{};
  Tensor operator[](int64_t offset) const {return {data+offset};}
  template<class U> Tensor<U> ReinterpretCast() const {return {reinterpret_cast<U*>(data)};}
};
template<class T> using LocalTensor=Tensor<T>;
template<class T> using GlobalTensor=Tensor<T>;
struct DataCopyParams {uint16_t blockCount, blockLen, srcStride, dstStride;};
struct LoadData2DParams {uint16_t repeatTimes{}, srcStride{};bool ifTranspose{};};
template<class T> void Duplicate(Tensor<T> destination,T value,uint32_t count) {
  std::fill_n(destination.data,count,value);
}
template<class T> void DataCopy(Tensor<T> destination,Tensor<T> source,uint32_t count) {
  std::copy_n(source.data,count,destination.data);
}
template<class T> void DataCopy(Tensor<T> destination,Tensor<T> source,DataCopyParams p) {
  for (uint32_t block=0;block<p.blockCount;block++) {
    std::memcpy(reinterpret_cast<uint8_t*>(destination.data)+block*(p.blockLen+p.dstStride)*32,
                reinterpret_cast<uint8_t*>(source.data)+block*(p.blockLen+p.srcStride)*32,p.blockLen*32);
  }
}
inline void LoadData(Tensor<int4b_t> destination,Tensor<int4b_t> source,LoadData2DParams p) {
  for (uint32_t repeat=0;repeat<p.repeatTimes;repeat++) {
    std::memcpy(destination.data+repeat*512,source.data+repeat*p.srcStride*512,512);
  }
}
}
"""

CPP_HARNESS = r"""
#include <iostream>
#include "qwen_streaming_operands.h"
using namespace qwen_streaming;
int main() {
  constexpr uint32_t rows=35,k=MAX_K,groups=k/GROUP;
  std::vector<int8_t> low(rows*k/2), high(rows*k/2), resident(N*k/2);
  for (uint32_t i=0;i<low.size();i++){low[i]=int8_t((i*13+1)%251);high[i]=int8_t((i*7+17)%251);}
  for (uint32_t i=0;i<resident.size();i++) resident[i]=int8_t((i*19+3)%251);
  std::vector<float> scale(rows*groups*LANES),sum(scale.size());
  for(uint32_t i=0;i<scale.size();i++){scale[i]=float(i)+0.25f;sum[i]=-float(i)-0.75f;}
  std::vector<int8_t> packed(PACKED_ACTIVATION_BYTES,71),a1(ACTIVATION_L1_BYTES,71),
                       a2(ACTIVATION_L0_BYTES,71),b2(WEIGHT_L0_BYTES,71);
  std::vector<float> meta(ACTIVATION_METADATA_BYTES/4,71);
  OperandProducer producer;
  producer.Init({low.data()},{high.data()},{scale.data()},{sum.data()},{packed.data()},{meta.data()},
                {a1.data()},{a2.data()},{b2.data()},{resident.data()},k);
  for(uint32_t live : {16u,1u,15u,0u,16u}) for(uint32_t group=0;group<groups;group++) {
    uint32_t slot=group%2,row=17;
    auto oldPacked=packed,oldA=a2,oldB=b2; auto oldMeta=meta;
    producer.Produce(slot,row,group,live);
    if(!AscendC::pending.empty()) return 1;
    // Independent explicit layout formulas, not helper-index round trips.
    for(uint32_t limb=0;limb<2;limb++) for(uint32_t kb=0;kb<2;kb++)
      for(uint32_t m=0;m<M;m++) for(uint32_t byte=0;byte<32;byte++) {
        uint32_t local=slot*2048+limb*1024+kb*512+m*32+byte;
        uint32_t source=(row+m)*k/2+group*64+kb*32+byte;
        int8_t expected=m<live?(limb?high[source]:low[source]):0;
        if(packed[local]!=expected || a1[local]!=expected || a2[local]!=expected) return 2;
      }
    for(uint32_t bank=0;bank<2;bank++) for(uint32_t m=0;m<M;m++) for(uint32_t lane=0;lane<LANES;lane++) {
      uint32_t address=slot*256+bank*128+m*8+lane;
      uint32_t source=((row+m)*groups+group)*8+lane;
      float expected=m<live?(bank?sum[source]:scale[source]):0;
      if(meta[address]!=expected) return 3;
    }
    for(uint32_t kb=0;kb<2;kb++) for(uint32_t nb=0;nb<N/16;nb++)
      for(uint32_t n=0;n<16;n++) for(uint32_t byte=0;byte<32;byte++) {
        uint32_t destination=slot*8192+kb*4096+nb*512+n*32+byte;
        uint32_t source=nb*groups*1024+group*1024+kb*512+n*32+byte;
        if(b2[destination]!=resident[source]) return 4;
      }
    uint32_t other=1-slot;
    if(!std::equal(packed.begin()+other*2048,packed.begin()+(other+1)*2048,oldPacked.begin()+other*2048))return 5;
    if(!std::equal(a2.begin()+other*2048,a2.begin()+(other+1)*2048,oldA.begin()+other*2048))return 6;
    if(!std::equal(b2.begin()+other*8192,b2.begin()+(other+1)*8192,oldB.begin()+other*8192))return 7;
    if(!std::equal(meta.begin()+other*256,meta.begin()+(other+1)*256,oldMeta.begin()+other*256))return 8;
  }
  std::cout<<"100 producer calls: packed limbs, scales, L1/L0A/L0B, tail poison and paired events exact\n";
}
"""


def test_compiled_native_producer_exact_packing_padding_slots_and_events(tmp_path):
    header = (ROOT / "tools/qwen4exp/qwen_streaming_operands.h").read_text()
    for helper in ("ActivationByte", "ActivationL0Byte", "WeightL1Byte", "WeightL0Byte"):
        assert "__aicore__ constexpr uint32_t " + helper in header
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("CPU C++ compiler unavailable")
    (tmp_path / "kernel_operator.h").write_text(CPP_STUB)
    source = tmp_path / "check.cpp"
    source.write_text(CPP_HARNESS)
    output = tmp_path / "check"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "tools/qwen4exp"),
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(output)], check=True, capture_output=True, text=True)
    assert "100 producer calls" in result.stdout


def test_maximum_physical_routes_remain_bounded():
    inputs, weights, ids, pack, dispatch, gather, calls = fixtures(2560, 10, 1)
    result = operands.prepare_grouped_operands(
        inputs, weights, ids, pack=pack, dispatch=dispatch, gather=gather, num_local_experts=4, expert_offset=16
    )
    assert result.physical_rows == 25600
    assert calls == [("pack", 2560), ("gather", 1)]


@pytest.mark.parametrize(
    "geometry",
    [
        (2561, 10, 1, 2560, 10),
        (2560, 11, 1, 2560, 10),
        (2, 2, 5, 256, 2),
        (2, 2, 1, 255, 2),
        (2, 2, 1, 256, 0),
        (2, 2, 1.0, 256, 2),
    ],
)
def test_invalid_cost_geometry(geometry):
    with pytest.raises(ValueError):
        operands.operand_transfer_cost(*geometry)
