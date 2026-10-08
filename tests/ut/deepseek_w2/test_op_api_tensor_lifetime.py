# SPDX-License-Identifier: Apache-2.0
"""Compile the production command macro against a deferred host queue.

ACL tensor descriptors only retain addresses. The command must own direct
Tensor arguments and workspace until its queued launch actually executes.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_deferred_command_keeps_input_output_and_workspace_alive(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    header = Path(__file__).resolve().parents[3] / "csrc/aclnn_torch_adapter/op_api_common.h"
    macro = (
        "#ifdef GLM_EXPERIMENTAL_OP_API_TENSOR_OWNERS"
        + header.read_text().split("#ifdef GLM_EXPERIMENTAL_OP_API_TENSOR_OWNERS", 1)[1].rsplit("#endif", 1)[0]
    )
    source = tmp_path / "lifetime.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>
std::weak_ptr<int> input_owner, output_owner, workspace_owner;
std::vector<std::function<int()>> queue;
namespace at {
struct Tensor {
    std::shared_ptr<int> value;
    struct Storage { void* pointer; void* data() const { return pointer; } };
    Storage storage() const { return {value.get()}; }
};
struct TensorOptions {
    explicit TensorOptions(int) {}
    TensorOptions dtype(int) const { return *this; }
};
Tensor empty(std::initializer_list<uint64_t>, TensorOptions) {
    Tensor result{std::make_shared<int>(0)};
    workspace_owner = result.value;
    return result;
}
}
namespace torch_npu { namespace utils { int get_npu_device_type() { return 0; } } }
namespace c10_npu {
struct Stream { void* stream(bool) const { return nullptr; } };
Stream getCurrentNPUStream() { return {}; }
}
namespace at_npu { namespace native {
struct OpCommand {
    std::function<int()> handler;
    void Name(const char*) {}
    void SetCustomHandler(std::function<int()> fn) { handler = fn; }
    void Run() { queue.push_back(handler); }
};
}}
using aclrtStream = void*;
struct aclOpExecutor {};
using InitHugeMemThreadLocal = void(*)(void*, bool);
using UnInitHugeMemThreadLocal = void(*)(void*, bool);
using ReleaseHugeMem = void(*)(void*, bool);
constexpr int kByte = 0;
int* input_address;
int* output_address;
int get_workspace(int* input, int* output, uint64_t* size, aclOpExecutor**) {
    input_address = input;
    output_address = output;
    *size = sizeof(int);
    return 0;
}
int launch(void* workspace, uint64_t, aclOpExecutor*, aclrtStream) {
    assert(!input_owner.expired());
    assert(!output_owner.expired());
    assert(!workspace_owner.expired());
    *static_cast<int*>(workspace) = *input_address + 7;
    *output_address = *static_cast<int*>(workspace);
    assert(*output_address == 49);
    return 0;
}
void* GetOpApiFuncAddr(const char* name) {
    if (std::strcmp(name, "FakeGetWorkspaceSize") == 0) return reinterpret_cast<void*>(&get_workspace);
    if (std::strcmp(name, "Fake") == 0) return reinterpret_cast<void*>(&launch);
    return nullptr;
}
const char* GetOpApiLibName() { return "fake"; }
const char* aclGetRecentErrMsg() { return "fake"; }
#define TORCH_CHECK(condition, ...) assert(condition)
auto ConvertTypes(const at::Tensor& input, const at::Tensor& output, uint64_t* size, aclOpExecutor** executor) {
    return std::make_tuple(input.value.get(), output.value.get(), size, executor);
}
template<class Tuple> auto ConvertToOpApiFunc(Tuple&, void* function) {
    return reinterpret_cast<decltype(&get_workspace)>(function);
}
template<class F, class T> int call(F function, T args) { return std::apply(function, args); }
template<class T> void ReleaseConvertTypes(T&) {}
"""
        + "#define GLM_EXPERIMENTAL_OP_API_TENSOR_OWNERS\n"
        + macro
        + r"""
int main() {
    {
        at::Tensor input{std::make_shared<int>(42)}, output{std::make_shared<int>(0)};
        input_owner = input.value;
        output_owner = output.value;
        EXEC_NPU_CMD(Fake, input, output);
    }
    assert(queue.size() == 1);
    assert(!input_owner.expired());
    assert(!output_owner.expired());
    assert(!workspace_owner.expired());
    assert(queue.front()() == 0);
    // Keep the completed queue slot alive, as the runtime queue can. Storage
    // must be reusable after launch rather than pinned until slot recycling.
    assert(input_owner.expired());
    assert(output_owner.expired());
    assert(workspace_owner.expired());
    queue.clear();
}
"""
    )
    binary = tmp_path / "lifetime"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Werror", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
