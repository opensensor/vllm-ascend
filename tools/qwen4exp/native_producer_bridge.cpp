// SPDX-License-Identifier: Apache-2.0
// Eager bridge: only the producer releases operands after host queue drain.
#include <torch/extension.h>
#include <torch/custom_class.h>
#include <acl/acl.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "producer_launch_arguments.h"

#ifndef GLM_RECONSTRUCTION_NAMESPACE
  #error "owned bridge requires an explicit new versioned namespace"
#endif
#define GLM_STRINGIFY_INNER(value) #value
#define GLM_STRINGIFY(value) GLM_STRINGIFY_INNER(value)
#define GLM_REGISTER_LIBRARY(value) TORCH_LIBRARY(value, m)

namespace GLM_RECONSTRUCTION_NAMESPACE {
using ArgumentOwner = qwen_native::ProducerLaunchArguments<at::Tensor>;

void Check(aclError status, const char* operation) {
  TORCH_CHECK(status == ACL_SUCCESS, operation, " failed: ", status, " ", aclGetRecentErrMsg());
}

class NativeKernel : public torch::CustomClassHolder {
 public:
  NativeKernel(std::string path, std::string name) {
    Check(aclrtGetDevice(&device_), "aclrtGetDevice");
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions options{&option, 1};
    std::ifstream file(path, std::ios::binary);
    TORCH_CHECK(file.good(), "cannot read native binary: ", path);
    bytes_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    TORCH_CHECK(!bytes_.empty(), "empty native binary: ", path);
    Check(aclrtBinaryLoadFromData(bytes_.data(), bytes_.size(), &options, &binary_), "aclrtBinaryLoadFromData");
    Check(aclrtBinaryGetFunction(binary_, name.c_str(), &function_), "aclrtBinaryGetFunction");
  }

  void RetainFailed(const std::shared_ptr<ArgumentOwner>& owner) {
    // Failure-only locking. The resident kernel resource retains failed
    // operands even if every framework callback copy is later destroyed.
    std::lock_guard<std::mutex> lock(failureMutex_);
    failedOwners_.push_back(owner);
  }

  // As in the qualified bridge, never unload code referenced by native graphs.
  aclrtBinHandle binary_ = nullptr;
  aclrtFuncHandle function_ = nullptr;
  int32_t device_ = -1;
  std::vector<char> bytes_;

 private:
  std::mutex failureMutex_;
  std::vector<std::shared_ptr<ArgumentOwner>> failedOwners_;
};

void Launch(c10::intrusive_ptr<NativeKernel> kernel, std::vector<at::Tensor> arguments, int64_t blocks) {
  TORCH_CHECK(!arguments.empty() && blocks > 0 && blocks <= 8, "invalid native launch geometry");
  const auto device = arguments[0].device();
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 && device.index() == kernel->device_,
              "kernel and arguments must belong to the same NPU");
  for (const auto& tensor : arguments) {
    TORCH_CHECK(tensor.device() == device && tensor.is_contiguous(), "arguments must be contiguous on one NPU");
  }
  const c10_npu::OptionalNPUGuard guard(device);
  const auto currentStream = c10_npu::getCurrentNPUStream();
  const auto stream = currentStream.stream(false);
  // Host submission is not device completion. Inputs created on a gather
  // stream must stay allocated until this consuming stream has finished.
  // Record allocator ownership on the producer, never inside the callback.
  for (const auto& tensor : arguments) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), currentStream);
  }
  auto owner = std::make_shared<ArgumentOwner>(std::move(arguments));
  auto handler = [kernel, owner, blocks, stream]() -> int {
    int status;
    try {
      status = owner->Run([&](const std::vector<at::Tensor>& tensors) -> int {
        std::vector<void*> addresses;
        addresses.reserve(tensors.size());
        for (const auto& tensor : tensors) addresses.push_back(tensor.data_ptr());
        std::vector<void*> pointers;
        pointers.reserve(addresses.size());
        for (auto& address : addresses) pointers.push_back(&address);
        return aclrtLaunchKernelWithArgsArray(kernel->function_, blocks, stream, nullptr, pointers.data());
      });
    } catch (...) {
      kernel->RetainFailed(owner);
      throw;
    }
    if (status != ACL_SUCCESS) kernel->RetainFailed(owner);
    Check(static_cast<aclError>(status), "aclrtLaunchKernelWithArgsArray");
    return status;
  };
  try {
    at_npu::native::OpCommand command;
    command.Name("QwenProducerNativeV1");
    command.SetCustomHandler(handler);
    command.Run();

    // Producer-thread drain only: never wait for this queue from its handler.
    // Queued callbacks may themselves wait for device work while this drains.
    TORCH_CHECK(currentStream.stream(true) == stream, "native stream changed while draining host queue");
    owner->Release();
  } catch (...) {
    // Queue acceptance may be uncertain even when the host command throws.
    // Keep arguments alive independently of the framework's callback copies.
    kernel->RetainFailed(owner);
    throw;
  }
}
}  // namespace GLM_RECONSTRUCTION_NAMESPACE

GLM_REGISTER_LIBRARY(GLM_RECONSTRUCTION_NAMESPACE) {
  m.class_<GLM_RECONSTRUCTION_NAMESPACE::NativeKernel>("Kernel").def(torch::init<std::string, std::string>());
  m.def("launch(__torch__.torch.classes." GLM_STRINGIFY(
      GLM_RECONSTRUCTION_NAMESPACE) ".Kernel kernel, Tensor[] arguments, int blocks) -> ()");
  m.impl("launch", torch::kPrivateUse1, &GLM_RECONSTRUCTION_NAMESPACE::Launch);
}
