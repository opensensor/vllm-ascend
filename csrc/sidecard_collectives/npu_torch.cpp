// SPDX-License-Identifier: Apache-2.0
#include "npu_group.h"

#include <torch/library.h>
#include <torch/types.h>

#include <limits>

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
constexpr uint64_t MAX_BYTES = 64 * 1024 * 1024;

void validate(const at::Tensor& input, int64_t handle, int64_t timeout) {
  TORCH_CHECK(input.is_contiguous(), "sidecard_comm requires contiguous input");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat,
              "sidecard_comm requires FP16 or FP32 input");
  TORCH_CHECK(input.nbytes() > 0 && input.nbytes() <= MAX_BYTES,
              "sidecard_comm input must contain 1 through 64 MiB of data");
  TORCH_CHECK(handle > 0 && static_cast<uint64_t>(handle) <= std::numeric_limits<uintptr_t>::max(),
              "sidecard_comm requires a live owner-local group handle");
  TORCH_CHECK(timeout > 0 && static_cast<uint64_t>(timeout) <= std::numeric_limits<uint32_t>::max(),
              "sidecard_comm timeout must fit a positive uint32");
}

at::Tensor all_reduce(const at::Tensor& input, int64_t handle, int64_t timeout) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1, "sidecard_comm requires PrivateUse1 input");
  validate(input, handle, timeout);
  auto* group = reinterpret_cast<void*>(static_cast<uintptr_t>(handle));
  uint64_t bytes = 0;
  uint32_t dtype = 0;
  int32_t device_index = -1;
  const int metadata_status = sidecard_group_metadata(group, &bytes, &dtype, &device_index);
  TORCH_CHECK(metadata_status == 0, "sidecard_comm group metadata failed: ", metadata_status);
  TORCH_CHECK(input.nbytes() == bytes && dtype == (input.scalar_type() == at::kFloat ? 0U : 1U) &&
                  device_index == input.get_device(),
              "sidecard_comm input differs from group geometry");
  const c10_npu::OptionalNPUGuard guard(input.device());
  auto output = at::empty_like(input);
  const auto stream = c10_npu::getCurrentNPUStream(input.get_device()).stream(false);
  auto handler = [input_owner = input, output_owner = output, handle, timeout, stream]() mutable -> int {
    const int status =
        sidecard_group_enqueue(reinterpret_cast<void*>(static_cast<uintptr_t>(handle)), input_owner.const_data_ptr(),
                               output_owner.mutable_data_ptr(), stream, static_cast<uint32_t>(timeout));
    TORCH_CHECK(status == 0, "sidecard_comm native enqueue failed: ", status);
    // Torch queue slots may retain a launched handler. Once submission is
    // complete, normal stream/allocator ownership governs tensor lifetime.
    input_owner = at::Tensor();
    output_owner = at::Tensor();
    return status;
  };
  at_npu::native::OpCommand command;
  command.Name("SidecardGroupAllReduce");
  command.SetCustomHandler(handler);
  command.Run();
  return output;
}

at::Tensor all_reduce_meta(const at::Tensor& input, int64_t handle, int64_t timeout) {
  validate(input, handle, timeout);
  return at::empty_like(input);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(sidecard_comm, ops) { ops.def("all_reduce(Tensor input, int handle, int timeout) -> Tensor"); }

TORCH_LIBRARY_IMPL(sidecard_comm, PrivateUse1, ops) { ops.impl("all_reduce", &all_reduce); }

TORCH_LIBRARY_IMPL(sidecard_comm, Meta, ops) { ops.impl("all_reduce", &all_reduce_meta); }
