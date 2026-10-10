#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.device_communicators.pipeline_all_gather import PipelineAllGatherTransport


class _NpuAll2AllManager:
    """No-op all2all_manager for NPU. Used by vLLM main's fault-tolerance
    check (data_parallel_size > 1 and is_moe); NPU does not register a real
    one because it uses mc2 / all_gather for MoE communication.
    """

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            use_all2all=use_all2all,
        )
        self.device = torch.npu.current_device()
        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager()
        self.pipeline_transport = None
        if unique_name.startswith("pp:") and get_current_hardware_profile().supports(
            HardwareCapability.DISTRIBUTED_COMMUNICATION_ADAPTATION
        ):
            self.pipeline_transport = PipelineAllGatherTransport(
                cpu_group, self.ranks, self.global_rank, torch.device("npu", self.device)
            )
            self.pipeline_transport.warmup(synchronize=torch.npu.synchronize)

    @property
    def supports_tensor_dict(self):
        return self.pipeline_transport is not None

    def send_tensor_dict(self, packet, dst=None):
        if self.pipeline_transport is None:
            raise RuntimeError("custom pipeline transport is unavailable for this communicator")
        self.pipeline_transport.send_tensor_dict(packet, dst)

    def recv_tensor_dict(self, src=None):
        if self.pipeline_transport is None:
            raise RuntimeError("custom pipeline transport is unavailable for this communicator")
        return self.pipeline_transport.recv_tensor_dict(src)

    def destroy(self):
        if self.pipeline_transport is not None:
            self.pipeline_transport.destroy()
        super().destroy()
