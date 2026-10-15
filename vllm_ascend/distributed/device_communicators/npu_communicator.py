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
from vllm.config import get_current_vllm_config_or_none
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
        self.custom_sum = None
        configuration = get_current_vllm_config_or_none()
        custom = (configuration.additional_config or {}).get("npu_collective") if configuration is not None else None
        if custom is not None and unique_name.split(":", 1)[0] == "tp":
            if configuration.model_config is None or (
                configuration.model_config.enforce_eager is not True
                and custom.get("execution_mode", "eager") != "graph"
            ):
                raise ValueError("experimental npu_collective requires an explicitly eager model or graph admission")
            # Optional experimental tooling is loaded only in configured worker
            # owners. The default installed plugin has no dependency on tools.
            from tools.sidecard.npu_collective_torch import create
            from vllm_ascend.utils import current_stream

            try:
                self.custom_sum = create(
                    custom,
                    tuple(self.ranks),
                    self.global_rank,
                    cpu_group,
                    self.device,
                    torch=torch,
                    dist=dist,
                    stream_getter=current_stream,
                )
            except BaseException as error:
                self.custom_sum = getattr(error, "collective_owner", None)
                raise
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
        self.close_custom_collective()
        if self.pipeline_transport is not None:
            self.pipeline_transport.destroy()
        super().destroy()

    def all_reduce(self, input_):
        if self.custom_sum is not None:
            return self.custom_sum.all_reduce(input_, lambda value: super(NPUCommunicator, self).all_reduce(value))
        return super().all_reduce(input_)

    def close_custom_collective(self):
        """Call on every TP rank after retiring eager work, before PG teardown."""
        if self.custom_sum is not None and self.custom_sum.state != "closed":
            self.custom_sum.close(graphs_retired=True)
