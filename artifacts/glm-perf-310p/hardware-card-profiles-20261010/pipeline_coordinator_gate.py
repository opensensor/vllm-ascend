"""Real 310P GroupCoordinator pipeline packets; no model equivalence claim."""

import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch


def main():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=180))
    original_gather = dist.all_gather_into_tensor

    def trace_gather(output, packet, *, group):
        print("GLM_GATE_COLLECTIVE_START", rank, dist.get_process_group_ranks(group), packet.numel(), flush=True)
        result = original_gather(output, packet, group=group)
        print("GLM_GATE_COLLECTIVE_DONE", rank, dist.get_process_group_ranks(group), flush=True)
        return result

    dist.all_gather_into_tensor = trace_gather
    with set_current_vllm_config(VllmConfig()):
        group = GroupCoordinatorPatch([[0, 2, 4], [1, 3, 5]], rank, "hccl", True, group_name="pp")
    dist.all_gather_into_tensor = original_gather
    assert group.use_cpu_custom_send_recv
    assert group.device_communicator.supports_tensor_dict
    generator = torch.Generator().manual_seed(149)
    rows = []
    held = []
    for tokens in (1, 17, 640):
        expected = {
            "hidden_states": torch.randn(tokens, 4096, generator=generator, dtype=torch.float16),
            "residual": torch.randn(tokens, 4, 4096, generator=generator),
            "post": torch.randn(tokens, 4, 1, generator=generator),
            "comb": torch.randn(tokens, 4, 4, generator=generator),
            "empty": torch.empty(tokens, 0),
            "control": {"tokens": tokens},
            "none": None,
        }
        expected["residual"].reshape(-1)[:2] = torch.tensor([-0.0, float("nan")])
        if group.rank_in_group == 0:
            packet = {
                key: value.to("npu") if isinstance(value, torch.Tensor) else value for key, value in expected.items()
            }
        else:
            packet, handles, postprocess = group.irecv_tensor_dict()
            assert handles == [] and postprocess == []
        if group.rank_in_group < 2:
            assert group.isend_tensor_dict(packet) == []
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                actual = packet[key].cpu().contiguous().reshape(-1).view(torch.int8)
                assert torch.equal(actual, value.contiguous().reshape(-1).view(torch.int8)), (rank, tokens, key)
            else:
                assert packet[key] == value
        rows.append({"tokens": tokens, "fields": list(packet), "byte_exact": True})
        held.append(packet)
    # All previous packets remain valid after subsequent receives.
    assert len({packet["hidden_states"].data_ptr() for packet in held}) == len(held)
    dist.barrier()
    output = Path("/srv/ai/storage/artifacts/glm-card-profiles-20261010")
    (output / f"coordinator-rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "passed": True,
                "transport": "pairwise_int8_hccl_allgather",
                "cases": rows,
                "scope": "Actual vLLM coordinator, byte-exact packets; not GLM forward",
            },
            indent=2,
        )
        + "\n"
    )
    print("GLM_COORDINATOR_BYTE_GATE_PASS", rank, flush=True)
    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
