from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.resident_worker import QwenResidentExtension

ROOT = Path("/srv/ai/src/qwen-prefill-fairness-20261010")
METHOD = "resident_transfer_profile_v16"


def validate():
    assert torch.ops.qwen_stage_diagnostic_v14.marker() == 1
    assert not hasattr(QwenResidentExtension, METHOD)

    def profile(self, action):
        rank = torch.distributed.get_rank()
        try:
            if action == "start":
                assert not hasattr(self, "_qwen_transfer_profiler_v16")
                self._qwen_transfer_profiler_v16 = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(ROOT / "transfer-profile-v16")),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.Memory,
                        export_type=[torch_npu.profiler.ExportType.Text, torch_npu.profiler.ExportType.Db],
                        op_attr=True,
                    ),
                )
                self._qwen_transfer_profiler_v16.start()
                return {"rank": rank, "passed": True, "action": action}
            if action == "stop":
                profiler = self._qwen_transfer_profiler_v16
                profiler.stop()
                del self._qwen_transfer_profiler_v16
                return {
                    "rank": rank,
                    "passed": True,
                    "action": action,
                    "raw_directory": str(ROOT / "transfer-profile-v16"),
                }
            raise ValueError("unknown action")
        except Exception as error:
            return {"rank": rank, "passed": False, "error": f"{type(error).__name__}: {error}"}

    setattr(QwenResidentExtension, METHOD, profile)
    return {"passed": True, "diagnostic_method": METHOD, "production_forward_changed": False}
