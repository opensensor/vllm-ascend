# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drain every NPU stream before the resident reset mutates KV/Mamba state.

Install on the actual qualified NPUWorker310 class. Its resident extension is
inherited, so baseline restoration removes the override and reveals the same
original method. Calling the extension class directly avoids recursive lookup
through the overridden worker method. No generation or graph replay is changed.
"""

import torch


def resident_reset(self):
    try:
        torch.npu.synchronize()
        # Worker isolation: the qualified extension imports torch_npu-dependent
        # runner components. Keep host preparation free of that initialization.
        from tools.sidecard.qfn_resident_worker import QfnResidentWorkerExtension

        return QfnResidentWorkerExtension.resident_reset(self)
    except Exception as error:
        # Collect every rank's error acknowledgment; no reset follows a failed
        # drain and the controller must retain its pause on any worker error.
        return self._resident_error(error)


def replacements():
    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_reset": resident_reset}
