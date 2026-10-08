"""One-time recovery of workers loaded before resident graph-pool renewal."""
import importlib

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

original_capture = NPUModelRunner.capture_model


def capture_model(self):
    # The failed attempt has already discarded its captures. Refresh only the
    # experimental control module, then renew the stale wrapper pool handles.
    # Existing extension methods share this module dictionary, so later baseline
    # switches also use the corrected clear_graph_state implementation.
    from tools.glm_perf import resident_worker

    control = importlib.reload(resident_worker)
    wrappers = [self.model]
    if self.drafter is not None:
        wrappers.append(self.drafter.model)
    control.renew_graph_pool(wrappers)
    return original_capture(self)


def replacements():
    return {"vllm_ascend.worker.model_runner_v1:NPUModelRunner.capture_model": capture_model}
