# SPDX-License-Identifier: Apache-2.0
"""Convert selected dense projections while preserving resident expert shards."""

import argparse
import dataclasses
import fcntl
import inspect
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient


def release_conversion_metadata(native_resources):
    """Release rebuildable W4A8 metadata after draining and dropping graphs."""
    # Worker-only import; the controller must not initialize an NPU context.
    import torch

    torch.npu.synchronize()
    for kernel in native_resources.values():
        if getattr(kernel, "requires_int8_layout", False):
            kernel.banks.clear()
            kernel.scales.clear()
    torch.npu.empty_cache()


def conversion_replacements(enable, native_resources):
    from tools.minimax_m3.dense_int8 import convert, restore
    from tools.minimax_m3.resident_dense_int8 import release_conversion_metadata

    def capture(self):
        self._m3_idle()
        state = self.__dict__.setdefault("_m3_dense_int8", {})
        try:
            release_conversion_metadata(native_resources)
            reports = convert(self.model_runner.model, state) if enable else restore(state)
            result = {"passed": True, "enabled": enable, "projections": reports or []}
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        return {**self.resident_status(), "dense_conversion": result}

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture": capture}


def mutate(client, enable):
    source = inspect.getsource(conversion_replacements) + (
        f"\ndef replacements(native_resources):\n    return conversion_replacements({enable!r}, native_resources)\n"
    )
    control = Control(uuid.uuid4().hex, "direct-target", "dense_int8_conversion", source)
    client.rpc("resident_prepare", json.dumps(dataclasses.asdict(control)))
    # Entering the previous generation's apply releases its captured graphs.
    client.rpc("resident_apply", control.generation)
    return client.rpc("resident_capture")


def transition(client, source, enable):
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise ValueError("refusing to take ownership of an existing pause")
    before = client.rpc("resident_status")
    expected = Control(before[0]["generation"], "direct-target", before[0]["candidate"], source)
    if any(row["digest"] != expected.digest or row["native_failed"] for row in before):
        raise ValueError("serving source must match every healthy resident generation")
    client.request("/pause?mode=wait&clear_cache=true")
    result = mutate(client, enable)
    if not all(row["dense_conversion"]["passed"] for row in result):
        # Successful workers must also recover if another rank failed conversion.
        recovery = mutate(client, False)
        if not all(row["dense_conversion"]["passed"] for row in recovery):
            raise RuntimeError(f"dense recovery failed; server remains paused: {recovery}")
        client.switch(Control(uuid.uuid4().hex, "direct-target", "m3_candidate", source))
        client.resume()
        raise RuntimeError(f"dense conversion failed and recovered: {result}")
    after = client.switch(Control(uuid.uuid4().hex, "direct-target", "m3_candidate", source))
    if [(r["rank"], r["pid"]) for r in before] != [(r["rank"], r["pid"]) for r in after]:
        raise RuntimeError("workers changed during conversion; server remains paused")
    client.resume()
    return {"before": before, "after": after, "conversion": result, "enabled": enable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enable", "disable"))
    parser.add_argument("--serving-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    client = ResidentClient(args.url, 6)
    with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = client.request("/resident/maintenance", method="GET")["maintenance"]
        client.request("/resident/maintenance")
        try:
            result = transition(client, args.serving_source.read_text(), args.action == "enable")
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if not previous and not client.request("/is_paused", method="GET")["is_paused"]:
                client.request("/resident/maintenance", method="DELETE")


if __name__ == "__main__":
    main()
