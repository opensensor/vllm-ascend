# SPDX-License-Identifier: Apache-2.0
"""Change operator determinism after draining requests and releasing graphs."""

import argparse
import dataclasses
import fcntl
import inspect
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient


def flag_replacements(enabled):
    # Worker-only import. The controller never initializes an NPU context.
    import torch

    def capture(self):
        self._m3_idle()
        previous = {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        try:
            torch.npu.synchronize()
            # Match Ascend's existing deterministic-operator setting. Unsupported
            # operators warn; this is not a promise of universal batch invariance.
            torch.use_deterministic_algorithms(enabled, warn_only=enabled)
            current = {
                "enabled": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            }
            result = {"passed": current == {"enabled": enabled, "warn_only": enabled}, "current": current}
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        return {**self.resident_status(), "determinism": {"previous": previous, **result}}

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture": capture}


def transition(client, source, enabled):
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise ValueError("refusing to take ownership of an existing pause")
    before = client.rpc("resident_status")
    expected = Control(before[0]["generation"], "direct-target", before[0]["candidate"], source)
    if any(row["digest"] != expected.digest or row["native_failed"] for row in before):
        raise ValueError("serving source must match every healthy resident generation")
    client.request("/pause?mode=wait&clear_cache=true")
    hook = inspect.getsource(flag_replacements) + (
        f"\ndef replacements(native_resources):\n    return flag_replacements({enabled!r})\n"
    )
    control = Control(uuid.uuid4().hex, "direct-target", "m3_determinism", hook)
    client.rpc("resident_prepare", json.dumps(dataclasses.asdict(control)))
    # The previous generation's apply hook releases its graph captures first.
    client.rpc("resident_apply", control.generation)
    result = client.rpc("resident_capture")
    if not all(row["determinism"]["passed"] for row in result):
        raise RuntimeError(f"determinism transition failed; server remains paused: {result}")
    candidate = "m3_candidate" if source else "baseline"
    after = client.switch(Control(uuid.uuid4().hex, "direct-target", candidate, source))
    identity = lambda rows: {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in rows}
    if identity(before) != identity(after):
        raise RuntimeError("resident weights or workers changed; server remains paused")
    client.resume()
    return {"before": before, "after": after, "flags": result, "enabled": enabled}


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
