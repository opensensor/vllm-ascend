# SPDX-License-Identifier: Apache-2.0
"""Load the same auxiliary native bindings used by the qualified TP4 runtime."""

import hashlib
import json
import os
from pathlib import Path

import torch

from vllm_ascend.utils import enable_custom_op

root = Path(__file__).with_name("card-bindings")
manifest = json.loads((root / "manifest.json").read_text())
enable_custom_op()
for record in manifest:
    library = root / record["file"]
    if hashlib.sha256(library.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"auxiliary binding checksum differs: {library}")
    if not hasattr(torch.ops._C_ascend, record["operator"]):
        torch.ops.load_library(str(library))
    if not hasattr(torch.ops._C_ascend, record["operator"]):
        raise RuntimeError(f"auxiliary operator was not registered: {record['operator']}")


class CardWorkerExtension:
    def card_start_replay_count(self):
        """Count actual stage graph replays without synchronizing the NPU."""
        if getattr(self, "_card_replay_count", None) is not None:
            return self.card_replay_status()
        model = self.model_runner.model
        original = getattr(model, "_replay", None)
        if original is None:
            return {"rank": self.rank, "error": "model has no stage graph replay method"}
        state = {"count": 0, "original": original}

        def counted(*args, **kwargs):
            result = original(*args, **kwargs)
            state["count"] += 1
            return result

        model._replay = counted
        self._card_replay_count = state
        return self.card_replay_status()

    def card_replay_status(self):
        state = getattr(self, "_card_replay_count", None)
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "replays": state["count"] if state is not None else None,
            "wrapper": type(self.model_runner.model).__name__,
        }

    def card_stop_replay_count(self):
        result = self.card_replay_status()
        state = getattr(self, "_card_replay_count", None)
        if state is not None:
            self.model_runner.model._replay = state["original"]
            self._card_replay_count = None
        return result
