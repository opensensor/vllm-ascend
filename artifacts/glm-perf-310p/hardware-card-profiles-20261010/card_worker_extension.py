# SPDX-License-Identifier: Apache-2.0
"""Load the same auxiliary native bindings used by the qualified TP4 runtime."""

import hashlib
import json
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
    pass
