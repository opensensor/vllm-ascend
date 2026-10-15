# SPDX-License-Identifier: Apache-2.0
"""Cold admission and graph-safe owner for exact device PLE request selection."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tools.qwen4exp.ple_storage import integer

HEADS = 16
ROW_BYTES = 320
MAX_TOKENS = 2048
MAX_REQUESTS = 4
SIGNED_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class PLERequestGeometry:
    tokens: int
    requests: int
    eos: int
    vocabulary: int
    multipliers: tuple[int, ...]
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    table_rows: int

    def __post_init__(self):
        for value in (self.tokens, self.requests, self.eos, self.vocabulary, self.table_rows):
            integer(value)
        if not 1 <= self.tokens <= MAX_TOKENS or not 1 <= self.requests <= MAX_REQUESTS:
            raise ValueError("PLE request exceeds bounded token or request capacity")
        if not 0 <= self.eos < self.vocabulary <= SIGNED_MAX or not 1 <= self.table_rows <= SIGNED_MAX // ROW_BYTES:
            raise ValueError("PLE vocabulary, EOS, or table address range is invalid")
        for name in ("multipliers", "sizes", "offsets"):
            object.__setattr__(self, name, tuple(integer(value) for value in getattr(self, name)))
        if len(self.multipliers) != 3 or len(self.sizes) != HEADS or len(self.offsets) != HEADS:
            raise ValueError("PLE kernel requires order 3 and eight heads per order")
        if any(value < 1 or value > SIGNED_MAX // self.vocabulary for value in self.multipliers):
            raise ValueError("PLE multiplier can overflow signed model hash arithmetic")
        end = 0
        for size, offset in zip(self.sizes, self.offsets):
            if size < 1 or offset != end:
                raise ValueError("PLE head spans must be positive and contiguous")
            end += size
        if end > self.table_rows:
            raise ValueError("PLE head span exceeds admitted table")

    def values(self):
        return (self.tokens, self.requests, self.eos, self.vocabulary, *self.multipliers, *self.sizes, *self.offsets)


def load(build):
    # Framework import is lazy so manifest/admission checks remain host-only.
    import torch

    root = Path(build).resolve(strict=True)
    manifest = json.loads((root / "provenance.json").read_text())
    if (
        manifest["entry"] != "qwen_ple_request_v1"
        or manifest["namespace"] != "qwen_checkpoint_preparation_v1"
        or manifest["flush_host_queue"] is not False
    ):
        raise ValueError("PLE native build contract differs")
    for name, expected in manifest["sources"].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("PLE native source provenance differs")
    for key in ("bridge", "binary"):
        entry = manifest[key]
        path = Path(entry["path"]).resolve(strict=True)
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("PLE native binary provenance differs")
    torch.ops.load_library(manifest["bridge"]["path"])
    namespace = manifest["namespace"]
    kernel = getattr(torch.classes, namespace).Kernel(manifest["binary"]["path"], manifest["entry"])
    return kernel, getattr(torch.ops, namespace).launch


class PLEDeviceRequest:
    """Stable device buffers, no token readback, host hashing, or runtime allocation.

    Receipt[:, 0] is 0 on success, 1 for invalid packed boundaries and 2 for
    invalid token/history IDs. Invalid output rows and offsets are -1; a future
    DMA consumer MUST inspect the device receipt before submitting any request.
    This owner neither submits DMA nor replaces the live model lookup.
    """

    def __init__(self, kernel, launch, geometry, device):
        import torch  # Lazy framework initialization in the owning worker only.

        device = torch.device(device)
        if device.type != "npu" or torch.npu.is_current_stream_capturing():
            raise ValueError("admit PLE device buffers on an NPU before capture")
        self.kernel, self.launch, self.geometry = kernel, launch, geometry
        self.config = torch.tensor(geometry.values(), dtype=torch.int64, device=device)
        self.rows = torch.empty((geometry.tokens, HEADS), dtype=torch.int64, device=device)
        self.offsets = torch.empty_like(self.rows)
        self.receipt = torch.empty((geometry.tokens, 8), dtype=torch.int32, device=device)

    def __call__(self, tokens, starts, history):
        cfg = self.geometry
        for tensor, shape, dtype in (
            (tokens, (cfg.tokens,), self.rows.dtype),
            (starts, (cfg.requests + 1,), self.receipt.dtype),
            (history, (cfg.requests, 2), self.rows.dtype),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.rows.device
                or not tensor.is_contiguous()
            ):
                raise ValueError("PLE input shape, dtype, device, or contiguity differs from admission")
        self.launch(
            self.kernel,
            [tokens, starts, history, self.config, self.rows, self.offsets, self.receipt],
            min(8, cfg.tokens),
        )
        return self.rows, self.offsets, self.receipt
