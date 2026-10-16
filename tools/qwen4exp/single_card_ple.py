# SPDX-License-Identifier: Apache-2.0
"""Test the existing shared PLE RAM table with the frozen TP2 resident API.

Only row gathering changes. Hashing, pinned staging, DMA and projection retain
their original implementations. Each forward restores its temporary binding;
resident rollback therefore restores the complete original path.
"""

import argparse
import inspect
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.qwen4exp import ple_shared_gather
from tools.sidecard.qfn_resident_server import QfnResidentClient

FORWARD_SHA256 = "d4c6505238425ce3f43d9792a7e1afc07aa1019578dbff864e84fcd6d3d217d3"


def build_forward(original, metadata, gather_type, binding_type):
    owners = {}

    def forward(self, *args, **kwargs):
        self._ensure_ple_method(args[0].device)
        method = self.ple.ple_method
        key = id(method)
        if key not in owners:
            gather = gather_type(method, metadata)
            binding = binding_type(method, gather, verify_calls=8)
            owners[key] = binding
        binding = owners[key]
        if binding.failed:
            raise RuntimeError("shared PLE shadow mismatch; restore baseline before continuing")
        binding.install()
        try:
            return original(self, *args, **kwargs)
        finally:
            binding.restore()

    return forward, owners


FACTORY = """
def replacements():
    from vllm_ascend.models.qwen4_exp.model import _PLEInjection
    from vllm_ascend._310p.worker_310p import NPUWorker310
    original = _PLEInjection._forward_eager
    actual = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest()
    if actual != FORWARD_SHA256:
        raise ValueError("requires the exact frozen TP2 PLE forward")
    forward, owners = build_forward(original, METADATA, SharedPLEGather, SharedPLEBinding)
    status_base = NPUWorker310.resident_status
    def status(worker):
        result = status_base(worker)
        result["single_card_shared_ple"] = [owner.status() for owner in owners.values()]
        return result
    return {
        "vllm_ascend.models.qwen4_exp.model:_PLEInjection._forward_eager": forward,
        "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status,
    }
"""


def candidate_source(metadata):
    return (
        "import hashlib, inspect, textwrap\n"
        + inspect.getsource(ple_shared_gather)
        + "\n"
        + inspect.getsource(build_forward)
        + f"\nMETADATA = {metadata!r}\nFORWARD_SHA256 = {FORWARD_SHA256!r}\n"
        + FACTORY
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--run-root", type=Path, help="authenticated custom-collective run directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing receipts")
    metadata = json.loads(args.table.read_text())
    source = candidate_source(metadata)
    if args.run_root is None:
        client = ResidentClient(args.base_url, expected_workers=2)
    else:
        client = QfnResidentClient(args.base_url, args.run_root / "control.token", expected_workers=2)
    result = dict(source=source, before=client.rpc("resident_status"))
    try:
        result["after"] = client.switch(Control(uuid.uuid4().hex, candidate="single_card_shared_ple", source=source))
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Shared RAM gather installed; inspect both worker shadows before timing.")


if __name__ == "__main__":
    main()
