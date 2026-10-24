# SPDX-License-Identifier: Apache-2.0
"""Reload compatible Python/CANN implementations on the six resident workers."""

import argparse
import fcntl
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest


def candidate_source(module_source=None, native_name=None):
    """Compose a complete generation; switching restores the prior generation."""
    if module_source is None and native_name is None:
        return ""
    source = "import types\n\ndef replacements(native_resources):\n    changes = {}\n"
    if module_source is not None:
        source += (
            "    module = types.ModuleType('minimax_resident_candidate')\n"
            f"    exec(compile({module_source!r}, '<minimax_ops>', 'exec'), module.__dict__)\n"
            "    for name in ('route_experts', 'clipped_swiglu', 'paged_sparse_attention', 'grouped_w4_mm'):\n"
            "        changes['vllm_ascend._310p.ops.minimax_m3:' + name] = getattr(module, name)\n"
            "    changes['vllm_ascend._310p.ops.minimax_m3:PackedM3Experts.forward'] = module.PackedM3Experts.forward\n"
            "    for name in ('clipped_swiglu', 'paged_sparse_attention'):\n"
            "        changes['vllm_ascend.models.minimax_m3.minimax_m3:' + name] = getattr(module, name)\n"
        )
    if native_name is not None:
        source += (
            f"    kernel = native_resources[{native_name!r}]\n"
            "    def project(x, codes, scales, ends):\n        return kernel(x, codes, scales, ends)\n"
            "    changes['vllm_ascend._310p.ops.minimax_m3:grouped_w4_mm'] = project\n"
        )
        if module_source is not None:
            source += "    module.grouped_w4_mm = project\n"
    return source + "    return changes\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "switch", "load-native", "resume"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--module-file", type=Path)
    parser.add_argument("--native-name")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    client = ResidentClient(args.url, args.workers)
    # Run this controller on the serving host; the HTTP control routes are local.
    with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "status":
            result = client.rpc("resident_status")
        else:
            previous = client.request("/resident/maintenance", method="GET")["maintenance"]
            client.request("/resident/maintenance", method="POST")
            try:
                if args.action == "switch":
                    source = candidate_source(
                        args.module_file.read_text() if args.module_file else None, args.native_name
                    )
                    result = client.switch(
                        Control(uuid.uuid4().hex, "direct-target", "m3_candidate" if source else "baseline", source)
                    )
                elif args.action == "load-native":
                    if args.manifest is None:
                        raise ValueError("load-native requires --manifest")
                    result = client.load_native(NativeManifest(json.loads(args.manifest.read_text())))
                else:
                    result = client.resume()
            finally:
                # Failed mutations keep the server paused and public traffic out.
                if not previous and not client.request("/is_paused", method="GET")["is_paused"]:
                    client.request("/resident/maintenance", method="DELETE")
        report = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(report)
        print(report)


if __name__ == "__main__":
    main()
