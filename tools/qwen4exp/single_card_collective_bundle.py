# SPDX-License-Identifier: Apache-2.0
"""Prepare an isolated frozen TP2 runtime with qualified collective hooks.

The original corrected bundle is verified and left untouched. Only four
plugin files receive collective integration; model, cache, weights and native
operators remain frozen. Tooling is copied separately for the cold controller.
Run the private graph gate on the target two devices before starting serving.
"""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from tools.qwen4exp.single_card_kv_fix import prepare

FROZEN = {
    "ascend_config.py": "3e5fd45d8fa9d32220ca37d155ae9b134f670bd60de99ea3c3d928b6d8418f92",
    "compilation/acl_graph.py": "8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955",
    "compilation/breakable_aclgraph.py": "782cc189bb192eb7e934a3e5ae3e53e4e7a80348e2db1ce685315407f79a29ee",
    "distributed/device_communicators/npu_communicator.py": (
        "90aaa39c4d53caecb5493ba00a45d953e89992cff5d2d5996388781aa302d5a8"
    ),
}
CURRENT_COMMUNICATOR = "15afd35098b4ad87a823b99c6eedf8f0ed46084d169054b856d32b75818f96c6"
CURRENT_GRAPHS = {
    "compilation/acl_graph.py": "9374fd84e3caf10f5dc5e83e1b1bd7f7849e7dc17f7662189a5c99e9115d880b",
    "compilation/breakable_aclgraph.py": "b7716081724a0240033228a14ee985ef25e7c782bbfae93319542bcfaa37a115",
}


def communicator_source(frozen, current):
    # Import the reviewed collective block, excluding unrelated PP transport.
    start, end = "        self.custom_sum = None\n", "        self.pipeline_transport = None\n"
    if current.count(start) != 1 or current.count(end) != 1:
        raise ValueError("collective initialization seams changed")
    block = current[current.index(start) : current.index(end)]
    source = frozen.replace(
        "from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase",
        "from vllm.config import get_current_vllm_config_or_none\n"
        "from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase",
    )
    if not source.endswith("        self.all2all_manager = _NpuAll2AllManager()\n"):
        raise ValueError("frozen communicator seam changed")
    source += block
    cls = next(n for n in ast.parse(current).body if isinstance(n, ast.ClassDef) and n.name == "NPUCommunicator")
    for name in ("all_reduce", "close_custom_collective"):
        node = next(n for n in cls.body if getattr(n, "name", None) == name)
        source += "\n" + "".join(current.splitlines(True)[node.lineno - 1 : node.end_lineno])
    source += textwrap.indent(
        "\n\ndef destroy(self):\n    self.close_custom_collective()\n    super().destroy()\n", "    "
    )
    compile(source, "<tp2-communicator>", "exec")
    return source


def config_source(frozen):
    assignment = (
        "    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}\n"
    )
    key = '        "enable_flashcomm1",\n'
    if frozen.count(assignment) != 1 or frozen.count(key) != 1:
        raise ValueError("frozen collective config seams changed")
    return frozen.replace(
        assignment,
        assignment
        + '    npu_collective = additional_config.get("npu_collective")\n'
        + "    if npu_collective is not None and type(npu_collective) is not dict:\n"
        + '        raise ValueError("additional_config.npu_collective must be a dict or None.")\n',
    ).replace(key, key + '        "npu_collective",  # Validated by the TP communicator.\n')


def prepare_collective(bundle, output, repo):
    if output.exists():
        raise ValueError("use a fresh destination; preserve prior collective owners")
    receipt = prepare(bundle, check=True)
    replacements = {}
    for relative, expected in FROZEN.items():
        frozen = (bundle / "runtime/vllm_ascend" / relative).read_text()
        current = (repo / "vllm_ascend" / relative).read_text()
        if hashlib.sha256(frozen.encode()).hexdigest() != expected:
            raise ValueError(f"frozen source changed: {relative}")
        if relative == "ascend_config.py":
            replacements[relative] = config_source(frozen)
        elif relative in CURRENT_GRAPHS:
            if hashlib.sha256(current.encode()).hexdigest() != CURRENT_GRAPHS[relative]:
                raise ValueError(f"graph lifecycle source changed: {relative}")
            replacements[relative] = current
        else:
            if hashlib.sha256(current.encode()).hexdigest() != CURRENT_COMMUNICATOR:
                raise ValueError("collective communicator source changed")
            replacements[relative] = communicator_source(frozen, current)
    output.mkdir(parents=True, mode=0o700)
    destination = output / "bundle"
    subprocess.run(["cp", "--reflink=auto", "-a", str(bundle), str(destination)], check=True)
    for relative, source in replacements.items():
        (destination / "runtime/vllm_ascend" / relative).write_text(source)
    # A separate first PYTHONPATH entry supplies current lifecycle/controller
    # tools; do not replace model or upstream packages to satisfy imports.
    tooling = output / "tooling"
    tooling.mkdir()
    shutil.copytree(repo / "tools", tooling / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Freeze tooling resolution ahead of inherited SDK/controller paths, which
    # may contain another regular package named tools.
    (tooling / "tools/__init__.py").write_text(
        '# SPDX-License-Identifier: Apache-2.0\n"""Isolated collective experiment tooling."""\n'
    )
    receipt.update(
        bundle=str(destination),
        tooling=str(tooling),
        replacements={name: hashlib.sha256(value.encode()).hexdigest() for name, value in replacements.items()},
        original_manifest_preserved=True,
        scope="isolated collective experiment; private graph and real serving gates required",
    )
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_collective(args.bundle.resolve(), args.output.resolve(), Path(__file__).resolve().parents[2]),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
