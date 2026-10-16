# SPDX-License-Identifier: Apache-2.0
"""Explicit candidate admission against an already authenticated bank owner."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .build import CLIPPED_SOURCE, HELPER_PACKAGE, ORIGINAL_PROVENANCE, contained, digest
from .contract import EXPERTS, TOP_K, WIDTH

BRIDGE_SHA256 = "620c86ed751450b9ac2d134304ab5f8769ead81a99aacd5276fc0200ff83f4c4"


def authenticate_candidate(root, manifest_sha256):
    root = Path(root).resolve()
    receipt = root / "multi-route-build.json"
    if digest(receipt) != manifest_sha256:
        raise ValueError("candidate build receipt differs")
    report = json.loads(receipt.read_bytes())
    if (
        report.get("schema_version") != 1
        or report.get("candidate") != "multi-route-r1-fixed-slots-v1"
        or report.get("original_provenance_sha256") != ORIGINAL_PROVENANCE
        or report.get("clipped_source_sha256") != CLIPPED_SOURCE
        or report.get("build_complete") is not True
        or report.get("bridge_namespace") != "glm_reconstruction_v1140"
        or report.get("bridge_binary_sha256") != BRIDGE_SHA256
        or report.get("swiglu_limit") != 10.0
        or report.get("activation_bits") != 4
        or report.get("rows") != 1
        or report.get("top_k") != TOP_K
    ):
        raise ValueError("candidate math/ABI/provenance differs")
    files = dict(report["source_sha256"])
    files.update(report["binaries"])
    files["candidate.patch"] = report["patch_sha256"]
    files["provenance.json"] = report["provenance_sha256"]
    files["glm_reconstruction_bridge_v1140.so"] = report["bridge_binary_sha256"]
    expected_binaries = {
        "glm_fused_pack.bin",
        "glm_fused_route_input.bin",
        "glm_fused_reduce.bin",
        "glm_fused_gate_up.bin",
        "glm_fused_gate_up_w4.bin",
        "glm_fused_down.bin",
        "glm_fused_down_w4.bin",
    }
    if set(report["binaries"]) != expected_binaries:
        raise ValueError("candidate stage binary set differs")
    for name, checksum in files.items():
        if digest(contained(root, name)) != checksum:
            raise ValueError("candidate file changed: " + name)
    provenance = json.loads((root / "provenance.json").read_bytes())
    if (
        provenance["_build"]["helper_package"] != HELPER_PACKAGE
        or provenance["_build"].get("namespace") != "glm_reconstruction_v1140"
        or provenance["_build"].get("version") != 1140
    ):
        raise ValueError("candidate helper namespace differs")
    if not all(
        provenance["_helpers"].get(Path(name).name) == checksum
        for name, checksum in report["source_sha256"].items()
        if name.startswith(HELPER_PACKAGE + "/")
    ):
        raise ValueError("candidate helper receipts disagree")
    return report, provenance


def load_helper(root):
    """Owner has loaded the frozen bridge; avoid registering a second .so copy."""
    root = Path(root).resolve()
    package_root = root / HELPER_PACKAGE
    if HELPER_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            HELPER_PACKAGE, package_root / "__init__.py", submodule_search_locations=[str(package_root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[HELPER_PACKAGE] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[HELPER_PACKAGE].__file__).resolve() != package_root / "__init__.py":
        raise ValueError("candidate helper belongs to another build")
    return importlib.import_module(HELPER_PACKAGE + ".glm_fused_moe")


class MultiRouteDecode:
    """One-token native path using the owner's immutable prepared bank tensors.

    No route-count readback, zero-route shortcut, collectives, or implicit graph
    capture. Every eligible R1 input runs the fixed-slot reducer. Caller owns
    stream/workspace lifetime; overlapping calls require distinct instances.
    Prefill and unsupported shapes retain the qualified actor fallback.
    """

    def __init__(self, actor, root, manifest_sha256):
        report, provenance = authenticate_candidate(root, manifest_sha256)
        if actor.rank not in (0, 1) or actor.stable_rows != 17 or actor.reduce_results:
            raise ValueError("candidate requires unreduced qualified 17-row bank owner")
        if (
            actor.native.__class__.__module__ != "glm_reconstruction_v1140_helpers.glm_fused_moe"
            or actor.native.activation_bits != 4
            or not actor.native.prepared_weight_layout
            or not actor.native.fp16_weight_scales
            or not actor.native.fp16_route_workspace
        ):
            raise ValueError("owner must have loaded the authenticated original helper/bridge")
        self.actor, self.torch, self.root = actor, actor.torch, Path(root).resolve()
        self.report = report
        helper = load_helper(self.root)
        if helper.FUSED_REDUCTION_TOKENS != 0:
            raise ValueError("candidate helper retained padded threshold")
        options = provenance["_build"]
        self.native = helper.NativeFusedMoE(
            self.root,
            namespace=options["namespace"],
            activation_bits=4,
            prepared_weight_layout=True,
            fp16_route_workspace=True,
            shared_prefill_max_tokens=1,
        )

    def __call__(self, layer, hidden, ids, weights):
        if hidden.shape != (1, WIDTH) or ids.shape != (1, TOP_K) or weights.shape != ids.shape:
            return self.actor(layer, hidden, ids, weights)
        t = self.torch
        owned = (ids >= self.actor.rank * EXPERTS) & (ids < (self.actor.rank + 1) * EXPERTS)
        local_ids = t.where(owned, ids, self.actor.rank * EXPERTS).to(t.int32).contiguous()
        local_weights = t.where(owned, weights, 0).float().contiguous()
        return self.native(hidden, *self.actor.banks[layer], local_weights, local_ids, self.actor.rank * EXPERTS)
