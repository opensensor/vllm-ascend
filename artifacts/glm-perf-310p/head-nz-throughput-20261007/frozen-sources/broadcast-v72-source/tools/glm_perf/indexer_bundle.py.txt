# SPDX-License-Identifier: Apache-2.0
"""Load a verified permanent AI-Core indexer bundle through model composition."""

import importlib.util
import json
from pathlib import Path
from types import FunctionType, MethodType

import torch

from .resident_native import NativeManifest, file_digest


def read_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(model, bundle):
    from vllm_ascend.models.glm5next.attention import Indexer
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    manifest = NativeManifest(json.loads((bundle / "manifest.json").read_text()))
    manifest.verify_files()
    options = json.loads((bundle / "options.json").read_text())
    namespace = options["namespace"]
    if manifest.value["operators"] != [namespace + "::launch"]:
        raise ValueError("indexer bundle namespace and manifest differ")
    for entry in manifest.value["libraries"]:
        # Hot serving may already own the same binary from its build path.
        # Loading a copied .so registers its TORCH_LIBRARY twice. Check the
        # bytes of loaded libraries rather than accepting a namespace alone.
        existing = any(
            Path(path).is_file() and file_digest(Path(path)) == entry["sha256"] for path in torch.ops.loaded_libraries
        )
        if not existing:
            torch.ops.load_library(entry["path"])
    helper = read_module(bundle / "bf16_cast.py", namespace + "_permanent_helper")
    candidate = read_module(bundle / "candidate.py", namespace + "_permanent_candidate")
    convert = helper.NativeBF16Cast(bundle, namespace)
    changes = candidate.replacements({"bf16_cast_v1": convert})
    forward = changes["vllm_ascend.models.glm5next.attention:Indexer.forward"]
    writer = changes["vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._write_pools"]
    score = changes["vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:score_kpool_paged"]
    # Only these loaded model instances use the qualified functions. Preserve
    # existing selector arithmetic and its fallback, replacing the score binding.
    original = SparseAttnIndexerKpool._select_tokens_fixed
    selector = FunctionType(
        original.__code__,
        dict(original.__globals__, score_kpool_paged=score),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    selector.__kwdefaults__ = original.__kwdefaults__
    count = 0
    for module in model.modules():
        if isinstance(module, Indexer):
            module.forward = MethodType(forward, module)
            module.indexer_op._write_pools = MethodType(writer, module.indexer_op)
            module.indexer_op._select_tokens_fixed = MethodType(selector, module.indexer_op)
            module._native_bf16_cast = convert
            count += 1
    if not count:
        raise ValueError("native indexer bundle found no GLM indexer instances")
    return count
