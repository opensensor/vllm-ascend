# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inventory source-level transfer and barrier candidates without importing models.

This is a coverage aid, not a profiler or reachability proof. In particular,
CPU item()/tolist(), no-op casts, views, and non-device waits need manual review.
Native sites are lexical candidates; library and compiled-kernel internals
remain outside this scanner. Roots should be immutable source snapshots.
"""

import argparse
import ast
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import regex as re

CALL_CATEGORIES = MappingProxyType(
    {
        "host_boundary": frozenset({"cpu", "numpy", "item", "tolist", "tolists", "copy_to_cpu", "copy_to_gpu"}),
        "copy_or_cast": frozenset(
            {"to", "cuda", "npu", "float", "half", "double", "bfloat16", "type", "type_as", "copy_"}
        ),
        "materialization": frozenset(
            {
                "clone",
                "contiguous",
                "cat",
                "stack",
                "repeat",
                "repeat_interleave",
                "index_select",
                "gather",
                "scatter",
                "scatter_",
                "index_copy_",
                "index_add_",
                "npu_format_cast",
                "npu_format_cast_",
                "maybe_trans_nz",
            }
        ),
        "dynamic_shape": frozenset({"nonzero", "masked_select", "unique", "unique_consecutive", "bincount"}),
        "allocation_or_fill": frozenset(
            {
                "empty",
                "empty_like",
                "new_empty",
                "zeros",
                "zeros_like",
                "new_zeros",
                "ones",
                "ones_like",
                "full",
                "full_like",
                "tensor",
                "as_tensor",
                "arange",
                "empty_with_format",
                "zero_",
                "fill_",
                "pin_memory",
            }
        ),
        "barrier_or_stream": frozenset(
            {
                "synchronize",
                "barrier",
                "wait",
                "wait_stream",
                "wait_event",
                "record_event",
                "record_stream",
                "synchronize_input_prep",
                "add_eager",
                "replay",
                "capture_begin",
                "capture_end",
            }
        ),
        "collective": frozenset(
            {
                "all_reduce",
                "_tp_reduce",
                "tensor_model_parallel_all_reduce",
                "all_gather",
                "all_gather_into_tensor",
                "all_gather_single",
                "tensor_model_parallel_all_gather",
                "reduce_scatter",
                "reduce_scatter_tensor",
                "broadcast",
                "all_to_all",
                "all_to_all_single",
                "send",
                "recv",
                "isend",
                "irecv",
            }
        ),
    }
)
NATIVE_CATEGORIES = MappingProxyType(
    {
        "native_copy": (
            r"(?:aclrtMemcpy(?:Async)?|memcpy|memmove|DataCopy(?:Pad)?|CopyIn|CopyOut|"
            r"LoadData(?:WithTranspose)?|TensorCopy)"
        ),
        "native_barrier": (
            r"(?:aclrtSynchronize(?:Device|Stream|Event)|PipeBarrier|SetFlag|WaitFlag|SyncAll|"
            r"CrossCoreSetFlag|CrossCoreWaitFlag|ib_set|ib_wait)"
        ),
        "native_collective": r"(?:hcclAllReduce|hcclAllGather|hcclReduceScatter|hcclAlltoAll|hcclBroadcast)",
    }
)
NATIVE_SUFFIXES = frozenset({".cpp", ".cc", ".c", ".h", ".hpp", ".cuh"})
MAX_EXCERPT = 240
NATIVE_NONCODE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*', re.DOTALL)


def python_sites(source, filename):
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines()
    sites = []

    def visit(node, scope):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            categories = [category for category, names in CALL_CATEGORIES.items() if name in names]
            if categories:
                sites.append(
                    {
                        "line": node.lineno,
                        "column": node.col_offset,
                        "scope": ".".join(scope),
                        "operation": name,
                        "categories": categories,
                        "excerpt": lines[node.lineno - 1].strip()[:MAX_EXCERPT],
                    }
                )
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return sorted(sites, key=lambda site: (site["line"], site["column"]))


def native_sites(source):
    # Preserve line/column positions while removing strings and comments.
    code = NATIVE_NONCODE.sub(lambda match: "".join("\n" if char == "\n" else " " for char in match.group()), source)
    lines = source.splitlines()
    sites = []
    for category, names in NATIVE_CATEGORIES.items():
        pattern = re.compile(r"\b(" + names + r")\s*(?:<[^;{}]*>)?\s*\(")
        for match in pattern.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            column = match.start() - code.rfind("\n", 0, match.start()) - 1
            sites.append(
                {
                    "line": line,
                    "column": column,
                    "scope": "lexical_native",
                    "operation": match.group(1),
                    "categories": [category],
                    "excerpt": lines[line - 1].strip()[:MAX_EXCERPT],
                }
            )
    return sorted(sites, key=lambda site: (site["line"], site["column"]))


def scan_root(root, label):
    manifest, sites, errors = [], [], []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts)
        ):
            continue
        if path.suffix != ".py" and path.suffix not in NATIVE_SUFFIXES:
            continue
        relative = str(path.relative_to(root))
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        manifest.append({"root": label, "path": relative, "sha256": digest, "bytes": len(raw)})
        try:
            source = raw.decode("utf-8")
            found = python_sites(source, relative) if path.suffix == ".py" else native_sites(source)
        except (UnicodeError, SyntaxError, RecursionError) as error:
            errors.append({"root": label, "path": relative, "error": str(error)})
            continue
        sites.extend({"root": label, "path": relative, "sha256": digest, **site} for site in found)
    return manifest, sites, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {}
    for value in args.root:
        label, separator, path = value.partition("=")
        if not separator or not label or label in roots or not Path(path).is_dir():
            parser.error("each root must have a unique label and an existing directory")
        roots[label] = Path(path)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, sites, errors = [], [], []
    for label, root in roots.items():
        files, found, failures = scan_root(root, label)
        manifest.extend(files)
        sites.extend(found)
        errors.extend(failures)
    with gzip.open(args.output / "inventory.jsonl.gz", "wt", encoding="utf-8") as output:
        for site in sites:
            output.write(json.dumps(site, ensure_ascii=False) + "\n")
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "static_candidates_only": True,
        "roots": {label: str(root) for label, root in roots.items()},
        "files": len(manifest),
        "sites": len(sites),
        "categories": dict(Counter(category for site in sites for category in site["categories"])),
        "by_root": {
            label: {
                "files": sum(file["root"] == label for file in manifest),
                "sites": sum(site["root"] == label for site in sites),
            }
            for label in roots
        },
        "parse_errors": errors,
    }
    (args.output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
