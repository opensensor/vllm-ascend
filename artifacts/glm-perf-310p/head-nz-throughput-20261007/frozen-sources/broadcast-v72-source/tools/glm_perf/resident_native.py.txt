# SPDX-License-Identifier: Apache-2.0
"""Append-only native library transactions for paused resident workers.

Libraries register unique operator names and own kernel discovery themselves.
This loader never rewrites OPP search paths or unloads libraries used by graphs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class NativeManifest:
    def __init__(self, value: Any):
        required = {"name", "libraries", "assets", "operators", "validation_source"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("native manifest requires name, libraries, assets, operators, validation_source")
        name = value["name"]
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("native name must be an identifier including its version")
        if not isinstance(value["validation_source"], str) or not value["validation_source"].strip():
            raise ValueError("native manifest requires a validation function source")
        operators = value["operators"]
        if (
            not isinstance(operators, list)
            or not operators
            or any(
                not isinstance(op, str) or len(op.split("::")) != 2 or not all(p.isidentifier() for p in op.split("::"))
                for op in operators
            )
        ):
            raise ValueError("native operators must be namespace::name identifiers")
        if len(set(operators)) != len(operators):
            raise ValueError("duplicate native operator")
        paths = []
        for field in ("libraries", "assets"):
            if not isinstance(value[field], list) or (field == "libraries" and not value[field]):
                raise ValueError("libraries and assets must be lists, with at least one library")
            for entry in value[field]:
                if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                    raise ValueError("native files require path and sha256")
                path, digest = entry["path"], entry["sha256"]
                if not isinstance(path, str) or not Path(path).is_absolute():
                    raise ValueError("native file path must be absolute")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("native files require a SHA256 digest")
                paths.append(path)
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate native file path")
        self.payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(self.payload.encode()).hexdigest()
        self.name = name

    @property
    def value(self):
        return json.loads(self.payload)

    def verify_files(self):
        for entry in self.value["libraries"] + self.value["assets"]:
            if file_digest(Path(entry["path"])) != entry["sha256"]:
                raise ValueError(f"native file digest mismatch: {entry['path']}")


class NativeSession:
    def __init__(self):
        self.pending = None
        self.loaded = {}
        self.resources = {}
        self.failed = False

    def prepare(self, value, exists):
        self.pending = None
        manifest = NativeManifest(value)
        manifest.verify_files()
        if self.failed:
            raise RuntimeError("previous native mutation failed; restart workers before another load")
        previous = self.loaded.get(manifest.name)
        if previous is not None and previous["native_digest"] != manifest.digest:
            raise ValueError("native name already loaded with different content; use a new version")
        if previous is None and any(exists(op) for op in manifest.value["operators"]):
            raise ValueError("native operator already registered; use a new versioned name")
        self.pending = manifest
        return {"native_name": manifest.name, "native_digest": manifest.digest, "already_loaded": previous is not None}

    def load(self, digest, load_library, exists, synchronize):
        if self.pending is None or self.pending.digest != digest:
            raise ValueError("native manifest has not been prepared")
        manifest = self.pending
        # Recheck every artifact before the first non-reversible operation.
        manifest.verify_files()
        if manifest.name in self.loaded:
            self.pending = None
            return self.loaded[manifest.name]
        self.failed = True
        synchronize()
        for library in manifest.value["libraries"]:
            load_library(library["path"])
        if not all(exists(op) for op in manifest.value["operators"]):
            raise RuntimeError("native library did not register all declared operators")
        module = ModuleType(f"resident_native_{manifest.name}")
        exec(compile(manifest.value["validation_source"], f"<{manifest.name}>", "exec"), module.__dict__)
        validate = getattr(module, "validate", None)
        if not callable(validate):
            raise ValueError("native validation source must define validate()")
        prepare = getattr(module, "prepare", None)
        resource = prepare() if callable(prepare) else None
        result = validate(resource) if callable(prepare) else validate()
        synchronize()
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise RuntimeError("native validation did not pass")
        # Receipts must be serializable before this transaction is accepted.
        result = json.loads(json.dumps(result, allow_nan=False))
        manifest.verify_files()
        receipt = {"native_name": manifest.name, "native_digest": manifest.digest, "validation": result}
        self.loaded[manifest.name] = receipt
        self.resources[manifest.name] = resource
        self.pending = None
        self.failed = False
        return receipt
