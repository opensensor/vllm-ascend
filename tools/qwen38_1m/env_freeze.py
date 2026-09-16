#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment freeze / revision recorder for the Qwen4Exp 1M 310P deployment (plan T0.2).

Every 1M-context run artifact (plan R1, R13) must be reproducible against an
*exact* software stack. This module captures and pins the revisions of the
components the deployment depends on -- vLLM, vllm-ascend, torch-npu, CANN,
Transformers, tokenizers, ModelSlim -- plus the checkpoint hash, and emits a
versioned JSON document that is embedded in every run artifact.

It also enforces the pinned container layout: the running server's ``vllm`` and
``vllm_ascend`` imports must resolve to ``/vllm-workspace/vllm`` and
``/vllm-workspace/vllm-ascend`` respectively. A stray ``pip``-installed copy on
``sys.path`` silently changes behaviour, so :func:`import_check` fails loudly
when the resolved paths do not match the expected prefixes.

The recorder mirrors the injectable-collector style of ``hw_probe.py`` (plan
T0.3): all environment access goes through small collector callables, the default
collectors read package metadata / shell out on the target, and the unit tests
substitute fakes. The module never imports torch-npu at load time (the host has
no NPU); the default torch-npu collector reads metadata lazily.

Output: a versioned JSON document (round-trips via
:func:`EnvFreezeReport.from_dict`) plus a human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1

# Components whose revision must be recorded for a run artifact to be reproducible.
REQUIRED_COMPONENTS = (
    "vllm",
    "vllm_ascend",
    "torch_npu",
    "cann",
    "transformers",
    "tokenizers",
    "modelslim",
)

# Pinned container paths the running server's imports must resolve under.
EXPECTED_VLLM_PREFIX = "/vllm-workspace/vllm"
EXPECTED_VLLM_ASCEND_PREFIX = "/vllm-workspace/vllm-ascend"


class ImportCheckError(RuntimeError):
    """Raised when a module resolves to a path outside its pinned container prefix."""


@dataclass
class ComponentRevision:
    """The pinned revision of a single stack component.

    ``version`` is the human-facing revision string; ``path`` is where the
    component resolved on disk (when applicable); ``source`` records how the
    value was obtained (e.g. ``"importlib.metadata"``, ``"env"``, ``"npu-smi"``)
    so a reader can judge how authoritative it is.
    """

    name: str
    version: str | None = None
    path: str | None = None
    source: str | None = None


@dataclass
class EnvFreezeReport:
    """Full environment freeze. Round-trips through :meth:`to_dict`/:meth:`from_dict`."""

    schema_version: int
    timestamp: float
    components: dict[str, ComponentRevision] = field(default_factory=dict)
    checkpoint_hash: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "components": {name: asdict(rev) for name, rev in self.components.items()},
            "checkpoint_hash": self.checkpoint_hash,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, payload: dict) -> EnvFreezeReport:
        return cls(
            schema_version=payload["schema_version"],
            timestamp=payload["timestamp"],
            components={name: ComponentRevision(**rev) for name, rev in payload.get("components", {}).items()},
            checkpoint_hash=payload.get("checkpoint_hash"),
        )

    def missing_required(self) -> list[str]:
        """Required component keys that are absent from the report."""
        return [name for name in REQUIRED_COMPONENTS if name not in self.components]

    def human_summary(self) -> str:
        lines = [f"310P environment freeze (schema v{self.schema_version})"]
        for name in REQUIRED_COMPONENTS:
            rev = self.components.get(name)
            if rev is None:
                lines.append(f"  {name}: <MISSING>")
                continue
            suffix = f" @ {rev.path}" if rev.path else ""
            src = f" [{rev.source}]" if rev.source else ""
            lines.append(f"  {name}: {rev.version}{suffix}{src}")
        # Any extra components beyond the required set.
        for name, rev in self.components.items():
            if name in REQUIRED_COMPONENTS:
                continue
            lines.append(f"  {name}: {rev.version}")
        lines.append(f"  checkpoint_hash: {self.checkpoint_hash}")
        return "\n".join(lines)


def import_check(
    module_paths: dict[str, str],
    *,
    expected_vllm_prefix: str = EXPECTED_VLLM_PREFIX,
    expected_vllm_ascend_prefix: str = EXPECTED_VLLM_ASCEND_PREFIX,
) -> None:
    """Assert ``vllm``/``vllm_ascend`` resolve under their pinned container prefixes.

    ``module_paths`` maps a module name to the filesystem path its import
    resolved to (typically the module's ``__file__``). Raises
    :class:`ImportCheckError` -- loudly, so a mispinned server aborts before it
    produces an artifact -- when a module is missing or resolves outside its
    expected prefix. Prefixes are parameters so the check is testable off-target.
    """
    expected = {
        "vllm": expected_vllm_prefix,
        "vllm_ascend": expected_vllm_ascend_prefix,
    }
    for module_name, prefix in expected.items():
        resolved = module_paths.get(module_name)
        if resolved is None:
            raise ImportCheckError(
                f"{module_name!r} did not resolve to any path; expected an import under "
                f"{prefix!r} (is it installed in the pinned container?)"
            )
        if not resolved.startswith(prefix):
            raise ImportCheckError(
                f"{module_name!r} resolved to {resolved!r} which is outside the pinned "
                f"prefix {prefix!r}; a stray installed copy is on sys.path"
            )


# Collector callable types. Defaults touch the host/target; tests inject fakes.
RevisionCollector = Callable[[], dict[str, ComponentRevision]]
CheckpointHashCollector = Callable[[], str | None]


def freeze(
    *,
    revision_collector: RevisionCollector,
    checkpoint_hash_collector: CheckpointHashCollector | None = None,
    now: Callable[[], float] = time.time,
    require_all: bool = True,
) -> EnvFreezeReport:
    """Assemble an :class:`EnvFreezeReport` from the supplied collectors.

    ``revision_collector`` returns the component-name -> :class:`ComponentRevision`
    mapping. When ``require_all`` is true (the default), every entry in
    :data:`REQUIRED_COMPONENTS` must be present or a :class:`ValueError` is
    raised -- an artifact missing a pinned revision is not reproducible.
    """
    components = dict(revision_collector())
    if require_all:
        missing = [name for name in REQUIRED_COMPONENTS if name not in components]
        if missing:
            raise ValueError(
                f"environment freeze missing required components: {missing}; "
                f"all of {list(REQUIRED_COMPONENTS)} must be recorded"
            )
    checkpoint_hash = checkpoint_hash_collector() if checkpoint_hash_collector is not None else None
    return EnvFreezeReport(
        schema_version=SCHEMA_VERSION,
        timestamp=now(),
        components=components,
        checkpoint_hash=checkpoint_hash,
    )


def _metadata_revision(name: str, dist: str) -> ComponentRevision:  # pragma: no cover - env path
    """Best-effort revision for a Python distribution via importlib.metadata.

    Never imports the package (avoids the torch-npu import at load); reads only
    packaging metadata and, when available, the module's resolved origin.
    """
    import importlib.metadata as md
    import importlib.util

    version: str | None
    try:
        version = md.version(dist)
    except md.PackageNotFoundError:
        version = None
    path: str | None = None
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            path = spec.origin
    except (ImportError, ValueError):
        path = None
    return ComponentRevision(name=name, version=version, path=path, source="importlib.metadata")


def _default_revision_collector() -> dict[str, ComponentRevision]:  # pragma: no cover - target path
    """Collect component revisions from package metadata / environment on the target.

    Python packages come from ``importlib.metadata``; CANN and ModelSlim, which
    are not always pip-installed, fall back to environment variables set by the
    pinned container. This runs on the device/container wave; the host unit tests
    inject a fake collector instead.
    """
    components: dict[str, ComponentRevision] = {
        "vllm": _metadata_revision("vllm", "vllm"),
        "vllm_ascend": _metadata_revision("vllm_ascend", "vllm-ascend"),
        "torch_npu": _metadata_revision("torch_npu", "torch-npu"),
        "transformers": _metadata_revision("transformers", "transformers"),
        "tokenizers": _metadata_revision("tokenizers", "tokenizers"),
        "modelslim": _metadata_revision("modelslim", "modelslim"),
    }
    cann_version = os.environ.get("ASCEND_TOOLKIT_VERSION")
    cann_home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    components["cann"] = ComponentRevision(name="cann", version=cann_version, path=cann_home, source="env")
    return components


def _default_checkpoint_hash_collector() -> str | None:  # pragma: no cover - target path
    """Read the pinned checkpoint hash from the environment on the target."""
    return os.environ.get("QWEN38_CHECKPOINT_HASH")


def _default_module_paths() -> dict[str, str]:  # pragma: no cover - target path
    """Resolve ``vllm``/``vllm_ascend`` origins without importing them.

    Uses ``find_spec`` so the pinned-path assertion never triggers the heavy
    (and NPU-touching) import side effects of the real packages.
    """
    import importlib.util

    paths: dict[str, str] = {}
    for name in ("vllm", "vllm_ascend"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            paths[name] = spec.origin
    return paths


def main(argv=None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="310P environment freeze / revision recorder (plan T0.2)")
    parser.add_argument("--json-out", help="write the JSON freeze document to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress the human summary")
    parser.add_argument(
        "--skip-import-check",
        action="store_true",
        help="do not assert pinned /vllm-workspace import paths (off-target authoring only)",
    )
    args = parser.parse_args(argv)

    if not args.skip_import_check:
        import_check(_default_module_paths())

    report = freeze(
        revision_collector=_default_revision_collector,
        checkpoint_hash_collector=_default_checkpoint_hash_collector,
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(report.to_json())
    if not args.quiet:
        print(report.human_summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
