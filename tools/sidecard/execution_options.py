# SPDX-License-Identifier: Apache-2.0
"""Strict, immutable, default-off admission for sidecard execution candidates.

Call validate_options before device allocation. In an initialized NPU owner,
load the qualified base ops library first, then load_native(spec, torch), then
make_mhc(options, torch). load_native returns (options, native_manifest_or_None).
It authenticates again immediately before loading the private addon. The parent
launcher owns inherited private OPP configuration; this module never changes it.

make_multi_route(actor, options) returns an optional standalone adapter. Runtime
dispatch must occur at R1 *before* NpuExperts pads to 17 rows. All R>=2, including
prefill tails R2..16, must retain the original actor and stable_rows=17. Never
replace the owner's native implementation or globally lower stable_rows.
"""

import os
from dataclasses import dataclass
from pathlib import Path

BLOCK_ARCHITECTURE = "cuda-prefix-ascend-middle-cuda-suffix"
EXPERT_ARCHITECTURE = "cuda-backbone-ascend-experts"
OPTION_KEYS = frozenset(
    ("kda_decode", "mhc_pre_rows", "mhc_post_rows", "multi_route", "dsa_decode_graphs", "native_build")
)
SHA256_LENGTH = 64
HEX_DIGITS = "0123456789abcdef"


@dataclass(frozen=True)
class BuildReference:
    root: Path
    manifest_sha256: str


@dataclass(frozen=True)
class ExecutionOptions:
    kda_decode: bool = False
    mhc_pre_rows: frozenset[int] = frozenset()
    mhc_post_rows: frozenset[int] = frozenset()
    multi_route: BuildReference | None = None
    dsa_decode_graphs: bool = False
    native_build: BuildReference | None = None

    @property
    def native_required(self):
        return bool(self.kda_decode or self.mhc_pre_rows or self.mhc_post_rows)

    @property
    def any_enabled(self):
        return bool(self.native_required or self.multi_route is not None or self.dsa_decode_graphs)


def _absolute(value, name):
    if type(value) is not str or not value or "\0" in value or not Path(value).is_absolute():
        raise ValueError(name + " must be an absolute path string")
    return Path(value).resolve()


def _reference(value, name):
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"root", "manifest_sha256"}:
        raise ValueError(name + " requires exactly root and manifest_sha256")
    checksum = value["manifest_sha256"]
    if (
        type(checksum) is not str
        or len(checksum) != SHA256_LENGTH
        or any(character not in HEX_DIGITS for character in checksum)
    ):
        raise ValueError(name + " requires a lowercase SHA256")
    return BuildReference(_absolute(value["root"], name + ".root"), checksum)


def _rows(value, name):
    if type(value) is not list or any(type(row) is not int or not 1 <= row <= 32 for row in value):
        raise ValueError(name + " requires a list of integers from 1 through 32")
    if len(set(value)) != len(value):
        raise ValueError(name + " requires unique rows")
    return frozenset(value)


def parse_options(spec):
    """Structural/combo admission only: no device/build imports or file hashing."""
    if type(spec) is not dict:
        raise ValueError("execution specification must be an object")
    raw = spec.get("execution_candidates", {})
    if type(raw) is not dict or set(raw) - OPTION_KEYS:
        raise ValueError("execution_candidates must be an object with known options")
    for name in ("kda_decode", "dsa_decode_graphs"):
        if type(raw.get(name, False)) is not bool:
            raise ValueError(name + " requires a boolean")
    options = ExecutionOptions(
        kda_decode=raw.get("kda_decode", False),
        mhc_pre_rows=_rows(raw.get("mhc_pre_rows", []), "mhc_pre_rows"),
        mhc_post_rows=_rows(raw.get("mhc_post_rows", []), "mhc_post_rows"),
        multi_route=_reference(raw.get("multi_route"), "multi_route"),
        dsa_decode_graphs=raw.get("dsa_decode_graphs", False),
        native_build=_reference(raw.get("native_build"), "native_build"),
    )
    if options.native_required:
        if options.native_build is None:
            raise ValueError("KDA/mHC candidates require native_build")
        _absolute(spec.get("npu_ops_library"), "npu_ops_library")
    elif options.native_build is not None:
        raise ValueError("unused native_build is forbidden")
    if options.any_enabled:
        architecture = spec.get("architecture", BLOCK_ARCHITECTURE)
        if architecture not in (BLOCK_ARCHITECTURE, EXPERT_ARCHITECTURE):
            raise ValueError("unsupported execution candidate architecture")
        if architecture == EXPERT_ARCHITECTURE and (options.native_required or options.dsa_decode_graphs):
            raise ValueError("expert-only architecture supports multi_route only")
    if options.dsa_decode_graphs and spec.get("npu_decode_graphs") is not True:
        raise ValueError("DSA decode capture requires npu_decode_graphs=true")
    return options


def _contained(root, value, name):
    if type(value) is not str or not value or "\0" in value or Path(value).is_absolute():
        raise ValueError(name + " must be a relative build path")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(name + " escapes the authenticated build")
    return path


def required_operators(options):
    """Torch operator names; manifest operators separately records CANN names."""
    required = set()
    if options.kda_decode:
        required.add("sidecard_kda_decode_310")
    if options.mhc_pre_rows:
        required.add("sidecard_mhc_pre_310")
    if options.mhc_post_rows:
        required.add("sidecard_mhc_post_310")
    return frozenset(required)


def _authenticate_native(spec, options):
    if options.native_build is None:
        return None
    from .execution_kernel_build import authenticate_build

    reference = options.native_build
    manifest = authenticate_build(reference.root, reference.manifest_sha256)
    if (
        type(manifest) is not dict
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("kind") != "sidecard-execution-build-v1"
        or manifest.get("build_complete") is not True
    ):
        raise ValueError("native execution build is incomplete or incompatible")
    operators = manifest.get("torch_operators")
    if (
        type(operators) is not list
        or any(type(name) is not str for name in operators)
        or len(operators) != len(set(operators))
        or not required_operators(options) <= set(operators)
    ):
        raise ValueError("native build lacks required Torch operators")
    base = _absolute(manifest.get("base_ops_library"), "base_ops_library")
    expected = _absolute(spec.get("npu_ops_library"), "npu_ops_library")
    if not base.is_file() or not expected.is_file() or not os.path.samefile(base, expected):
        raise ValueError("native build base ops library differs from runtime")
    addon = _contained(reference.root, manifest.get("addon_library"), "addon_library")
    opp = _contained(reference.root, manifest.get("opp_path"), "opp_path")
    if not addon.is_file() or not opp.is_dir():
        raise ValueError("native addon/OPP payload is missing")
    # Optional launch metadata authenticates the inherited private vendor path.
    # No SDK environment variable is read or changed here. The launcher must
    # establish its private OPP path before importing/initializing torch_npu.
    if "native_opp_path" in spec and _absolute(spec["native_opp_path"], "native_opp_path") != opp:
        raise ValueError("private OPP launch metadata differs from native build")
    return manifest


def _validate(spec):
    options = parse_options(spec)
    manifest = _authenticate_native(spec, options)
    if options.multi_route is not None:
        from .native_multi_route.adapter import authenticate_candidate

        authenticate_candidate(options.multi_route.root, options.multi_route.manifest_sha256)
    return options, manifest


def validate_options(spec):
    """Reauthenticate every selected receipt; returns immutable options."""
    return _validate(spec)[0]


def load_native(spec, torch):
    """Return (options, manifest); parent loads the qualified base library first.

    Hashes are rechecked immediately before loading the selected private addon.
    With all candidates off this performs no torch calls and no build imports.
    Operator registration errors propagate; unsafe partial launches never retry.
    """
    options, manifest = _validate(spec)
    if manifest is not None:
        library = _contained(options.native_build.root, manifest["addon_library"], "addon_library")
        torch.ops.load_library(str(library))
        for name in required_operators(options):
            if not callable(getattr(torch.ops._C_ascend, name, None)):
                raise RuntimeError("selected native operator is not registered: " + name)
    return options, manifest


def _unselected_operator(*args, **kwargs):
    raise RuntimeError("unselected mHC operator was dispatched")


def make_mhc(options, torch):
    """Create owner-local mHC policy from explicitly qualified row sets only."""
    if not options.mhc_pre_rows and not options.mhc_post_rows:
        return None
    from .npu_mhc import NativeMHC

    namespace = torch.ops._C_ascend
    return NativeMHC(
        pre_op=namespace.sidecard_mhc_pre_310 if options.mhc_pre_rows else _unselected_operator,
        post_op=namespace.sidecard_mhc_post_310 if options.mhc_post_rows else _unselected_operator,
        sinkhorn=namespace.mhc_sinkhorn_310,
        pre_rows=options.mhc_pre_rows,
        post_rows=options.mhc_post_rows,
    )


def make_multi_route(actor, options):
    """Create a standalone R1 adapter; retain original R2..16 prefill padding.

    Parent dispatches to this adapter only before original actor padding at R1.
    The adapter's multi-row fallback invokes the same original actor. Attaching
    a hook must not replace actor.native or cause recursive actor fallback.
    """
    if options.multi_route is None:
        return None
    from .native_multi_route.adapter import MultiRouteDecode

    if actor.stable_rows != 17:
        raise ValueError("multi-route requires original deterministic 17-row padding")
    return MultiRouteDecode(actor, options.multi_route.root, options.multi_route.manifest_sha256)
