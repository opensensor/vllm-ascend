# SPDX-License-Identifier: Apache-2.0
"""Authenticated optional LOCAL vector-add kernel; no peer vector reads."""

import ctypes as c
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalAddSpec:
    library: str
    library_sha256: str
    binary: str
    binary_sha256: str
    blocks: int = 1

    def __post_init__(self):
        if type(self.blocks) is not int or self.blocks not in (1, 2, 4, 8):
            raise ValueError("local add grid must be 1, 2, 4 or 8")
        self.authenticate()

    def authenticate(self):
        for name in ("library", "binary"):
            value, checksum = getattr(self, name), getattr(self, name + "_sha256")
            if type(value) is not str or type(checksum) is not str:
                raise ValueError("local add requires explicit paths and SHA256 values")
            path = Path(value)
            if (
                not path.is_absolute()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != checksum
            ):
                raise ValueError("local add " + name + " path or SHA256 differs")

    @property
    def identity(self):
        value = dict(library=self.library_sha256, binary=self.binary_sha256, blocks=self.blocks)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class LocalAdd:
    """Retain loaded code, bound local buffers and the callback until pair drain."""

    def __init__(self, spec, owner):
        spec.authenticate()
        self.spec, self.owner = spec, owner
        self.handle = c.c_void_p()
        self.library = c.CDLL(spec.library)
        self.library.sidecard_add_abi.argtypes = []
        self.library.sidecard_add_abi.restype = c.c_uint64
        if self.library.sidecard_add_abi() != 1:
            raise ValueError("local add ABI differs")
        pointer = c.c_void_p
        for name, arguments in (
            (
                "sidecard_add_create",
                [pointer, c.c_uint64, c.c_uint64, c.c_uint32, pointer, pointer, pointer, c.POINTER(pointer)],
            ),
            ("sidecard_add_enqueue", [pointer, pointer]),
            ("sidecard_add_destroy", [c.POINTER(pointer), c.c_uint32]),
        ):
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, c.c_int
        self.image = c.create_string_buffer(Path(spec.binary).read_bytes())

    def attach(self):
        owner = self.owner
        if owner.config.rows not in (1, 4, 32):
            raise ValueError("local add has fixed binaries for rows 1, 4 and 32")
        status = self.library.sidecard_add_create(
            self.image,
            len(self.image) - 1,
            owner.config.elements,
            self.spec.blocks,
            owner.input,
            owner.inbox,
            owner.output,
            c.byref(self.handle),
        )
        if status:
            raise RuntimeError(f"local add creation failed: {status}; retain loaded code")
        function = owner.native.sidecard_pair_set_local_add
        function.argtypes, function.restype = [c.c_void_p, c.c_void_p, c.c_void_p], c.c_int
        owner._call(
            "sidecard_pair_set_local_add",
            owner.handle,
            self.handle,
            c.cast(self.library.sidecard_add_enqueue, c.c_void_p),
        )

    def close(self):
        # Only the backend's both-peer import-release path calls this. No
        # destructor or failure path unloads potentially executing device code.
        context = c.c_void_p()
        self.owner._call("aclrtGetCurrentContext", c.byref(context))
        status = self.library.sidecard_add_destroy(c.byref(self.handle), 1)
        if status:
            raise RuntimeError(f"local add unload failed: {status}; retain loaded code")
