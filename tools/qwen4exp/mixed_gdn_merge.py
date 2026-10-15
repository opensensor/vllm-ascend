# SPDX-License-Identifier: Apache-2.0
"""Exact token reordering for mixed speculative/prefill GDN output.

The metadata builder supplies disjoint indices covering every actual token.
Build their inverse once per step; all GDN layers reuse it. No tensor values are
read by the host in normal serving. Explicit verification returns the reference
on failure and disables the candidate without interrupting rank collectives.
"""

import hashlib
import inspect
import textwrap

import torch

QUALIFIED_SOURCE_SHA256 = "6418b4dcba7f5692bc255bcb31f463e999e8272f234e956e78316a4d429f3d97"


def inverse_order(metadata, rows):
    spec, other = metadata.spec_token_indx, metadata.non_spec_token_indx
    cached = getattr(metadata, "_qfn_gdn_merge_order", None)
    if cached is not None and cached[0] is spec and cached[1] is other and cached[2].numel() == rows:
        return cached[2]
    if (
        spec.ndim != 1
        or other.ndim != 1
        or spec.numel() + other.numel() != rows
        or spec.dtype not in (torch.int32, torch.int64)
        or other.dtype != spec.dtype
        or other.device != spec.device
    ):
        raise ValueError("mixed GDN indices must partition all output rows")
    order = torch.empty(rows, dtype=torch.int64, device=spec.device)
    order[spec.long()] = torch.arange(spec.numel(), dtype=torch.int64, device=spec.device)
    order[other.long()] = torch.arange(spec.numel(), rows, dtype=torch.int64, device=spec.device)
    metadata._qfn_gdn_merge_order = (spec, other, order)
    return order


def reference_merge(branches, metadata):
    first, second = branches
    output = first.new_empty((first.shape[0] + second.shape[0], *first.shape[1:]))
    output[metadata.spec_token_indx] = first
    output[metadata.non_spec_token_indx] = second
    return output


class MixedGDNMerge:
    def __init__(self):
        self.enabled = self.verify = True
        self.failed = False
        self.calls = self.checked = self.mismatches = 0
        self.original = self.replacement = None

    def merge(self, branches, metadata):
        if not self.enabled or self.failed:
            return reference_merge(branches, metadata)
        packed = torch.cat(branches, dim=0)
        output = packed.index_select(0, inverse_order(metadata, packed.shape[0]))
        self.calls += 1
        if self.verify:
            expected = reference_merge(branches, metadata)
            # This synchronous comparison is used only in explicit live gates.
            if not torch.equal(output.view(torch.int16), expected.view(torch.int16)):
                self.failed = True
                self.mismatches += 1
            self.checked += 1
            return expected
        return output

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            calls=self.calls,
            checked=self.checked,
            mismatches=self.mismatches,
        )


def build_forward(original, owner):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("mixed GDN method differs from the qualified source")
    substitutions = (
        ("out = mixed.new_empty((mixed.shape[0], self.num_v_heads, self.params.head_v_dim))", "branches = []"),
        ("out[indices] = branch_out.to(out.dtype)", "branches.append(branch_out.to(mixed.dtype))"),
        ("return out.to(self.compute_dtype)", "return merge(branches, metadata).to(self.compute_dtype)"),
    )
    for before, after in substitutions:
        if source.count(before) != 1:
            raise RuntimeError("mixed GDN replacement must match exactly once")
        source = source.replace(before, after)
    namespace = dict(original.__globals__, merge=owner.merge)
    exec(compile(source, "<mixed-gdn-merge>", "exec"), namespace)
    return namespace[original.__name__]
