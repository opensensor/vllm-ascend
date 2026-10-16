# SPDX-License-Identifier: Apache-2.0
"""Port the current pinned-host QSA copy into the exact frozen TP2 forward.

Only its logical-position dispatch changes. Preserve all original attention,
graph ownership, weights, and globals; do not transplant the newer whole model.
"""

import hashlib
import inspect
import textwrap
from types import FunctionType

FORWARD_SHA256 = "e4675230f8d0cec6f4d7c6af8c47f888f8ad57ba3f5f373fd762e812a1855bde"


def build_positions_forward(original, host_source, position_source, *, source=None):
    original = getattr(original, "_single_card_position_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != FORWARD_SHA256:
        raise ValueError("requires the exact frozen TP2 attention forward")
    seam = "self._logical_query_positions("
    if source.count(seam) != 1 or not source.startswith("def forward("):
        raise ValueError("logical-position dispatch seam changed")
    body = host_source + "\n" + position_source + "\n" + source.replace(seam, "_logical_query_positions(")
    factory_source = "def factory():\n" + textwrap.indent(body, "    ") + "    return forward\n"
    namespace = dict(original.__globals__)
    exec(compile(factory_source, "<single-card-pinned-positions>", "exec"), namespace)
    # Construct nested helpers with the live module globals, not a stale copy.
    factory = FunctionType(namespace["factory"].__code__, original.__globals__)
    forward = factory()
    forward.__qualname__ = original.__qualname__
    forward._single_card_position_base = original
    return forward
