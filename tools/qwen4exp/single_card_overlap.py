# SPDX-License-Identifier: Apache-2.0
"""Reversible TP2 trials of early draft, complete-prefix QSA and pinned copies.

Uses the frozen bundle's resident API. Switching drains requests, invalidates
prefixes and recaptures graphs; weights stay loaded. This is not a hot update
that preserves active conversations or caches.
"""

import argparse
import ast
import hashlib
import inspect
import json
import textwrap
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.qwen4exp import single_card_positions
from tools.qwen4exp.resident_candidates import qfn_early_tp_draft
from tools.qwen4exp.tp6_overlap import BASE_QSA_SHA256, build_selection

QUALIFIED_PREFIX_SHA256 = "73e745eb324837fd1934cd89bf6ee84ea2b22461d38d74feb0aa2f84f36ba423"
QUALIFIED_POSITION_SHA256 = "1c2b4ce9d202656f60eb43db620e657d7023601c8c345fa0b4718b78c19df7be"

FACTORY = """
def shadow_selection(original, selection):
    # Private verification state belongs to this function, not module globals.
    checked = set()

    def checked_selection(*args, **kwargs):
        import torch

        result = selection(*args, **kwargs)
        signature = (tuple(args[0].shape), tuple(result.group_indices.shape), str(args[4].dtype),
                     (kwargs.get("max_visible_groups") or 0) > kwargs["token_topk"] // kwargs["compress_ratio"])
        if signature not in checked and not torch.npu.is_current_stream_capturing():
            expected = original(*args, **kwargs)
            for name in ("group_indices", "group_counts", "tail_starts", "tail_counts"):
                torch.testing.assert_close(getattr(result, name), getattr(expected, name), rtol=0, atol=0)
            checked.add(signature)
            print("SINGLE_CARD_QSA_EXACT", signature, flush=True)
        return result

    checked_selection._single_card_selection_base = original
    return checked_selection


def replacements():
    from vllm_ascend.models.qwen4_exp import model
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    result = {}
    if EARLY:
        result[TARGET] = build_forward(NPUModelRunner.sample_tokens)
    if PREFIX:
        original = model.qsa_indexer_select_groups_310
        original = getattr(original, "_single_card_selection_base", original)
        selection = build_selection(original, QSA_SOURCE)
        selection._single_card_selection_base = original
        if VERIFY:
            selection = shadow_selection(original, selection)
        result["vllm_ascend.models.qwen4_exp.model:qsa_indexer_select_groups_310"] = selection
    if POSITIONS:
        result["vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward"] = build_positions_forward(
            model._QSAAttention.forward, HOST_SOURCE, POSITION_SOURCE)
    return result
"""


def candidate_source(*, early, prefix, verify=False, positions=False):
    if any(type(value) is not bool for value in (early, prefix, positions)) or not (early or prefix or positions):
        raise ValueError("select at least one boolean optimization")
    if type(verify) is not bool or (verify and not prefix):
        raise ValueError("verification requires prefix selection")
    # Read the function without importing the NPU model on the control host.
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/models/qwen4_exp/ops/qsa_indexer.py"
    text = path.read_text()
    node = next(node for node in ast.parse(text).body if getattr(node, "name", None) == "qsa_indexer_select_groups_310")
    function = "".join(text.splitlines(True)[node.lineno - 1 : node.end_lineno])
    if hashlib.sha256(function.encode()).hexdigest() != QUALIFIED_PREFIX_SHA256:
        raise ValueError("complete-prefix candidate source changed; requalify before applying")
    model = path.parents[1] / "model.py"
    model_source = model.read_text()
    model_tree = ast.parse(model_source)

    def extract(node):
        return textwrap.dedent("".join(model_source.splitlines(True)[node.lineno - 1 : node.end_lineno]))

    host_source = extract(next(n for n in model_tree.body if getattr(n, "name", None) == "_host_qsa_logical_positions"))
    attention = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "_QSAAttention")
    position_source = extract(next(n for n in attention.body if getattr(n, "name", None) == "_logical_query_positions"))
    if hashlib.sha256((host_source + "\n" + position_source).encode()).hexdigest() != QUALIFIED_POSITION_SHA256:
        raise ValueError("logical-position candidate source changed; requalify before applying")
    return (
        inspect.getsource(qfn_early_tp_draft)
        + f"\nBASE_QSA_SHA256 = {BASE_QSA_SHA256!r}\n"
        + inspect.getsource(build_selection)
        + "\n"
        + inspect.getsource(single_card_positions)
        + f"\nEARLY = {early!r}\nPREFIX = {prefix!r}\nVERIFY = {verify!r}\nQSA_SOURCE = {function!r}\n"
        + f"POSITIONS = {positions!r}\nHOST_SOURCE = {host_source!r}\nPOSITION_SOURCE = {position_source!r}\n"
        + FACTORY
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--early", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument(
        "--positions", action="store_true", help="copy pinned causal positions directly into graph buffers"
    )
    parser.add_argument("--verify", action="store_true", help="shadow exact QSA outputs; exclude from timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing receipts")
    if args.verify and not args.prefix:
        parser.error("verification requires --prefix")
    source = (
        candidate_source(early=args.early, prefix=args.prefix, verify=args.verify, positions=args.positions)
        if args.early or args.prefix or args.positions
        else ""
    )
    control = Control(uuid.uuid4().hex, candidate="single_card_overlap" if source else "baseline", source=source)
    client = ResidentClient(args.base_url, expected_workers=2)
    result = dict(
        early=args.early,
        prefix=args.prefix,
        verify=args.verify,
        positions=args.positions,
        source=source,
        before=client.rpc("resident_status"),
    )
    try:
        result["after"] = client.switch(control)
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Both TP2 workers switched and recaptured; weights retained, prefix caches cleared.")


if __name__ == "__main__":
    main()
