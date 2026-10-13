# SPDX-License-Identifier: Apache-2.0
"""Earlier MTP capture metadata must survive a later builder invocation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/attention/metadata_builder.py"
tree = ast.parse(SOURCE.read_text())
cls = next(
    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionMetadataBuilder310"
)
nodes = [
    node for node in cls.body if getattr(node, "name", None) in ["_fill_query_lens_cpu", "build_for_graph_capture"]
]


class Parent:
    def build_for_graph_capture(self, common, state):
        return SimpleNamespace(
            query_lens_cpu=self._fill_query_lens_cpu(common.num_reqs, common.query_start_loc_cpu),
            seq_lens_cpu=common.seq_lens_cpu,
            attn_state=state,
        )


scope = {
    "torch": torch,
    "AscendAttentionMetadataBuilder": Parent,
    "AscendCommonAttentionMetadata": object,
    "AscendAttentionState": SimpleNamespace(DecodeOnly="decode"),
    "get_query_lens_cpu": lambda metadata: metadata.query_lens_cpu,
    "set_query_lens_cpu": lambda metadata, lengths: setattr(metadata, "query_lens_cpu", lengths),
}
new_class = ast.ClassDef(
    name="Builder",
    bases=[ast.Name(id="AscendAttentionMetadataBuilder", ctx=ast.Load())],
    keywords=[],
    body=nodes,
    decorator_list=[],
)
exec(compile(ast.fix_missing_locations(ast.Module(body=[new_class], type_ignores=[])), str(SOURCE), "exec"), scope)


@pytest.mark.parametrize("requests", [1, 6])
def test_capture_builder_owns_each_steps_host_query_lengths(requests):
    builder = scope["Builder"]()
    rows = requests * 3
    builder._query_lens_cpu_buffer = torch.empty(rows, dtype=torch.int32)
    first_common = SimpleNamespace(
        num_reqs=requests,
        query_start_loc_cpu=torch.arange(0, rows + 1, 3, dtype=torch.int32),
        seq_lens_cpu=torch.full((requests,), 3, dtype=torch.int32),
    )
    later_common = SimpleNamespace(
        num_reqs=rows,
        query_start_loc_cpu=torch.arange(rows + 1, dtype=torch.int32),
        seq_lens_cpu=torch.ones(rows, dtype=torch.int32),
    )
    first = builder.build_for_graph_capture(first_common, "spec")
    later = builder.build_for_graph_capture(later_common, "spec")
    assert first.query_lens_cpu.tolist() == [3] * requests
    assert later.query_lens_cpu.tolist() == [1] * rows
    assert first.query_lens_cpu.data_ptr() != later.query_lens_cpu.data_ptr()
    assert first.query_lens_cpu.data_ptr() != builder._query_lens_cpu_buffer.data_ptr()
    # Exercise the QSA causal-position reconstruction independently.
    positions = torch.cat(
        [
            torch.arange(seq - length, seq)
            for seq, length in zip(first.seq_lens_cpu.tolist(), first.query_lens_cpu.tolist(), strict=True)
        ]
    )
    assert positions.shape == (rows,)
    assert positions.tolist() == [0, 1, 2] * requests
