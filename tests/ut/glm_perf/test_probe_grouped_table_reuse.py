# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from argparse import Namespace

import pytest
import torch

from tools.deepseek_w2.w2_format import pack_codes as pack_reference
from tools.glm_perf.probe_grouped_table_reuse_310 import cases, compare, pack_codes
from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_probe_packer_matches_runtime_canonical_and_nz(bits: int):
    generator = torch.Generator().manual_seed(310 + bits)
    signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (2, 256, 256), dtype=torch.int8, generator=generator)
    canonical = pack_reference(signed, bits)
    assert torch.equal(pack_codes(signed, bits, nz_packed=False), canonical)

    repack = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
    expected = torch.stack([repack(canonical[expert], 256) for expert in range(2)]).view(torch.int8)
    assert torch.equal(pack_codes(signed, bits, nz_packed=True), expected)


def test_probe_cases_cover_empty_experts_and_wide_w3():
    matrix = cases()
    assert {case.bits for case in matrix} == {2, 3, 4}
    assert len(matrix) == 8
    assert any(case.group_ends[0] == 0 for case in matrix)
    assert any(case.group_ends[1] == case.group_ends[0] for case in matrix)
    assert all(case.group_ends[-1] < case.rows for case in matrix)
    assert any(case.bits == 3 and case.k == 4096 and case.group_ends[-1] > 32 for case in matrix)
    assert any(case.bits == 3 and len(case.group_ends) == 16 for case in matrix)


def test_probe_comparison_requires_distinct_binary_and_bitwise_output(tmp_path, capsys):
    package = {
        "binding_sha256": "binding",
        "grouped_source_sha256": "control-source",
        "standalone_binaries": {"single.o": "single"},
        "binaries": {"uint8": {"sha256": "before"}, "int8": {"sha256": "same"}},
    }
    record = {
        "case": "w3_small_first_empty",
        "bits": 3,
        "layout": "canonical",
        "shape": [4, 256, 256],
        "group_ends": [0, 1, 1, 3],
        "inputs_sha256": {"codes": "inputs"},
        "output_sha256": "output",
        "median_synchronized_call_ms": 1.0,
    }
    control = {"schema_version": 1, "package": package, "records": [record]}
    candidate = json.loads(json.dumps(control))
    candidate["package"]["grouped_source_sha256"] = "candidate-source"
    control_path, candidate_path = tmp_path / "control.json", tmp_path / "candidate.json"
    args = Namespace(control=control_path, candidate=candidate_path)
    control_path.write_text(json.dumps(control))
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="binaries are identical"):
        compare(args)

    candidate["package"]["binaries"]["uint8"]["sha256"] = "after"
    candidate["package"]["grouped_source_sha256"] = "control-source"
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="source hashes are identical"):
        compare(args)

    candidate["package"]["grouped_source_sha256"] = "candidate-source"
    candidate_path.write_text(json.dumps(candidate))
    compare(args)
    assert '"bitwise_parity": "passed"' in capsys.readouterr().out

    candidate["records"][0]["output_sha256"] = "different"
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="differ bitwise"):
        compare(args)
