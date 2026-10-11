# SPDX-License-Identifier: Apache-2.0
"""Real weights, changed-input graphs and device timing; explicitly opt-in."""

import json

import pytest


def test_real_glm_channel_scale_complete_pipeline(request, tmp_path):
    overlay = request.config.getoption("--glm-channel-overlay")
    bundle = request.config.getoption("--glm-channel-bundle")
    if overlay is None or bundle is None:
        pytest.skip("requires explicitly reserved NPU plus candidate overlay and bundle paths")
    # Lazy worker imports keep offline collection free of backend initialization.
    from tools.glm_perf.probe_channel_scale import run

    report = run(overlay, bundle, 3, 0, 4, (1, 2, 3, 4, 37, 640), request.config.getoption("--glm-channel-device"))
    (tmp_path / "channel-scale-operator-gate.json").write_text(json.dumps(report, indent=2) + "\n")
    assert len(report["cases"]) == 6
    assert set().union(*(case["physical_cube_rows_from_routes"] for case in report["cases"])) == {16, 32, 48, 64}
    assert all(case["implementation_nrmse"] <= 0.02 for case in report["cases"])
    assert all(case["changed_input_graph_nrmse"] <= 0.02 for case in report["cases"] if case["tokens"] <= 4)
