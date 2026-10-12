# SPDX-License-Identifier: Apache-2.0
"""Accuracy configuration cannot imply broader coverage than the API gate."""

from pathlib import Path

import pytest
import yaml

from tools.sidecard.layer_validate import evaluation_config


def smoke_config():
    return {
        "model_type": "sidecard-layer-runtime",
        "num_fewshot": 0,
        "tasks": [{"name": "sidecard_arithmetic_smoke", "metrics": [{"name": "exact_match", "value": 1.0}]}],
    }


def test_published_configuration_names_the_actual_smoke():
    source = Path(__file__).resolve().parents[3]
    config = evaluation_config(source / "tests/e2e/models/configs/GLM5.3-Flash-Sidecard.yaml")
    assert config["serve"]["max_model_len"] == 65536
    assert config["tasks"] == smoke_config()["tasks"]


@pytest.mark.parametrize("field,value", [("model_type", "vllm"), ("num_fewshot", 5), ("tasks", [{"name": "mmlu"}])])
def test_reject_accuracy_scope_not_executed_by_this_gate(tmp_path, field, value):
    config = smoke_config()
    config[field] = value
    path = tmp_path / "unsupported.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        evaluation_config(path)
