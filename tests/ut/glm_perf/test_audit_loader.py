"""Metadata-only checks for the GLM W2 safetensors loader audit."""

import json
import struct
from pathlib import Path

import pytest

from tools.glm_perf.audit_loader import audit_checkpoint


def _checkpoint(
    tmp_path: Path,
    *,
    omit: str | None = None,
    wrong_dtype: str | None = None,
    duplicate: bool = False,
) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {
                    "model_type": "glm5_next_text",
                    "num_hidden_layers": 3,
                    "first_k_dense_replace": 1,
                    "n_routed_experts": 4,
                    "num_nextn_predict_layers": 1,
                },
            }
        )
    )
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    weight_map = {}
    for layer in (1, 2, 3):
        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for kind, dtype, size in (("codes", "U8", 1), ("scale", "F32", 4)):
                    name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_{kind}"
                    if name == omit:
                        continue
                    shard = list(shards)[expert % 2]
                    descriptor = {
                        "dtype": "F16" if name == wrong_dtype else dtype,
                        "shape": [1],
                        "data_offsets": [0, size],
                    }
                    shards[shard][name] = descriptor
                    weight_map[name] = shard
    if duplicate:
        name = "model.language_model.layers.1.mlp.experts.1.gate_proj_codes"
        # The index points to shard 2. Shard 1 contains an older payload that
        # the default iterator also yields before the indexed version.
        shards[list(shards)[0]][name] = shards[list(shards)[1]][name].copy()
    for shard in shards:
        name = f"model.language_model.{shard}.dense.weight"
        shards[shard][name] = {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
        weight_map[name] = shard
        header = json.dumps(shards[shard]).encode()
        # Deliberately omit tensor payload: a passing audit cannot have read it.
        (tmp_path / shard).write_bytes(struct.pack("<Q", len(header)) + header)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path


def _audit(checkpoint: Path) -> dict:
    return audit_checkpoint(
        checkpoint,
        expected_shards=2,
        tp_size=2,
        expected_experts=4,
        expected_decoder_layers=2,
        expected_mtp_layers=1,
        expected_peer_bytes=None,
        expected_local_bytes=None,
        expected_mtp_bytes=None,
        expected_extra_entries=None,
        expected_extra_bytes=None,
    )


def test_metadata_only_audit_covers_all_ranges_and_preserves_mtp(tmp_path: Path) -> None:
    report = _audit(_checkpoint(tmp_path))
    assert report["status"] == "pass"
    assert report["indexed_shards"] == 2
    assert report["decoder_layers"] == 2
    assert [rank["expert_range"] for rank in report["ranks"]] == [[0, 2], [2, 4]]
    for rank in report["ranks"]:
        assert rank["local_decoder_tensors"] == 24
        assert rank["local_decoder_bytes"] == 60
        assert rank["skipped_peer_tensors"] == 24
        assert rank["skipped_peer_bytes"] == 60
        assert rank["mtp_tensors_retained"] == 24
        assert rank["mtp_bytes_retained"] == 60
        assert rank["retained_shards"] == 2


def test_audit_rejects_missing_expert_tensor(tmp_path: Path) -> None:
    missing = "model.language_model.layers.2.mlp.experts.1.gate_proj_codes"
    with pytest.raises(ValueError, match="coverage"):
        _audit(_checkpoint(tmp_path, omit=missing))


def test_audit_rejects_wrong_expert_dtype(tmp_path: Path) -> None:
    wrong = "model.language_model.layers.1.mlp.experts.0.down_proj_scale"
    with pytest.raises(ValueError, match="expected F32"):
        _audit(_checkpoint(tmp_path, wrong_dtype=wrong))


def test_audit_counts_superseded_header_tensor_in_iterator_stream(tmp_path: Path) -> None:
    report = _audit(_checkpoint(tmp_path, duplicate=True))
    assert report["extra_header_tensors"] == 1
    assert report["extra_header_payload_bytes"] == 1
    assert report["final_tensor_sources_match_index"] is True
    assert report["ranks"][0]["local_decoder_tensors"] == 25
    assert report["ranks"][0]["local_decoder_bytes"] == 61
    assert report["ranks"][1]["skipped_peer_tensors"] == 25
    assert report["ranks"][1]["skipped_peer_bytes"] == 61
