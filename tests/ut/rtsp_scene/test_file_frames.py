import base64
from urllib.parse import unquote, urlparse

import pytest

from tools.rtsp_scene.launch import build_command
from tools.rtsp_scene.runner import image_reference


def frame(path):
    path.write_bytes(b"P6\n2 1\n255\n" + bytes((10, 13, 32, 255, 0, 128)))
    return path


def test_local_reference_preserves_file_and_default_transport(tmp_path):
    path = frame(tmp_path / "frame with space.ppm")
    original = path.read_bytes()
    reference = image_reference(path, "file", tmp_path)
    assert urlparse(reference).scheme == "file"
    assert unquote(urlparse(reference).path) == str(path.resolve())
    assert base64.b64decode(image_reference(path).split(",", 1)[1]) == original
    assert path.read_bytes() == original


def test_local_frames_reject_escape_and_remote_endpoint(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = frame(tmp_path / "outside.ppm")
    linked = allowed / "linked.ppm"
    linked.symlink_to(outside)
    for path in (outside, linked):
        with pytest.raises(ValueError, match="inside the shared"):
            image_reference(path, "file", allowed)
    inside = frame(allowed / "frame.ppm")
    with pytest.raises(ValueError, match="loopback"):
        image_reference(inside, "file", allowed, "http://example.com:8003")
    with pytest.raises(ValueError, match="explicit shared"):
        image_reference(inside, "file")
    with pytest.raises(ValueError, match="explicit shared"):
        image_reference(inside, "unknown", allowed)


def test_local_reference_rejects_truncated_payload_without_encoding(tmp_path):
    path = frame(tmp_path / "frame.ppm")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload mismatch"):
        image_reference(path, "file", tmp_path)


def test_local_media_access_is_explicit_and_preserves_launch_configuration(tmp_path):
    original = build_command("/model", 8003, 8, True, 2, decode_graph_sizes=[1, 8], dense=True)
    command = build_command(
        "/model", 8003, 8, True, 2, decode_graph_sizes=[1, 8], dense=True, local_frame_root=str(tmp_path)
    )
    index = command.index("--allowed-local-media-path")
    assert command[index + 1] == str(tmp_path)
    assert command[:index] + command[index + 2 :] == original
    assert "--allowed-local-media-path" not in original
    with pytest.raises(ValueError, match="absolute"):
        build_command("/model", 8003, 8, True, 2, local_frame_root="relative")
