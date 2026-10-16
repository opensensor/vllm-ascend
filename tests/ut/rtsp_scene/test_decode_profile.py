import pytest

from tools.rtsp_scene.decode_profile import profile_body


def test_controlled_context_and_output_budget():
    body = profile_body(2244, 16, 1234)
    assert len(body["prompt"]) == 2244 and set(body["prompt"]) == {1234}
    assert body["ignore_eos"] and body["max_tokens"] == 16
    assert body["temperature"] == 0


@pytest.mark.parametrize("context,output,token", [(0, 16, 1234), (2244, 1, 1234), (2244, 16, -1)])
def test_reject_invalid_profile_contract(context, output, token):
    with pytest.raises(ValueError):
        profile_body(context, output, token)
