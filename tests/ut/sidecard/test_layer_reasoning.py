# SPDX-License-Identifier: Apache-2.0
"""Reasoning/answer fields remain identical at every possible stream split."""

import pytest

from tools.sidecard.layer_reasoning import ReasoningStream, chat_message


@pytest.mark.parametrize("split", range(29))
def test_reasoning_stream_at_every_boundary(split):
    text = "Consider α carefully.</think>56"
    parser = ReasoningStream()
    result = {}
    for chunk in (text[:split], text[split:]):
        for key, value in parser.feed(chunk).items():
            result[key] = result.get(key, "") + value
    for key, value in parser.feed("", final=True).items():
        result[key] = result.get(key, "") + value
    assert result == {"reasoning_content": "Consider α carefully.", "content": "56"}


def test_unfinished_reasoning_and_partial_marker_are_preserved():
    assert chat_message("Working </think") == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Working </think",
    }
    assert chat_message("</think>hello") == {"role": "assistant", "content": "hello"}


def test_marker_in_answer_is_not_reparsed():
    parser = ReasoningStream()
    assert parser.feed("</think>answer") == {"content": "answer"}
    assert parser.feed(" uses </think>") == {"content": " uses </think>"}
