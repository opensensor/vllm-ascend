# SPDX-License-Identifier: Apache-2.0
"""Regression for adaptive chat prompts containing historical thinking closers."""

import importlib
import string

import pytest


@pytest.fixture
def parser_class():
    pytest.importorskip("vllm.reasoning.minimax_m3_reasoning_parser")
    return importlib.import_module("tools.minimax_m3.reasoning_parser").MiniMaxM31ReasoningParser


class Tokenizer:
    def __init__(self):
        self.vocab = {token: i for i, token in enumerate(["<mm:think>", "</mm:think>", *string.printable])}
        self.inverse = {i: token for token, i in self.vocab.items()}

    def get_vocab(self):
        return self.vocab

    def encode(self, text, **kwargs):
        result = []
        while text:
            token = next((t for t in ("<mm:think>", "</mm:think>") if text.startswith(t)), text[0])
            result.append(self.vocab[token])
            text = text[len(token) :]
        return result

    def decode(self, ids, **kwargs):
        return "".join(self.inverse[i] for i in ids)


@pytest.mark.parametrize("history", ["Use </mm:think> to end reasoning.", "<mm:think>old thought</mm:think>old answer"])
def test_history_does_not_disable_adaptive_response_parsing(history, parser_class):
    tokenizer = Tokenizer()
    parser = parser_class(tokenizer)
    prompt = tokenizer.encode("]~b]system\n" + history + "[e~[\n]~b]user\nHello[e~[\n]~b]ai\n")
    assert not parser.is_reasoning_end(prompt)
    closer = tokenizer.encode("</mm:think>")
    delta = parser.extract_reasoning_streaming("", "</mm:think>", "</mm:think>", [], closer, closer)
    assert delta is None
    hello = tokenizer.encode("Hello!")
    delta = parser.extract_reasoning_streaming(
        "</mm:think>", "</mm:think>Hello!", "Hello!", closer, closer + hello, hello
    )
    assert delta.content == "Hello!" and delta.reasoning is None


@pytest.mark.parametrize(
    "suffix,ended",
    [("", False), ("<mm:think>current thought", False), ("<mm:think>done</mm:think>", True), ("</mm:think>", True)],
)
def test_current_assistant_prefill_controls_initial_state(suffix, ended, parser_class):
    tokenizer = Tokenizer()
    parser = parser_class(tokenizer)
    prompt = "]~b]ai\n</mm:think>previous[e~[\n]~b]user\nQuestion[e~[\n]~b]ai\n" + suffix
    assert parser.is_reasoning_end(tokenizer.encode(prompt)) is ended
    assert parser.is_reasoning_end(tokenizer.encode(suffix)) is ended
