# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the long-context corpus / retrieval-probe generator (plan T0.4).

The real Qwen4Exp checkpoint tokenizer is not present on this host, so every test
injects a deterministic FAKE tokenizer (whitespace word-splitter with a growing
vocab and a fixed eos id). Token counts are therefore exact and reproducible, and
exact-count validation against the real tokenizer is deferred to the device wave.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_MODULE_PATH = _REPO_ROOT / "tools" / "qwen38_1m" / "corpus_gen.py"


def _load_corpus_gen():
    spec = importlib.util.spec_from_file_location("qwen38_corpus_gen", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


corpus_gen = _load_corpus_gen()


class FakeTokenizer:
    """Deterministic whitespace tokenizer: one token per whitespace-delimited word.

    A growing vocab maps each distinct word to a stable id, so token counts are an
    exact function of the text and reproducible across runs.
    """

    def __init__(self, eos_id: int = 0):
        self.eos_id = eos_id
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            token_id = self._vocab.get(word)
            if token_id is None:
                token_id = len(self._vocab) + 1
                self._vocab[word] = token_id
            ids.append(token_id)
        return ids


def test_target_sizes_are_the_five_documented_sizes():
    assert corpus_gen.TARGET_TOKEN_SIZES == (8192, 131072, 262144, 524288, 1048060)


@pytest.mark.parametrize("target", corpus_gen.TARGET_TOKEN_SIZES)
def test_token_counts_are_exact_at_every_size(target):
    tok = FakeTokenizer()
    built = corpus_gen.build_prompt(target, tok, seed=1234)
    assert built.target_tokens == target
    assert built.token_count == target
    # The recorded count matches an independent re-encode of the emitted text.
    assert len(tok.encode(built.text)) == target


def test_eight_needles_plus_one_adversarial_across_five_zones():
    built = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=1234)
    answerable = [n for n in built.needles if not n.is_adversarial]
    adversarial = [n for n in built.needles if n.is_adversarial]
    assert len(answerable) == 8
    assert len(adversarial) == 1
    labels = {n.position_label for n in answerable}
    assert labels == {"beginning", "quarter", "middle", "three_quarter", "end"}


def test_adversarial_case_is_present_and_marked_no_answer():
    built = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=1234)
    adv = [n for n in built.needles if n.is_adversarial]
    assert len(adv) == 1
    assert adv[0].expected_answer == corpus_gen.NO_ANSWER
    answers = built.expected_answers()
    assert answers[adv[0].needle_id] == corpus_gen.NO_ANSWER


def test_expected_answer_map_is_extractable():
    built = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=1234)
    answers = built.expected_answers()
    # 8 answerable + 1 adversarial (marked) = 9 entries.
    assert len(answers) == 9
    for needle in built.needles:
        assert answers[needle.needle_id] == needle.expected_answer
        # Every answerable expected value actually appears in the document text.
        if not needle.is_adversarial:
            assert needle.expected_answer in built.text


def test_deterministic_for_fixed_seed():
    a = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=1234)
    b = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=1234)
    assert a.text == b.text
    assert a.expected_answers() == b.expected_answers()
    assert [n.position_label for n in a.needles] == [n.position_label for n in b.needles]
    assert [n.fraction for n in a.needles] == [n.fraction for n in b.needles]


def test_different_seed_changes_answers_but_not_positions():
    a = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=1)
    b = corpus_gen.build_prompt(8192, FakeTokenizer(), seed=2)
    assert a.expected_answers() != b.expected_answers()
    # Positional layout is seed-independent (needle *content* is seeded, not placement).
    assert [n.fraction for n in a.needles] == [n.fraction for n in b.needles]


def test_needles_positioned_in_ascending_document_order():
    built = corpus_gen.build_prompt(131072, FakeTokenizer(), seed=1234)
    # Every needle's FACT marker appears, and beginning-zone precedes end-zone.
    for needle in built.needles:
        assert needle.needle_id in built.text
    begin = next(n for n in built.needles if n.position_label == "beginning")
    end = next(n for n in built.needles if n.position_label == "end")
    assert built.text.index(begin.needle_id) < built.text.index(end.needle_id)


def test_build_all_covers_every_size():
    tok = FakeTokenizer()
    prompts = corpus_gen.build_all(tok, seed=1234)
    assert set(prompts) == set(corpus_gen.TARGET_TOKEN_SIZES)
    for target, built in prompts.items():
        assert built.token_count == target


def test_rejects_target_smaller_than_fixed_content():
    with pytest.raises(ValueError):
        corpus_gen.build_prompt(4, FakeTokenizer(), seed=1234)
