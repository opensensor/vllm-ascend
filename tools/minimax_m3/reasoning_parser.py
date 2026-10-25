# SPDX-License-Identifier: Apache-2.0
"""Explicit M3.1 parser plugin that ignores reasoning markers in chat history.

Load with --reasoning-parser-plugin <this file> --reasoning-parser minimax_m31.
The upstream M3 parser scans the entire prompt when initializing reasoning state
and when the structured-output engine checks streamed tokens. M3.1's adaptive
template ends with an empty AI turn, so historical closers must not disable
parsing or activate a JSON grammar inside the new thinking block. Remove this
override after upstream scopes both predicates to the current assistant turn.
"""

from collections.abc import Iterable, Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser


@ReasoningParserManager.register_module("minimax_m31")
class MiniMaxM31ReasoningParser(MiniMaxM3ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._assistant_prefix_ids = tuple(self._encode_text("]~b]ai\n"))

    def _current_assistant_ids(self, input_ids: Sequence[int]) -> Sequence[int]:
        start = self._rfind_token_sequence(input_ids, self._assistant_prefix_ids)
        if start >= 0:
            input_ids = input_ids[start + len(self._assistant_prefix_ids) :]
        return input_ids

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return super().is_reasoning_end(self._current_assistant_ids(input_ids))

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        # The structured-output engine passes prompt + generated tokens here.
        # Historical closers must not activate a JSON grammar inside a new
        # thinking block. API streaming also calls this with output tokens only.
        return super().is_reasoning_end_streaming(self._current_assistant_ids(input_ids), delta_ids)
