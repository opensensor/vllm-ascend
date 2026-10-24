# SPDX-License-Identifier: Apache-2.0
"""Explicit M3.1 parser plugin that ignores reasoning markers in chat history.

Load with --reasoning-parser-plugin <this file> --reasoning-parser minimax_m31.
The upstream M3 parser scans the entire prompt when initializing reasoning state.
M3.1's adaptive template ends with an empty AI turn, so historical closers must
not disable parsing for the response. This override can be removed after the
upstream parser scopes its initial state to the current assistant turn.
"""

from collections.abc import Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser


@ReasoningParserManager.register_module("minimax_m31")
class MiniMaxM31ReasoningParser(MiniMaxM3ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._assistant_prefix_ids = tuple(self._encode_text("]~b]ai\n"))

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start = self._rfind_token_sequence(input_ids, self._assistant_prefix_ids)
        if start >= 0:
            input_ids = input_ids[start + len(self._assistant_prefix_ids) :]
        return super().is_reasoning_end(input_ids)
