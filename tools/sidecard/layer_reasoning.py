# SPDX-License-Identifier: Apache-2.0
"""Split the checkpoint's generated reasoning terminator across stream chunks."""

REASONING_END = "</think>"


class ReasoningStream:
    def __init__(self):
        self.pending = ""
        self.reasoning = True

    def feed(self, text, *, final=False):
        if not self.reasoning:
            return {"content": text} if text else {}
        self.pending += text
        before, found, after = self.pending.partition(REASONING_END)
        if found:
            self.pending = ""
            self.reasoning = False
            return {key: value for key, value in (("reasoning_content", before), ("content", after)) if value}
        keep = 0
        if not final:
            for length in range(1, len(REASONING_END)):
                if self.pending.endswith(REASONING_END[:length]):
                    keep = length
        emitted = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        return {"reasoning_content": emitted} if emitted else {}


def chat_message(text):
    fields = ReasoningStream().feed(text, final=True)
    return {"role": "assistant", "content": fields.get("content", ""), **fields}
