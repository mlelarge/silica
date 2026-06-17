"""Incremental, UTF-8-safe detokenization + stop handling.

The audit's single biggest functional hole: the plan stopped at token IDs.
Qwen3 uses a byte-level BPE tokenizer, so decoding token-by-token can split a
multibyte character (emoji, accents) mid-sequence. This streams text safely by
decoding the whole token buffer and emitting only the *newly completed* prefix,
withholding any trailing U+FFFD replacement char until the next token completes
the character.

It also owns termination: EOS/EOT token ids (handled in generate.py) and
arbitrary string stop-sequences (the `stop` arg is a server-layer feature in
mlx-lm; silica wires it into the loop directly).
"""

from __future__ import annotations

REPLACEMENT = "�"  # U+FFFD


class IncrementalDetokenizer:
    """Wraps any HF tokenizer exposing `.decode(list[int]) -> str`."""

    def __init__(self, tokenizer, stop: tuple[str, ...] = ()):
        self._tok = tokenizer
        self._ids: list[int] = []
        self._emitted = 0          # chars already returned to the caller
        self._text = ""            # full decode of buffered ids
        self._stop = tuple(s for s in stop if s)
        self.finished = False
        self.stop_reason: str | None = None

    def add_token(self, token_id: int) -> str:
        """Feed one token id; return the newly emittable text (may be "")."""
        self._ids.append(token_id)
        decoded = self._tok.decode(self._ids)

        # Withhold an incomplete trailing multibyte character.
        if decoded.endswith(REPLACEMENT):
            return ""
        self._text = decoded

        # String stop-sequence: emit only up to the first occurrence, then stop.
        cut = self._first_stop_index(decoded)
        if cut is not None:
            self.finished = True
            self.stop_reason = "stop"
            segment = decoded[self._emitted:cut]
            self._emitted = cut
            return segment

        segment = decoded[self._emitted:]
        self._emitted = len(decoded)
        return segment

    def _first_stop_index(self, text: str) -> int | None:
        idxs = [text.index(s) for s in self._stop if s in text]
        return min(idxs) if idxs else None

    @property
    def text(self) -> str:
        return self._text[:self._emitted] if self.finished else self._text
