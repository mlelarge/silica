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
        # Hold back this many trailing chars from streaming: they could be the
        # start of a stop string and must not be emitted before we know. Without
        # this, a stop split across tokens (" EN" then "D" -> "END") leaks its
        # prefix to the caller. No stop strings -> no hold-back (hold = 0).
        self._hold = max((len(s) for s in self._stop), default=1) - 1
        self.finished = False
        self.stop_reason: str | None = None

    def add_token(self, token_id: int) -> str:
        """Feed one token id; return the newly emittable text (may be "")."""
        self._ids.append(token_id)
        decoded = self._tok.decode(self._ids)

        # Withhold an incomplete trailing multibyte character (flushed in finalize).
        if decoded.endswith(REPLACEMENT):
            return ""
        self._text = decoded

        # Complete stop-sequence: emit only up to its first occurrence, then stop.
        cut = self._first_stop_index(decoded)
        if cut is not None:
            self.finished = True
            self.stop_reason = "stop"
            segment = decoded[self._emitted:cut] if cut > self._emitted else ""
            self._emitted = max(self._emitted, cut)
            return segment

        # Otherwise stream everything except a trailing hold-back window that
        # could still become the prefix of a stop string.
        safe = max(self._emitted, len(decoded) - self._hold)
        segment = decoded[self._emitted:safe]
        self._emitted = safe
        return segment

    def finalize(self) -> str:
        """Flush any held-back / incomplete-char tail at end of generation.

        Without this the held-back window and any trailing partial multibyte
        character (U+FFFD) would be silently lost. Returns the final segment.
        """
        if self.finished:
            return ""
        decoded = self._tok.decode(self._ids)   # includes any trailing U+FFFD
        self._text = decoded
        segment = decoded[self._emitted:]
        self._emitted = len(decoded)
        return segment

    def _first_stop_index(self, text: str) -> int | None:
        idxs = [text.index(s) for s in self._stop if s in text]
        return min(idxs) if idxs else None

    @property
    def text(self) -> str:
        return self._text[:self._emitted] if self.finished else self._text
