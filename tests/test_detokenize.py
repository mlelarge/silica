"""Detokenizer tests (pure Python — a fake tokenizer, no MLX/checkpoint needed).

Covers the two audit bugs: a stop string split across tokens must not leak its
prefix, and a held-back / incomplete-multibyte tail must be flushed at end of
generation, not silently dropped.
"""

from silica.detokenize import IncrementalDetokenizer, REPLACEMENT


class FakeTok:
    """decode(ids) concatenates per-id string pieces (cumulative, like a real BPE)."""

    def __init__(self, table):
        self.table = table

    def decode(self, ids):
        return "".join(self.table[i] for i in ids)


def _run(table, ids, stop=()):
    d = IncrementalDetokenizer(FakeTok(table), stop=stop)
    segments = [d.add_token(i) for i in ids]
    segments.append(d.finalize())
    return d, "".join(segments)


def test_plain_streaming_no_stop():
    d, out = _run({0: "a", 1: "b", 2: "c"}, [0, 1, 2])
    assert out == "abc"
    assert not d.finished


def test_stop_split_across_tokens_does_not_leak_prefix():
    # "hello STOP world" with STOP = "ST"+"OP"; stop must not leak "ST"/"STOP".
    table = {0: "hello ", 1: "ST", 2: "OP", 3: " world"}
    d, out = _run(table, [0, 1, 2, 3], stop=("STOP",))
    assert out == "hello ", f"leaked stop prefix: {out!r}"
    assert "STOP" not in out and "ST" not in out
    assert d.finished and d.stop_reason == "stop"


def test_held_back_window_is_flushed():
    # No stop occurs; the hold-back window (for a possible "STOP") must be flushed.
    d, out = _run({0: "hel", 1: "lo"}, [0, 1], stop=("STOP",))
    assert out == "hello"
    assert not d.finished


def test_trailing_incomplete_multibyte_not_dropped():
    # Generation ends mid-character (decode ends in U+FFFD); finalize must emit it.
    d = IncrementalDetokenizer(FakeTok({0: "x", 1: REPLACEMENT}))
    assert d.add_token(0) == "x"
    assert d.add_token(1) == ""          # withheld: incomplete char
    flushed = d.finalize()
    assert flushed == REPLACEMENT, "trailing partial char was silently dropped"


def test_no_stop_means_no_holdback_delay():
    # Without stop strings, text streams immediately (hold == 0), no end delay.
    d = IncrementalDetokenizer(FakeTok({0: "abc"}))
    assert d.add_token(0) == "abc"
    assert d.finalize() == ""
