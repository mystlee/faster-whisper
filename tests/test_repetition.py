import pytest

from faster_whisper.transcribe import _truncate_repetition


def test_truncate_single_token():
    tokens = [1, 1, 1, 1, 1]
    assert _truncate_repetition(tokens, 5) == []


def test_truncate_pattern():
    tokens = [1, 2] * 5
    assert _truncate_repetition(tokens, 5) == []


def test_no_truncate_when_below_threshold():
    tokens = [3, 3, 3, 3]
    assert _truncate_repetition(tokens, 5) == tokens


def test_truncate_with_prefix():
    tokens = [7, 8, 9, 9, 9, 9, 9]
    assert _truncate_repetition(tokens, 5) == [7, 8]
