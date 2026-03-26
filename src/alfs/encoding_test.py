"""Tests for shared text utilities."""

from alfs.encoding import context_window


def test_context_window_basic() -> None:
    text = "The quick brown fox jumps"
    # "fox" starts at byte 16
    byte_offset = text.index(
        "fox"
    ).bit_length()  # just use char index as byte offset for ASCII
    byte_offset = len(b"The quick brown ")
    snippet, word_start = context_window(text, byte_offset, "fox", 10)
    assert "fox" in snippet
    assert snippet[word_start : word_start + 3] == "fox"


def test_context_window_at_start() -> None:
    text = "hello world"
    snippet, word_start = context_window(text, 0, "hello", 5)
    assert snippet.startswith("hello")
    assert word_start == 0


def test_context_window_limits_to_text_boundaries() -> None:
    text = "hi"
    snippet, word_start = context_window(text, 0, "hi", 100)
    assert snippet == "hi"
    assert word_start == 0


def test_context_window_mid_character_byte_offset_does_not_raise() -> None:
    # "café" — 'é' is 2 bytes (0xc3 0xa9). Byte offset 4 lands mid-character.
    text = "café world"
    # byte offset 4 splits 'é' (bytes 3-4); should degrade gracefully
    snippet, word_start = context_window(text, 4, "world", 20)
    assert isinstance(snippet, str)


def test_context_window_multibyte_alignment() -> None:
    # "日本語" — each character is 3 bytes
    text = "日本語test"
    # "test" starts at byte 9
    byte_offset = len("日本語".encode())
    snippet, word_start = context_window(text, byte_offset, "test", 10)
    assert "test" in snippet
    assert snippet[word_start : word_start + 4] == "test"
