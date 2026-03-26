"""Shared text utilities."""


def context_window(
    text: str, byte_offset: int, form: str, context_chars: int
) -> tuple[str, int]:
    """Return (snippet, word_start_in_snippet) for a byte-offset occurrence.

    ``snippet`` is the text window of width ``context_chars`` on each side of
    the occurrence.  ``word_start_in_snippet`` is the index within ``snippet``
    where ``form`` begins, so callers can apply their own formatting.

    Uses ``errors='ignore'`` when decoding the byte prefix so that mid-character
    byte offsets and any corpus encoding issues degrade gracefully instead of
    raising.
    """
    encoded = text.encode()
    char_offset = len(encoded[:byte_offset].decode(errors="ignore"))
    start = max(0, char_offset - context_chars)
    end = char_offset + len(form) + context_chars
    snippet = text[start:end]
    word_start = char_offset - start
    return snippet, word_start
