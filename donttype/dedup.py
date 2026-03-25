"""Detect and truncate Whisper-style repetition loops.

Whisper occasionally gets stuck in a decoder loop, repeating the last
word or phrase indefinitely. This module detects that pattern and
truncates the output at the first repetition.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def truncate_repetition(text: str, min_phrase_len: int = 3, min_repeats: int = 3) -> str:
    """Remove repeated phrases from the end of transcription text.

    Detects when a phrase of at least `min_phrase_len` characters is
    repeated `min_repeats` or more times consecutively, and truncates
    to keep only the first occurrence.

    Parameters
    ----------
    text : str
        The transcription text to clean.
    min_phrase_len : int
        Minimum character length of a phrase to consider as repetition.
    min_repeats : int
        Minimum number of consecutive repeats before truncating.

    Returns
    -------
    str
        The cleaned text, or the original if no repetition detected.
    """
    if not text or len(text) < min_phrase_len * min_repeats:
        return text

    # Try phrase lengths from long to short — catch longer repeated
    # phrases first (e.g., "I think so. I think so. I think so.")
    max_phrase = len(text) // min_repeats
    for phrase_len in range(max_phrase, min_phrase_len - 1, -1):
        # Check if the text ends with the same phrase repeated
        tail = text[-phrase_len:]
        count = 0
        pos = len(text)
        while pos >= phrase_len:
            candidate = text[pos - phrase_len:pos]
            # Allow minor whitespace variation
            if candidate.strip() == tail.strip():
                count += 1
                pos -= phrase_len
            else:
                break

        if count >= min_repeats:
            # Truncate to just before the repetitions, keeping one instance
            truncated = text[:pos + phrase_len].strip()
            removed = count - 1
            logger.warning(
                "Truncated %d repetitions of %r (phrase_len=%d)",
                removed, tail.strip()[:50], phrase_len,
            )
            return truncated

    return text
