"""Detect and clean Whisper transcription artifacts.

Whisper has two known failure modes:
1. Decoder loop: repeats the last word/phrase indefinitely
2. Silence hallucination: outputs "Thank you." or similar on silent input

This module also carries bounded post-transcription repairs for observed
Epistaxis ontology vocabulary failures.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


_ONTOLOGY_REPAIRS = (
    (r"\bepistaxism\b", "Epistaxis main"),
    (r"\bepistaxistopos\b", "Epistaxis topos"),
    (r"\b(?:epistaxes|epistax|nepistaxis|epistexis|epistek|epistaxity)\b", "Epistaxis"),
    (r"\b(?:topoie|topoit)\b", "topoi"),
    (r"\ban (?:afro|afra)\b", "anaphora"),
    (r"\bafra\b", "anaphora"),
)


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

    best_truncated = None
    best_removed = 0

    # Try phrase lengths from short to long — find the atomic repeated
    # unit (e.g., "on this " rather than a large block of repeated text)
    max_phrase = len(text) // min_repeats
    for phrase_len in range(min_phrase_len, max_phrase + 1):
        # Check if the text ends with the same phrase repeated
        tail = text[-phrase_len:]
        count = 0
        pos = len(text)
        while pos >= phrase_len:
            candidate = text[pos - phrase_len:pos]
            if candidate.strip() == tail.strip():
                count += 1
                pos -= phrase_len
            else:
                break

        if count >= min_repeats and count - 1 > best_removed:
            best_truncated = text[:pos + phrase_len].strip()
            best_removed = count - 1

    if best_truncated is not None:
        logger.warning(
            "Truncated %d repetitions (best match)",
            best_removed,
        )
        return best_truncated

    return text


# Known Whisper silence hallucinations — these appear when the model
# receives silent or near-silent audio. Case-insensitive, stripped.
_SILENCE_HALLUCINATIONS = {
    "thank you.",
    "thank you",
    "thanks for watching.",
    "thanks for watching",
    "thanks for listening.",
    "thanks for listening",
    "you",
    "bye.",
    "bye",
    "the end.",
    "the end",
    "",
}


def is_hallucination(text: str) -> bool:
    """Return True if the text is a known Whisper silence hallucination."""
    return text.strip().lower() in _SILENCE_HALLUCINATIONS


def repair_ontology_terms(text: str) -> str:
    """Repair observed Epistaxis ontology vocabulary failures.

    This is intentionally log-backed and bounded. Only variants seen in
    launch logs are repaired here.
    """
    repaired = text
    for pattern, replacement in _ONTOLOGY_REPAIRS:
        repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)

    if repaired != text:
        logger.info("Repaired ontology vocabulary: %r -> %r", text, repaired)
    return repaired
