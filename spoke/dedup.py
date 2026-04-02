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
    (r"\bspoke-up as taxes\b", "spoke Epístaxis", False),
    (r"\bup as taxes\b", "Epístaxis", False),
    (r"\bin his taxes\b", "Epístaxis", False),
    (r"\bepistaxism\b", "Epístaxis main", False),
    (r"\bepistaxistopos\b", "Epístaxis tópos", False),
    (
        r"\b(?:epistaxis|epistaxes|epistax|epistaxists|nepistaxis|epistexis|epistek|epistaxity)\b",
        "Epístaxis",
        False,
    ),
    (r"\b(?:metadose(?:\s+(?:ii|so))?|metadosis)\b", "metádosis", True),
    (r"\b(?:uxis|of seizes|auxesis|oxygesis|oxesis|buxies|auxesus)\b", "aúxesis", True),
    (r"\b(?:sylloge|syllogy|silagee|sueji|silegy|sylergy)\b", "syllogé", True),
    (r"\b(?:tipos|topos)\b", "tópos", True),
    (r"\b(?:topoie|topoit|topoi)\b", "tópoi", True),
    (r"\b(?:catastasis|katastasis)\b", "katástasis", True),
    (
        r"\b(?:aposkepsis|aposcepsis|episcipsis|episcapsis|episcopes|episcus|upper skepticism|appless kept says|appless kepts)\b",
        "aposképsis",
        True,
    ),
    (
        r"\b(?:kerygma|kerigma|kergma|carrygma|carigma|curigma|karigma|charygma|chorigma)\b",
        "kérygma",
        True,
    ),
    (r"\ban (?:afro|afra)\b", "anaphorá", True),
    (r"\b(?:anaphora|afra|aphro)\b", "anaphorá", True),
    (r"\b(?:epin\s+orthosis|epinorthosis|epanorthosis|evanorthosis)\b", "epanórthosis", True),
    (r"\b(?:epispokisis|epispokosis|epispokisis|epispocasis)\b", "epispókisis", True),
    (r"\b(?:semi-hostess(?: concepts?)?|semi-oce's|semiosis|semeiosis|sēmeiōsis)\b", "sēmeiōsis", True),
    (r"\b(?:semion|semian|semeion|sēmeion)\b", "sēmeion", True),
    (r"\b(?:probolia|proboli|probly|probaly|probally|probole)\b", "probolé", True),
    (r"\b(?:autopoiesis|autopoises|autopuise|otopoiesis)\b", "autopoíesis", True),
    (r"\b(?:ooxisis)\b", "aúxesis", True),
    (r"\blysis\b", "lýsis", True),
)

_ONTOLOGY_DISPLAY_FORMS = tuple(
    sorted(
        {
            "Epístaxis",
            "tópos",
            "tópoi",
            "metádosis",
            "aúxesis",
            "syllogé",
            "katástasis",
            "aposképsis",
            "kérygma",
            "anaphorá",
            "epanórthosis",
            "epispókisis",
            "sēmeiōsis",
            "sēmeion",
            "probolé",
            "autopoíesis",
            "lýsis",
            "Epistaxis",
            "topos",
            "topoi",
            "metadosis",
            "auxesis",
            "sylloge",
            "katastasis",
            "aposkepsis",
            "kerygma",
            "anaphora",
            "epanorthosis",
            "epispokisis",
            "semiosis",
            "semeion",
            "probole",
            "autopoiesis",
            "lysis",
        },
        key=len,
        reverse=True,
    )
)
_ONTOLOGY_DISPLAY_PATTERNS = tuple(
    re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)
    for term in _ONTOLOGY_DISPLAY_FORMS
)


def _match_initial_case(replacement: str, observed: str) -> str:
    if observed.isupper():
        return replacement.upper()
    if observed[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def ontology_term_spans(text: str) -> list[tuple[int, int]]:
    """Return non-overlapping ranges for visible ontology terms in text."""
    spans: list[tuple[int, int]] = []
    for pattern in _ONTOLOGY_DISPLAY_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))

    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return merged


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
    for pattern, replacement, match_case in _ONTOLOGY_REPAIRS:
        if match_case:
            repaired = re.sub(
                pattern,
                lambda match, repl=replacement: _match_initial_case(repl, match.group(0)),
                repaired,
                flags=re.IGNORECASE,
            )
            continue
        repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)

    if repaired != text:
        logger.info("Repaired ontology vocabulary: %r -> %r", text, repaired)
    return repaired
