"""Post-paste OCR verification.

After a synthetic Cmd+V, captures the screen and runs Vision OCR to
confirm the pasted text actually appeared. If the text is not found,
the caller can enter recovery mode.

Uses Apple's Vision framework (VNRecognizeTextRequest) which runs on
the Neural Engine — ~50ms for a full-screen OCR pass after warmup.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_missing_ocr_dependency_logged = False

# Minimum fuzzy match ratio to consider the paste successful.
# Generous because OCR may misread a few characters, and the pasted
# text might only be partially visible (scrolled, clipped, etc.).
_MATCH_THRESHOLD = 0.5

# Minimum length of pasted text to bother verifying — very short
# strings (1-2 words) are too likely to appear in UI chrome.
_MIN_VERIFY_LENGTH = 15


def capture_screen_text() -> str:
    """Capture the full screen and return all recognized text.

    Returns a single string with all OCR results concatenated.
    Returns empty string on any failure.
    """
    global _missing_ocr_dependency_logged
    try:
        from Vision import VNRecognizeTextRequest, VNImageRequestHandler, VNRequestTextRecognitionLevelFast
        from Quartz import (
            CGWindowListCreateImage,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            CGRectInfinite,
        )

        image = CGWindowListCreateImage(
            CGRectInfinite, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, 0
        )
        if image is None:
            logger.warning("Paste verify OCR failed: screen capture returned no image")
            return ""

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelFast)
        request.setUsesLanguageCorrection_(False)

        success, error = handler.performRequests_error_([request], None)
        if not success:
            logger.warning("Paste verify OCR failed: Vision request unsuccessful: %s", error)
            return ""

        results = request.results()
        if not results:
            logger.warning("Paste verify OCR produced no text observations")
            return ""

        lines = []
        for observation in results:
            candidates = observation.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())

        return " ".join(lines)
    except ModuleNotFoundError as exc:
        if not _missing_ocr_dependency_logged:
            logger.warning("Paste verify OCR unavailable: missing dependency: %s", exc)
            _missing_ocr_dependency_logged = True
        return ""
    except Exception:
        logger.debug("Screen OCR failed", exc_info=True)
        return ""


def text_appears_on_screen(expected: str, screen_text: str) -> bool:
    """Check whether the expected text appears in the OCR output.

    Uses a greedy fuzzy match — we know exactly what we're looking for,
    so even a partial match is high confidence.

    For very short texts (< _MIN_VERIFY_LENGTH chars), always returns
    True since short strings match UI chrome too easily and produce
    false negatives.
    """
    if not expected or not screen_text:
        return False

    if len(expected) < _MIN_VERIFY_LENGTH:
        # Too short to reliably verify — assume success
        logger.debug("Text too short to verify (%d chars), assuming success", len(expected))
        return True

    # Normalize whitespace for comparison
    expected_norm = " ".join(expected.split()).lower()
    screen_norm = " ".join(screen_text.split()).lower()

    # Check if a substantial substring of expected appears in screen text.
    if expected_norm in screen_norm:
        logger.info("Paste verify: exact substring match found")
        return True

    # Fuzzy match using SequenceMatcher ratio over the full texts.
    # This accounts for OCR errors, line breaks splitting words,
    # and partial visibility at screen edges.
    matcher = SequenceMatcher(None, expected_norm, screen_norm)

    # get_matching_blocks returns all matching subsequences, not just
    # the longest one. Sum their lengths for total character coverage.
    blocks = matcher.get_matching_blocks()
    total_matched = sum(b.size for b in blocks)
    coverage = total_matched / len(expected_norm) if expected_norm else 0

    logger.info(
        "Paste verify: %d/%d chars matched (%.0f%% coverage, %d blocks, threshold %.0f%%)",
        total_matched, len(expected_norm), coverage * 100,
        len(blocks) - 1,  # last block is always (len_a, len_b, 0)
        _MATCH_THRESHOLD * 100,
    )
    if logger.isEnabledFor(logging.DEBUG):
        # Show the first 100 chars of expected and a sample of screen text
        logger.debug("  expected: %r", expected_norm[:100])
        # Find the region of screen text near the best match
        best = max(blocks[:-1], key=lambda b: b.size) if len(blocks) > 1 else None
        if best:
            start = max(0, best.b - 20)
            end = min(len(screen_norm), best.b + best.size + 20)
            logger.debug("  best match region: %r", screen_norm[start:end])

    if coverage >= _MATCH_THRESHOLD:
        return True

    # Fallback: if even one distinctive word from the expected text appears
    # on screen, the paste almost certainly went through — the rest is just
    # scrolled, clipped, or at screen edges the OCR can't read.
    return _has_distinctive_word_match(expected_norm, screen_norm)


# Words too common to be a reliable signal that paste succeeded.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "to", "of", "or", "and",
    "for", "at", "by", "so", "if", "as", "be", "do", "he", "we", "my",
    "no", "up", "am", "me", "us", "was", "are", "has", "had", "his",
    "her", "its", "our", "but", "not", "can", "did", "may", "all",
    "you", "she", "him", "who", "how", "get", "got", "let", "say",
    "this", "that", "with", "from", "have", "will", "been", "them",
    "they", "than", "then", "what", "when", "just", "also", "like",
    "some", "into", "each", "only", "very", "much", "such", "here",
    "i", "oh", "ok", "um", "uh", "yeah", "well", "okay",
})

# Minimum word length to consider distinctive (skips "I", "a", etc.)
_MIN_WORD_LENGTH = 3


def _has_distinctive_word_match(expected: str, screen: str) -> bool:
    """Check if a distinctive word with adjacent context appears in screen text.

    For each distinctive word in the expected text, check whether the word
    plus at least one character of its surrounding context from the expected
    text appears in the screen text. This prevents false positives from UI
    chrome (e.g., "Settings" in a menu bar) while being flexible enough to
    handle OCR word-splitting and partial visibility.

    For a word at position i in the expected string, we check:
      - left context:  expected[i-2:i] + word  (char before the space + space + word)
      - right context: word + expected[j:j+2]  (word + space + char after the space)

    Either match confirms the paste.
    """
    for match_start in _iter_distinctive_word_positions(expected):
        word_end = expected.index(" ", match_start) if " " in expected[match_start:] else len(expected)
        word = expected[match_start:word_end]

        # Left context: one char before the space that precedes this word
        if match_start >= 2:
            left_probe = expected[match_start - 2:word_end]
            if left_probe in screen:
                logger.info("Paste verify: word+left context match '%s'", left_probe)
                return True

        # Right context: one char after the space that follows this word
        if word_end + 2 <= len(expected):
            right_probe = expected[match_start:word_end + 2]
            if right_probe in screen:
                logger.info("Paste verify: word+right context match '%s'", right_probe)
                return True

        # Edge case: word is at the very start or end of the text.
        # Only has context on one side. If it's a boundary word and
        # appears on screen, accept it — it can't be chrome because
        # chrome wouldn't have the same boundary.
        if match_start == 0 and word in screen:
            # Check it's not just a substring of a longer screen word
            # by verifying a space or boundary follows it in screen
            for probe in [word + " ", " " + word]:
                if probe in screen:
                    logger.info("Paste verify: boundary word match '%s'", word)
                    return True

    return False


def _iter_distinctive_word_positions(text: str):
    """Yield the start positions of distinctive words in text."""
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        if len(word) >= _MIN_WORD_LENGTH and word not in _STOPWORDS:
            yield start
        pos = start + len(word)
