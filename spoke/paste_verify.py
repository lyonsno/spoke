"""Post-paste OCR verification.

After a synthetic Cmd+V, captures the screen and runs Vision OCR to
confirm the pasted text actually appeared. If the text is not found,
the caller can enter recovery mode.

Uses Apple's Vision framework (VNRecognizeTextRequest) which runs on
the Neural Engine — ~50ms for a full-screen OCR pass after warmup.
"""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_missing_ocr_dependency_logged = False

# Minimum fuzzy match ratio to consider the paste successful.
# Generous because OCR may misread a few characters, and the pasted
# text might only be partially visible (scrolled, clipped, etc.).
_MATCH_THRESHOLD = 0.5
_MIN_CONTIGUOUS_MATCH = 0.35
_WINDOW_WORD_SLACK = 2
_ORDERED_WORD_SPAN_SLACK = 5

# Minimum length of pasted text to bother verifying — very short
# strings (1-2 words) are too likely to appear in UI chrome.
_MIN_VERIFY_LENGTH = 15


def capture_screen_text() -> str:
    """Capture the full screen and return all recognized text.

    Returns a single string with all OCR results concatenated.
    Returns empty string on any failure.
    """
    window_text = _capture_active_window_text()
    if window_text:
        return window_text
    return _capture_full_screen_text()


def _capture_active_window_text() -> str:
    """Capture OCR text from the frontmost window when possible.

    Uses Accurate recognition at half resolution for reliable paste
    verification (~450-500 ms).  Fast mode is unreliable on terminal /
    dark-background content — it misreads common characters (r→p, etc.).
    """
    try:
        from .scene_capture import (
            _capture_active_window,
            _capture_screen,
            _downsample_image,
            _image_dimensions,
            _run_ocr,
        )

        result = _capture_active_window()
        scope = "active_window"
        if result is None:
            result = _capture_screen()
            scope = "screen"
        if result is None:
            return ""

        image, _, _, _ = result
        # Downsample to half-res before Accurate OCR to keep latency
        # under ~500 ms (full-res Accurate can take 800-900 ms).
        image = _downsample_image(image)
        width, height = _image_dimensions(image)
        ocr_text, blocks = _run_ocr(
            image, width, height, f"paste-verify-{scope}", accurate=True,
        )
        if not " ".join(ocr_text.split()):
            return ""
        logger.info(
            "Paste verify OCR captured %d blocks from %s (%dx%d, accurate)",
            len(blocks),
            scope,
            width,
            height,
        )
        return ocr_text
    except Exception:
        logger.debug("Active-window OCR path failed", exc_info=True)
        return ""


def _capture_full_screen_text() -> str:
    """Legacy full-screen OCR fallback."""
    global _missing_ocr_dependency_logged
    try:
        from Vision import (
            VNRecognizeTextRequest,
            VNImageRequestHandler,
            VNRequestTextRecognitionLevelAccurate,
        )
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
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
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

    # Compare against the best local OCR window rather than the whole screen.
    # Whole-scene comparison collapses in terminals because surrounding text
    # dilutes the match even when the pasted line is actually visible.
    total_matched, longest_match, coverage, contiguous_coverage, block_count = (
        _best_match_stats(expected_norm, screen_norm)
    )

    logger.info(
        "Paste verify: %d/%d chars matched (%.0f%% coverage, %.0f%% contiguous, %d blocks, thresholds %.0f%%/%.0f%%)",
        total_matched,
        len(expected_norm),
        coverage * 100,
        contiguous_coverage * 100,
        block_count,
        _MATCH_THRESHOLD * 100,
        _MIN_CONTIGUOUS_MATCH * 100,
    )
    if logger.isEnabledFor(logging.DEBUG):
        # Show the first 100 chars of expected and a sample of screen text
        logger.debug("  expected: %r", expected_norm[:100])

    if coverage >= _MATCH_THRESHOLD and contiguous_coverage >= _MIN_CONTIGUOUS_MATCH:
        return True

    if _has_compact_ordered_word_match(expected_norm, screen_norm):
        return True

    # Fallback: accept only a strong distinctive probe match. A few words from
    # a dense background can inflate total character coverage, so the fallback
    # requires either both sides of an internal word or a boundary word with
    # its available side context.
    return _has_strong_distinctive_match(expected_norm, screen_norm)


def classify_paste_result(
    expected: str,
    screen_text: str,
    *,
    preexisting_match: bool | None = None,
) -> str:
    """Classify OCR verification as confirmed, ambiguous, missing, or unavailable."""
    if not " ".join(screen_text.split()):
        return "unavailable"
    if text_appears_on_screen(expected, screen_text):
        if preexisting_match is True:
            return "ambiguous"
        return "confirmed"
    return "missing"


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
_OCR_CONFUSABLES = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "8": "b",
})


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


def _has_strong_distinctive_match(expected: str, screen: str) -> bool:
    """Check for a stronger distinctive-word signal than a single probe hit.

    Internal words must match with context on both sides. Boundary words can
    match with the one side of context they actually have in the expected text.
    This keeps small clipped edge fragments viable without blessing dense
    background scenes that happen to share a couple of isolated words.
    """
    for match_start in _iter_distinctive_word_positions(expected):
        word_end = expected.index(" ", match_start) if " " in expected[match_start:] else len(expected)
        word = expected[match_start:word_end]

        left_ok = False
        if match_start >= 2:
            left_probe = expected[match_start - 2:word_end]
            left_ok = left_probe in screen

        right_ok = False
        if word_end + 2 <= len(expected):
            right_probe = expected[match_start:word_end + 2]
            right_ok = right_probe in screen

        if match_start == 0 and right_ok:
            logger.info("Paste verify: strong start-boundary probe for '%s'", word)
            return True

        if word_end == len(expected) and left_ok:
            logger.info("Paste verify: strong end-boundary probe for '%s'", word)
            return True

        if left_ok and right_ok:
            logger.info("Paste verify: strong two-sided probe for '%s'", word)
            return True

    return False


def _has_compact_ordered_word_match(expected: str, screen: str) -> bool:
    """Accept a compact ordered run of distinctive words through noisy OCR text."""
    expected_words = [
        word for word in _tokenize_words(expected)
        if len(word) >= _MIN_WORD_LENGTH and word not in _STOPWORDS
    ]
    screen_words = _tokenize_words(screen)
    if len(expected_words) < 4 or not screen_words:
        return False

    required = _ordered_word_match_requirement(len(expected_words))
    best_match_count = 0

    for start in range(len(screen_words)):
        match_count = 0
        matched_chars = 0
        screen_idx = start
        first_idx = None
        last_idx = None
        anchored = False

        for expected_idx, expected_word in enumerate(expected_words):
            found_idx = _find_matching_word(expected_word, screen_words, screen_idx)
            if found_idx is None:
                continue
            if first_idx is None:
                first_idx = found_idx
            last_idx = found_idx
            if expected_idx == 0:
                anchored = True
            match_count += 1
            matched_chars += len(expected_word)
            screen_idx = found_idx + 1

        best_match_count = max(best_match_count, match_count)
        if first_idx is None or last_idx is None:
            continue

        span = last_idx - first_idx + 1
        char_ratio = matched_chars / sum(len(word) for word in expected_words)
        if (
            anchored
            and
            match_count >= required
            and span <= match_count + _ORDERED_WORD_SPAN_SLACK
            and char_ratio >= 0.45
        ):
            logger.info(
                "Paste verify: ordered distinctive-word match (%d/%d words, span=%d)",
                match_count,
                len(expected_words),
                span,
            )
            return True

    logger.debug(
        "Paste verify: ordered distinctive-word fallback missed (%d/%d words)",
        best_match_count,
        len(expected_words),
    )
    return False


def _best_match_stats(expected: str, screen: str) -> tuple[int, int, float, float, int]:
    """Return the strongest local fuzzy-match stats for expected within screen."""
    full_stats = _match_stats(expected, screen)
    expected_words = expected.split()
    screen_words = screen.split()
    if len(screen_words) <= len(expected_words) + _WINDOW_WORD_SLACK:
        return full_stats

    best = full_stats
    min_window_words = max(1, len(expected_words) - _WINDOW_WORD_SLACK)
    max_window_words = min(len(screen_words), len(expected_words) + _WINDOW_WORD_SLACK)

    for window_words in range(min_window_words, max_window_words + 1):
        for start in range(0, len(screen_words) - window_words + 1):
            candidate = " ".join(screen_words[start:start + window_words])
            stats = _match_stats(expected, candidate)
            if stats[2] > best[2] or (stats[2] == best[2] and stats[3] > best[3]):
                best = stats

    return best


def _match_stats(expected: str, candidate: str) -> tuple[int, int, float, float, int]:
    """Return character-coverage stats for expected against one candidate text."""
    matcher = SequenceMatcher(None, expected, candidate)
    blocks = matcher.get_matching_blocks()
    total_matched = sum(b.size for b in blocks)
    longest_match = max((b.size for b in blocks), default=0)
    coverage = total_matched / len(expected) if expected else 0
    contiguous_coverage = longest_match / len(expected) if expected else 0
    block_count = max(0, len(blocks) - 1)
    return total_matched, longest_match, coverage, contiguous_coverage, block_count


def _ordered_word_match_requirement(expected_word_count: int) -> int:
    """Return how many distinctive words must survive OCR to confirm a paste."""
    if expected_word_count <= 5:
        return 4
    if expected_word_count <= 8:
        return 5
    return max(6, math.ceil(expected_word_count * 0.5))


def _find_matching_word(expected_word: str, screen_words: list[str], start: int) -> int | None:
    """Find the next screen word that plausibly matches the expected word."""
    for idx in range(start, len(screen_words)):
        candidate = screen_words[idx]
        if _words_match(expected_word, candidate):
            return idx
    return None


def _words_match(expected_word: str, candidate: str) -> bool:
    """Return True when two OCR tokens are close enough to count as the same word."""
    if expected_word == candidate:
        return True
    ratio = SequenceMatcher(None, expected_word, candidate).ratio()
    threshold = 0.84 if len(expected_word) <= 4 else 0.74
    return ratio >= threshold


def _tokenize_words(text: str) -> list[str]:
    """Tokenize text into OCR-tolerant lowercase words."""
    normalized = text.lower().translate(_OCR_CONFUSABLES)
    return re.findall(r"[a-z0-9]+", normalized)


def _iter_distinctive_word_positions(text: str):
    """Yield the start positions of distinctive words in text."""
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        if len(word) >= _MIN_WORD_LENGTH and word not in _STOPWORDS:
            yield start
        pos = start + len(word)
