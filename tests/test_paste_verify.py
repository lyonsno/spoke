"""Tests for post-paste OCR verification."""

import importlib
import sys
from unittest.mock import patch


def _import_module():
    sys.modules.pop("spoke.paste_verify", None)
    return importlib.import_module("spoke.paste_verify")


class TestTextAppearsOnScreen:
    """Test the fuzzy matching logic (no OCR dependency needed)."""

    def test_exact_match(self):
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Hello world this is a test sentence",
            "Some UI chrome Hello world this is a test sentence more stuff"
        ) is True

    def test_partial_match_above_threshold(self):
        mod = _import_module()
        # OCR might miss a few characters but the bulk is there
        assert mod.text_appears_on_screen(
            "Hello world this is a test sentence",
            "Some chrome Hello world this is a test more stuff"
        ) is True

    def test_no_match(self):
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Hello world this is a test sentence",
            "Completely different text on screen about other things entirely"
        ) is False

    def test_short_text_always_passes(self):
        mod = _import_module()
        # Short texts skip verification to avoid false negatives
        assert mod.text_appears_on_screen("Hi", "Anything") is True

    def test_empty_expected_returns_false(self):
        mod = _import_module()
        assert mod.text_appears_on_screen("", "screen text") is False

    def test_empty_screen_returns_false(self):
        mod = _import_module()
        assert mod.text_appears_on_screen("some expected text here", "") is False

    def test_case_insensitive(self):
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Hello World This Is Important",
            "hello world this is important"
        ) is True

    def test_whitespace_normalized(self):
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Hello   world\nthis is\ta test sentence",
            "Hello world this is a test sentence"
        ) is True

    def test_threshold_boundary(self):
        mod = _import_module()
        # Exactly at the boundary — half the text matches
        expected = "abcdefghijklmnopqrstuvwxyz"
        screen = "abcdefghijklm totally different"
        result = mod.text_appears_on_screen(expected, screen)
        assert result is True  # 13/26 = 50% coverage

    def test_below_threshold(self):
        mod = _import_module()
        expected = "abcdefghijklmnopqrstuvwxyz"
        screen = "abcde totally completely different text here"
        result = mod.text_appears_on_screen(expected, screen)
        assert result is False  # 5/26 = ~19% coverage

    def test_text_appended_to_existing_content(self):
        """Pasted text appended after existing text in a field should match."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "This is the newly dictated text that was just pasted",
            "Some existing content in the field This is the newly dictated text that was just pasted more chrome"
        ) is True

    def test_text_split_across_ocr_observations(self):
        """OCR may split text across line boundaries — should still match."""
        mod = _import_module()
        # The OCR returns observations that get joined with spaces,
        # potentially splitting words at line breaks
        assert mod.text_appears_on_screen(
            "Hello world this is a test sentence for OCR verification",
            "menu bar stuff Hello world this is a test sentence for OCR verific ation bottom bar"
        ) is True

    def test_ocr_minor_errors_still_match(self):
        """OCR may misread a few characters — should still match above threshold."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "The quick brown fox jumps over the lazy dog",
            "chrome The quicK brown fox jumps 0ver the lazy dog more stuff"
        ) is True

    def test_text_partially_clipped_at_edge(self):
        """Text near screen edge may be partially clipped by OCR."""
        mod = _import_module()
        # Only the first 70% of the text is visible
        expected = "This is a long dictated sentence that goes to the edge of the screen"
        visible = expected[:int(len(expected) * 0.7)]
        assert mod.text_appears_on_screen(
            expected,
            f"some chrome {visible} more stuff"
        ) is True

    def test_word_with_context_matches(self):
        """Distinctive word + adjacent character from expected confirms paste."""
        mod = _import_module()
        # "surprising results" — "g r" context around "results" confirms
        assert mod.text_appears_on_screen(
            "The preliminary investigation revealed surprising results",
            "totally unrelated surprising results more unrelated"
        ) is True

    def test_single_chrome_word_does_not_match(self):
        """A single word matching UI chrome without context should NOT confirm."""
        mod = _import_module()
        # "settings" appears in chrome but without the adjacent chars from dictation
        assert mod._has_distinctive_word_match(
            "please open the settings panel and adjust volume",
            "system settings general about privacy security"
        ) is False

    def test_word_with_right_context_matches(self):
        """Word + one char of right context from expected should match."""
        mod = _import_module()
        # "settings p" (word + space + first char of next word) is in screen
        assert mod._has_distinctive_word_match(
            "please open the settings panel and adjust volume",
            "other stuff settings panel more stuff"
        ) is True

    def test_word_with_left_context_matches(self):
        """One char of left context + word from expected should match."""
        mod = _import_module()
        # "e settings" (last char of prev word + space + word) is in screen
        assert mod._has_distinctive_word_match(
            "please open the settings panel and adjust volume",
            "other stuff e settings more stuff"
        ) is True

    def test_stopwords_only_do_not_match(self):
        """Common words alone should not trigger a false positive."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "The quick brown fox jumps over the lazy dog",
            "the and is to for with from that this"
        ) is False

    def test_url_bar_overflow_with_context_visible(self):
        """Long text pasted into URL bar — most clipped, word+context visible."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Please navigate to the authentication dashboard and check credentials",
            "chrome tabs authentication dashboard browser stuff"
        ) is True
