"""Tests for post-paste OCR verification."""

import builtins
import importlib
import logging
import sys
import types
from unittest.mock import patch


def _import_module():
    sys.modules.pop("spoke.paste_verify", None)
    return importlib.import_module("spoke.paste_verify")


def _install_fake_ocr_modules(
    monkeypatch, *, image="fake-image", success=True, error=None, lines=(), capture=None
):
    class _FakeCandidate:
        def __init__(self, text):
            self._text = text

        def string(self):
            return self._text

    class _FakeObservation:
        def __init__(self, text):
            self._text = text

        def topCandidates_(self, count):
            return [_FakeCandidate(self._text)]

    observations = [_FakeObservation(line) for line in lines]

    class _FakeRequest:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            if capture is not None:
                capture["request"] = self
            return self

        def setRecognitionLevel_(self, level):
            self.level = level

        def setUsesLanguageCorrection_(self, enabled):
            self.enabled = enabled

        def results(self):
            return observations

    class _FakeHandler:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithCGImage_options_(self, cgimage, options):
            self.cgimage = cgimage
            self.options = options
            return self

        def performRequests_error_(self, requests, options):
            return success, error

    monkeypatch.setitem(
        sys.modules,
        "Vision",
        types.SimpleNamespace(
            VNRecognizeTextRequest=_FakeRequest,
            VNImageRequestHandler=_FakeHandler,
            VNRequestTextRecognitionLevelAccurate="accurate",
            VNRequestTextRecognitionLevelFast="fast",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "Quartz",
        types.SimpleNamespace(
            CGWindowListCreateImage=lambda *args: image,
            kCGWindowListOptionOnScreenOnly=1,
            kCGNullWindowID=0,
            CGRectInfinite="infinite",
        ),
    )


def _dense_background(*segments):
    """Build a text-dense OCR scene with lots of unrelated chrome-like noise."""
    noise = [
        "menu bar file edit view history window help",
        "project notes inbox account security notifications",
        "recent activity dashboard settings browser tabs sidebar",
        "search results panel status indicators connection details",
    ]
    return " ".join([*noise[:2], *segments, *noise[2:]])


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

    def test_dense_background_scattered_words_do_not_match(self):
        """Scattered expected words in a dense background should not count as success."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "The preliminary investigation revealed surprising results",
            _dense_background(
                "investigation tools revealed panel with surprising charts",
                "result settings and unrelated summaries",
            ),
        ) is False

    def test_dense_background_partial_phrase_without_strong_context_does_not_match(self):
        """A tempting middle phrase alone should not survive a text-dense scene."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Please navigate to the authentication dashboard and check credentials",
            _dense_background(
                "authentication settings dashboard chrome credentials help center",
                "account security authentication dashboard",
            ),
        ) is False

    def test_dense_background_end_phrase_with_context_still_matches(self):
        """A strong boundary phrase should still confirm success amid dense background."""
        mod = _import_module()
        assert mod.text_appears_on_screen(
            "Please open the quarterly revenue workbook and verify surprising results",
            _dense_background(
                "clipboard history unrelated tabs and account chrome",
                "verify surprising results more unrelated text",
            ),
        ) is True

    def test_long_paste_in_noisy_terminal_scene_matches_local_window(self):
        """A long pasted line in a terminal should survive surrounding OCR noise."""
        mod = _import_module()
        expected = (
            "yeah well i just saw some smoke it pasted successfully into wes term "
            "where it cannot read the field and where ocr should be helping us "
            "and ocr did not help us it still caught it anyway"
        )
        screen = _dense_background(
            "shell prompt build output status panel sidebar notifications",
            "yeah well i just saw some sm0ke it pasted successfully into wes term "
            "where it cannot read the field and where ocr should be helping us "
            "and ocr did n0t help us it still caught it anyway",
            "git status branch ahead prompt terminal session tab bar",
        )
        assert mod.text_appears_on_screen(expected, screen) is True

    def test_ordered_distinctive_words_confirm_terminal_ocr_with_interleaved_noise(self):
        """An ordered run of distinctive words should confirm even with noisy OCR gaps."""
        mod = _import_module()
        expected = "alright cool so are we ready to bring this year to katastasis then"
        screen = (
            "terminal prompt alright menu cool status ready sidebar bring output "
            "year logs katastroasis pane then shell history tabs"
        )
        assert mod.text_appears_on_screen(expected, screen) is True

    def test_scattered_ordered_words_with_wide_gaps_do_not_confirm(self):
        """Ordered words spread too far across the scene should still fail."""
        mod = _import_module()
        expected = "alright cool so are we ready to bring this year to katastasis then"
        screen = (
            "alright menu account security cool browser project notes ready window "
            "help center bring git branch status year notifications recent activity "
            "katastasis sidebar connection info then"
        )
        assert mod.text_appears_on_screen(expected, screen) is False


class TestCaptureScreenText:
    def test_prefers_active_window_ocr_text(self):
        mod = _import_module()

        with patch.object(mod, "_capture_active_window_text", return_value="window text"), \
             patch.object(mod, "_capture_full_screen_text") as mock_full:
            assert mod.capture_screen_text() == "window text"

        mock_full.assert_not_called()

    def test_missing_vision_logs_warning(self, monkeypatch, caplog):
        mod = _import_module()
        real_import = builtins.__import__

        def _import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "Vision":
                raise ModuleNotFoundError("No module named 'Vision'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _import)

        with patch.object(mod, "_capture_active_window_text", return_value=""), \
             caplog.at_level(logging.WARNING):
            assert mod.capture_screen_text() == ""

        assert "Paste verify OCR unavailable" in caplog.text

    def test_screen_capture_none_logs_warning(self, monkeypatch, caplog):
        mod = _import_module()
        _install_fake_ocr_modules(monkeypatch, image=None)

        with patch.object(mod, "_capture_active_window_text", return_value=""), \
             caplog.at_level(logging.WARNING):
            assert mod.capture_screen_text() == ""

        assert "screen capture returned no image" in caplog.text

    def test_success_returns_joined_text(self, monkeypatch):
        mod = _import_module()
        _install_fake_ocr_modules(monkeypatch, lines=("first line", "second line"))

        with patch.object(mod, "_capture_active_window_text", return_value=""):
            assert mod.capture_screen_text() == "first line second line"

    def test_full_screen_fallback_uses_accurate_recognition(self, monkeypatch):
        mod = _import_module()
        capture = {}
        _install_fake_ocr_modules(
            monkeypatch,
            lines=("first line", "second line"),
            capture=capture,
        )

        with patch.object(mod, "_capture_active_window_text", return_value=""):
            assert mod.capture_screen_text() == "first line second line"

        assert capture["request"].level == "accurate"


class TestClassifyPasteResult:
    def test_empty_capture_is_unavailable(self):
        mod = _import_module()

        assert mod.classify_paste_result("expected dictated text", "") == "unavailable"

    def test_match_is_confirmed(self):
        mod = _import_module()

        assert (
            mod.classify_paste_result(
                "Hello world this is a test sentence",
                "Some UI chrome Hello world this is a test sentence more stuff",
            )
            == "confirmed"
        )

    def test_non_match_with_capture_is_missing(self):
        mod = _import_module()

        assert (
            mod.classify_paste_result(
                "Please navigate to the authentication dashboard and check credentials",
                "totally unrelated browser chrome account avatar preferences",
            )
            == "missing"
        )

    def test_match_with_preexisting_signal_is_ambiguous(self):
        mod = _import_module()

        assert (
            mod.classify_paste_result(
                "Hello world this is a test sentence",
                "Some UI chrome Hello world this is a test sentence more stuff",
                preexisting_match=True,
            )
            == "ambiguous"
        )
