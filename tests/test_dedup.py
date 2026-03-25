"""Tests for repetition loop detection and truncation."""

from donttype.dedup import truncate_repetition


class TestTruncateRepetition:
    """Test Whisper repetition loop guard."""

    def test_no_repetition_unchanged(self):
        text = "Hello world, this is a normal sentence."
        assert truncate_repetition(text) == text

    def test_empty_string(self):
        assert truncate_repetition("") == ""

    def test_short_string(self):
        assert truncate_repetition("Hi") == "Hi"

    def test_detects_word_repetition(self):
        text = "I think so. so. so. so. so."
        result = truncate_repetition(text)
        # Should truncate the repeated "so." instances
        assert result.count("so.") < text.count("so.")

    def test_detects_phrase_repetition(self):
        text = "The weather is nice. The weather is nice. The weather is nice. The weather is nice."
        result = truncate_repetition(text)
        # Keeps at most 2 — the original phrase plus one repetition boundary
        assert result.count("The weather is nice.") <= 2
        assert len(result) < len(text)

    def test_preserves_legitimate_text(self):
        text = "Yes yes, I agree. But no no, that won't work."
        assert truncate_repetition(text) == text

    def test_detects_long_phrase_loop(self):
        phrase = "I think we should probably go ahead and do that. "
        text = phrase * 5
        result = truncate_repetition(text)
        assert len(result) < len(text)

    def test_minimum_repeats_threshold(self):
        # Only 2 repeats — below default threshold of 3
        text = "hello. hello."
        assert truncate_repetition(text) == text

    def test_three_repeats_triggers(self):
        text = "okay. okay. okay. okay."
        result = truncate_repetition(text, min_repeats=3)
        assert len(result) < len(text)
