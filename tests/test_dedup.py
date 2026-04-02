"""Tests for transcription cleanup helpers."""

import logging

from spoke.dedup import truncate_repetition, is_hallucination, repair_ontology_terms


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

    def test_short_phrase_many_repeats(self):
        """Short repeated phrases like 'on this' should be caught even with many reps."""
        text = "Alright, " + "on this " * 38 + "on"
        result = truncate_repetition(text)
        assert result.count("on this") <= 2
        assert len(result) < 50


class TestIsHallucination:
    """Test Whisper silence hallucination detection."""

    def test_thank_you_is_hallucination(self):
        assert is_hallucination("Thank you.") is True

    def test_thank_you_case_insensitive(self):
        assert is_hallucination("thank you.") is True
        assert is_hallucination("THANK YOU.") is True

    def test_thanks_for_watching(self):
        assert is_hallucination("Thanks for watching.") is True

    def test_bye(self):
        assert is_hallucination("Bye.") is True

    def test_empty_string(self):
        assert is_hallucination("") is True

    def test_whitespace_only(self):
        assert is_hallucination("   ") is True

    def test_real_text_is_not_hallucination(self):
        assert is_hallucination("I want to thank you for your help.") is False

    def test_thank_you_in_longer_text(self):
        assert is_hallucination("Thank you for coming today.") is False


class TestRepairOntologyTerms:
    """Test log-backed ontology vocabulary repairs."""

    def test_repairs_observed_epistaxis_variants(self):
        text = "Leave a review ticket for it in Epistaxes and read Nepistaxis."
        assert repair_ontology_terms(text) == "Leave a review ticket for it in Epistaxis and read Epistaxis."

    def test_repairs_concatenated_epistaxis_topos(self):
        text = "Are we including that in our Epistaxistopos?"
        assert repair_ontology_terms(text) == "Are we including that in our Epistaxis topos?"

    def test_repairs_observed_topoi_and_anaphora_variants(self):
        text = "Epistexis Topoie and an Afro are both wrong."
        assert repair_ontology_terms(text) == "Epistaxis topoi and anaphora are both wrong."

    def test_logs_when_repair_fires(self, caplog):
        with caplog.at_level(logging.INFO, logger="spoke.dedup"):
            repaired = repair_ontology_terms("And then read Epistaxes.")

        assert repaired == "And then read Epistaxis."
        assert "Repaired ontology vocabulary" in caplog.text

    def test_leaves_unrelated_text_unchanged(self):
        text = "This is ordinary dictation with no Greek ontology vocabulary."
        assert repair_ontology_terms(text) == text
