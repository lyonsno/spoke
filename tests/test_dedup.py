"""Tests for transcription cleanup helpers."""

import logging

from spoke.dedup import (
    is_hallucination,
    ontology_term_spans,
    repair_ontology_terms,
    truncate_repetition,
)


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
        assert repair_ontology_terms(text) == "Leave a review ticket for it in Epístaxis and read Epístaxis."

    def test_repairs_concatenated_epistaxis_topos(self):
        text = "Are we including that in our Epistaxistopos?"
        assert repair_ontology_terms(text) == "Are we including that in our Epístaxis tópos?"

    def test_repairs_observed_topoi_and_anaphora_variants(self):
        text = "Epistexis Topoie and an Afro are both wrong."
        assert repair_ontology_terms(text) == "Epístaxis Tópoi and anaphorá are both wrong."

    def test_repairs_recent_high_frequency_variants(self):
        text = (
            "Read spoke-up as taxes, chase the Metadose II, check the Uxis "
            "document, and compile our tipos into a Syllogy."
        )
        assert repair_ontology_terms(text) == (
            "Read spoke Epístaxis, chase the Metádosis, check the Aúxesis "
            "document, and compile our tópos into a Syllogé."
        )

    def test_repairs_recent_sylloge_and_new_ontology_variants(self):
        text = "Check the sueji, the kerigma badge, and epinorthosis."
        assert repair_ontology_terms(text) == (
            "Check the syllogé, the kérygma badge, and epanórthosis."
        )

    def test_repairs_aposkepsis_semiosis_and_katastasis_variants(self):
        text = "The appless kept says naming and semi-hostess concept are close to Catastasis."
        assert repair_ontology_terms(text) == (
            "The aposképsis naming and sēmeiōsis are close to Katástasis."
        )

    def test_repairs_in_his_taxes_phrase(self):
        text = "When you see in his taxes in the logs, that should be Epistaxis."
        assert repair_ontology_terms(text) == (
            "When you see Epístaxis in the logs, that should be Epístaxis."
        )

    def test_repairs_recent_smoke_sentence_to_accented_canonical_forms(self):
        text = (
            "Nice work. Thank you. I'm gonna test now epistaxis Epinorthosis lysis, "
            "Syllogy Episcapsis probly anaphora Charygma otopoiesis auxesus "
            "Epistaxis Epinorthosis Semian Charygma Semian"
        )
        assert repair_ontology_terms(text) == (
            "Nice work. Thank you. I'm gonna test now Epístaxis Epanórthosis lýsis, "
            "Syllogé Aposképsis probolé anaphorá Kérygma autopoíesis aúxesis "
            "Epístaxis Epanórthosis Sēmeion Kérygma Sēmeion"
        )

    def test_repairs_safe_nonword_followup_regressions(self):
        text = "Probally chorigma ooxisis epispokosis and probaly should all normalize."
        assert repair_ontology_terms(text) == (
            "Probolé kérygma aúxesis epispókisis and probolé should all normalize."
        )

    def test_logs_when_repair_fires(self, caplog):
        with caplog.at_level(logging.INFO, logger="spoke.dedup"):
            repaired = repair_ontology_terms("And then read Epistaxes.")

        assert repaired == "And then read Epístaxis."
        assert "Repaired ontology vocabulary" in caplog.text

    def test_leaves_unrelated_text_unchanged(self):
        text = "This is ordinary dictation with no Greek ontology vocabulary."
        assert repair_ontology_terms(text) == text

    def test_reports_ranges_for_plain_and_accented_ontology_forms(self):
        text = "Epístaxis and sēmeion beside plain Epistaxis."
        spans = ontology_term_spans(text)

        assert [text[start:end] for start, end in spans] == [
            "Epístaxis",
            "sēmeion",
            "Epistaxis",
        ]
