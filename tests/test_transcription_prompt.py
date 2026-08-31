from pathlib import Path

from spoke.transcription_prompt import TranscriptionPromptProvider


def test_prompt_provider_reloads_caller_owned_file_and_preserves_all_text(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    first_file_text = "Kaminos, Trellis2MLX, CoreML, ANE."
    prompt_path.write_text(first_file_text, encoding="utf-8")
    inline = "Spoke special vocabulary. " * 500
    provider = TranscriptionPromptProvider(
        path=prompt_path,
        inline=inline,
        include_builtin=False,
    )

    first = provider.resolve()

    assert first.text == f"{first_file_text}\n{inline.strip()}"
    assert first.sources == (f"file:{prompt_path}", "env:SPOKE_TRANSCRIPTION_PROMPT")
    assert first.char_count == len(first.text)

    prompt_path.write_text("Perceptasia, Grapheus.", encoding="utf-8")
    second = provider.resolve()

    assert second.text == f"Perceptasia, Grapheus.\n{inline.strip()}"
    assert second.sha256 != first.sha256


def test_prompt_provider_receipt_distinguishes_requested_from_effective(tmp_path):
    provider = TranscriptionPromptProvider(
        path=Path(tmp_path / "missing.txt"),
        inline="Spoke, WhisperKit.",
        include_builtin=False,
    )

    prompt = provider.resolve()

    assert prompt.receipt(supported=False, effective=False) == {
        "schema": "spoke.transcription-prompt.v1",
        "requested": True,
        "supported": False,
        "effective": False,
        "sha256": prompt.sha256,
        "char_count": len("Spoke, WhisperKit."),
        "sources": ["env:SPOKE_TRANSCRIPTION_PROMPT"],
    }
