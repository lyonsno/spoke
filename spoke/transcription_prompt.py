"""Live, source-identifiable context prompts for transcription backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPT_SCHEMA = "spoke.transcription-prompt.v1"
_DEFAULT_PROMPT_PATH = Path.home() / ".config" / "spoke" / "transcription_prompt.txt"
_FALSE_VALUES = {"0", "false", "no", "off"}
_BUILTIN_PROMPT = (
    "Vocabulary: Spoke, operator memory, Kaminos, Perceptasia, Grapheus, "
    "WhisperKit, MLX, CoreML, ANE, Trellis2MLX."
)


@dataclass(frozen=True)
class TranscriptionPrompt:
    text: str
    sources: tuple[str, ...]
    sha256: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    def receipt(self, *, supported: bool, effective: bool) -> dict:
        return {
            "schema": _PROMPT_SCHEMA,
            "requested": bool(self.text),
            "supported": supported,
            "effective": effective,
            "sha256": self.sha256,
            "char_count": self.char_count,
            "sources": list(self.sources),
        }


class TranscriptionPromptProvider:
    """Resolve built-in, file, and inline prompt context for each utterance."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        inline: str = "",
        include_builtin: bool = True,
    ) -> None:
        self._path = path
        self._inline = inline
        self._include_builtin = include_builtin

    @classmethod
    def from_environment(cls) -> "TranscriptionPromptProvider":
        path_value = os.environ.get("SPOKE_TRANSCRIPTION_PROMPT_PATH", "").strip()
        path = Path(path_value).expanduser() if path_value else _DEFAULT_PROMPT_PATH
        include_builtin = (
            os.environ.get("SPOKE_TRANSCRIPTION_PROMPT_BUILTIN", "1").strip().lower()
            not in _FALSE_VALUES
        )
        return cls(
            path=path,
            inline=os.environ.get("SPOKE_TRANSCRIPTION_PROMPT", ""),
            include_builtin=include_builtin,
        )

    def resolve(self) -> TranscriptionPrompt:
        parts: list[str] = []
        sources: list[str] = []
        if self._include_builtin:
            parts.append(_BUILTIN_PROMPT)
            sources.append("builtin:spoke-vocabulary-v1")

        if self._path is not None:
            try:
                file_text = self._path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                file_text = ""
            except OSError:
                logger.warning(
                    "Failed to read transcription prompt file %s",
                    self._path,
                    exc_info=True,
                )
                file_text = ""
            if file_text:
                parts.append(file_text)
                sources.append(f"file:{self._path}")

        inline = self._inline.strip()
        if inline:
            parts.append(inline)
            sources.append("env:SPOKE_TRANSCRIPTION_PROMPT")

        text = "\n".join(parts)
        return TranscriptionPrompt(
            text=text,
            sources=tuple(sources),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
