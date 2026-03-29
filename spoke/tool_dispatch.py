"""Tool dispatch for screen context.

Defines the tool schemas for capture_context and read_aloud, handles
accumulation of streamed tool call deltas, and executes tools locally.

See docs/screen-context-v1.md for the design.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from spoke.scene_capture import SceneCaptureCache

logger = logging.getLogger(__name__)


# ── Tool schemas (OpenAI function calling format) ────────────────


_CAPTURE_CONTEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "capture_context",
        "description": (
            "Capture the frontmost app's active window (or full screen as "
            "fallback). Returns structured metadata, OCR text blocks with "
            "refs, and optional accessibility hints. Use this when the user "
            "refers to something visible on screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["active_window", "screen"],
                    "description": (
                        "What to capture. 'active_window' captures only the "
                        "frontmost app's window (preferred). 'screen' captures "
                        "the entire main screen."
                    ),
                },
            },
            "required": [],
        },
    },
}

_READ_ALOUD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_aloud",
        "description": (
            "Resolve a source ref to exact text and speak it aloud via TTS. "
            "Use scene_block or ax_hint refs from a previous capture_context "
            "call, or use clipboard/selection/last_response/literal refs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_ref": {
                    "type": "string",
                    "description": (
                        "A source ref string. Format: 'kind:value'. "
                        "Examples: 'scene_block:scene-abc:block-1', "
                        "'ax_hint:scene-abc:focus', 'clipboard:current', "
                        "'selection:frontmost', 'last_response:current', "
                        "'literal:text to speak'."
                    ),
                },
            },
            "required": ["source_ref"],
        },
    },
}


def get_tool_schemas() -> list[dict]:
    """Return the tool schemas for the assistant."""
    return [_CAPTURE_CONTEXT_SCHEMA, _READ_ALOUD_SCHEMA]


# ── Tool call accumulation ───────────────────────────────────────


class ToolCallAccumulator:
    """Accumulates streamed tool call deltas into complete tool calls.

    OpenAI-compatible SSE streams send tool calls as incremental deltas
    with an index field. This class collects them and produces the final
    list of tool calls when the stream ends.
    """

    def __init__(self):
        self._calls: dict[int, dict] = {}

    @property
    def has_calls(self) -> bool:
        return len(self._calls) > 0

    def feed(self, delta: dict) -> None:
        """Feed a single tool_calls delta entry."""
        index = delta.get("index", 0)

        if index not in self._calls:
            self._calls[index] = {
                "id": delta.get("id", ""),
                "type": delta.get("type", "function"),
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }

        call = self._calls[index]

        # Update id if present
        if "id" in delta and delta["id"]:
            call["id"] = delta["id"]

        # Update function name/arguments
        fn = delta.get("function", {})
        if fn.get("name"):
            call["function"]["name"] = fn["name"]
        if "arguments" in fn:
            call["function"]["arguments"] += fn["arguments"]

    def finish(self) -> list[dict]:
        """Return the accumulated tool calls, ordered by index."""
        if not self._calls:
            return []
        return [self._calls[i] for i in sorted(self._calls.keys())]


# ── Tool execution ───────────────────────────────────────────────


def _execute_capture_context(
    arguments: dict,
    scene_cache: SceneCaptureCache | None = None,
) -> Any:
    """Execute capture_context and return the SceneCapture."""
    from spoke.scene_capture import capture_context

    scope = arguments.get("scope", "active_window")
    return capture_context(scope=scope, cache=scene_cache)


def _execute_read_aloud(
    arguments: dict,
    scene_cache: SceneCaptureCache | None = None,
    last_response: str | None = None,
    tts_client: Any | None = None,
) -> str:
    """Execute read_aloud: resolve the ref and speak it."""
    from spoke.source_ref import RefResolutionError, SourceRef, parse_ref, resolve

    ref_str = arguments.get("source_ref", "")
    try:
        ref = parse_ref(ref_str)
        text = resolve(ref, scene_cache=scene_cache, last_response=last_response)
    except (ValueError, RefResolutionError) as e:
        return f"Error resolving ref: {e}"

    # Speak via TTS if available
    if tts_client is not None:
        try:
            tts_client.speak(text)
        except Exception:
            logger.warning("TTS playback failed", exc_info=True)

    return f"Spoke: {text}"


def execute_tool(
    name: str,
    arguments: dict,
    *,
    scene_cache: SceneCaptureCache | None = None,
    last_response: str | None = None,
    tts_client: Any | None = None,
) -> str:
    """Execute a tool by name and return the result as a JSON string.

    Returns a JSON-encoded result for the model to consume.
    """
    if name == "capture_context":
        capture = _execute_capture_context(arguments, scene_cache=scene_cache)
        if capture is None:
            return json.dumps({"error": "Capture failed"})

        # Build the return shape from the design doc
        result = {
            "scene_ref": capture.scene_ref,
            "scope": capture.scope,
            "app_name": capture.app_name,
            "window_title": capture.window_title,
            "ocr_blocks": [
                {
                    "ref": b.ref,
                    "text": b.text,
                    "bbox": list(b.bbox),
                }
                for b in capture.ocr_blocks
            ],
            "ax_hints": [
                {
                    "ref": h.ref,
                    "role": h.role,
                    "label": h.label,
                }
                for h in capture.ax_hints
            ],
        }
        return json.dumps(result)

    elif name == "read_aloud":
        return _execute_read_aloud(
            arguments,
            scene_cache=scene_cache,
            last_response=last_response,
            tts_client=tts_client,
        )

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})
