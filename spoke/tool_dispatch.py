"""Tool dispatch for screen context and bounded local operator actions.

Defines the tool schemas for capture_context and read_aloud, handles
accumulation of streamed tool call deltas, and executes tools locally.

See docs/screen-context-v1.md for the design.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Filesystem tools resolve relative paths against ~/dev so the model
# can use short paths like "epistaxis/projects/spoke/epistaxis.md"
# regardless of which launcher target started the process.
_TOOLS_HOME = os.path.expanduser("~/dev")

from spoke.epistaxis_operator import (
    EpistaxisOperator,
    EpistaxisOperatorError,
    tool_schema as epistaxis_tool_schema,
)
from spoke.brave_search_operator import (
    BraveSearchOperator,
    BraveSearchOperatorError,
    tool_schema as brave_search_tool_schema,
)
from spoke.gmail_operator import (
    GmailOperator,
    GmailOperatorError,
    tool_schema as gmail_tool_schema,
)
from spoke.terminal_operator import (
    TerminalOperator,
    TerminalOperatorError,
    tool_schema as terminal_tool_schema,
)
from spoke.scene_capture import SceneCaptureCache

logger = logging.getLogger(__name__)

_EDIT_FILE_TELEMETRY_PATH = Path.home() / ".config" / "spoke" / "edit-file-telemetry.jsonl"
_EDIT_FILE_TELEMETRY_COUNTER_KEYS = (
    "total",
    "success",
    "not_found",
    "not_unique",
    "malformed_request",
    "normalization_assisted",
)


def _is_local_omnivoice_cold_tts(tts_client: Any) -> bool:
    """Whether the active TTS client is a cold local OmniVoice instance."""
    model_id = getattr(tts_client, "_model_id", "")
    if not isinstance(model_id, str) or model_id.strip().lower() != "k2-fsa/omnivoice":
        return False
    if getattr(tts_client, "_model", None) is not None:
        return False
    base_url = getattr(tts_client, "_base_url", "")
    return not bool(base_url)


def _omnivoice_warmup_inflight(tts_client: Any) -> bool:
    warming = getattr(tts_client, "is_warming", False)
    return warming if isinstance(warming, bool) else False


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
                "include_image": {
                    "type": "boolean",
                    "description": (
                        "When true, attach a downscaled model-facing screenshot "
                        "on vision-capable backends alongside the OCR refs. "
                        "Text-only backends ignore this flag."
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
            "call, or use clipboard/selection/last_response/literal refs. "
            "Use a literal ref when the user wants an arbitrary phrase or sentence spoken."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_ref": {
                    "type": "string",
                    "description": (
                        "A source ref string. Pass block refs from "
                        "capture_context directly (e.g., 'scene-abc:block-1'). "
                        "Other formats: 'clipboard:current', "
                        "'selection:frontmost', 'last_response:current', "
                        "'literal:text to speak'. Use literal when you need to "
                        "speak exact text directly rather than reading it from another source."
                    ),
                },
            },
            "required": ["source_ref"],
        },
    },
}

_ADD_TO_TRAY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_to_tray",
        "description": (
            "Place exact text into the tray for later insertion or sending. "
            "Use this when the user wants to save, hold onto, or keep "
            "something for later rather than speaking it aloud immediately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The exact text to place into the tray. Preserve the "
                        "content literally unless the user explicitly asked "
                        "for a summary or rewrite."
                    ),
                },
            },
            "required": ["text"],
        },
    },
}



_LIST_DIRECTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": (
            "List contents of a directory with metadata. Returns name, type "
            "(file/dir), size, and modification date for each entry. "
            "Supports an optional glob pattern to filter entries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "Absolute or relative directory path"},
                "pattern": {"type": "string", "description": "Optional glob pattern to filter entries (e.g. '*.md', 'epistaxis*')"},
            },
            "required": ["dir_path"]
        }
    }
}

_READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file. For large files, it returns an outline of functions/classes instead, requiring you to make a follow-up call with start_line and end_line to read specific sections.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute or relative file path"},
                "start_line": {"type": "integer", "description": "Optional 1-based line number to start reading from"},
                "end_line": {"type": "integer", "description": "Optional 1-based line number to end reading at (inclusive)"}
            },
            "required": ["file_path"]
        }
    }
}

_WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write text content to a file, overwriting if it exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute or relative file path"},
                "content": {"type": "string", "description": "Full text content to write"}
            },
            "required": ["file_path", "content"]
        }
    }
}

_EDIT_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Apply a uniquely matching targeted text edit to an existing file. "
            "The edit succeeds only when old_string matches exactly one location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Absolute or relative file path"},
                "old_string": {"type": "string", "description": "Exact text to replace; must match exactly one location"},
                "new_string": {"type": "string", "description": "Replacement text to write in place of old_string"},
            },
            "required": ["file", "old_string", "new_string"],
        },
    },
}

_SEARCH_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_file",
        "description": "Search file contents in a directory using grep or equivalent.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or string pattern to search"},
                "dir_path": {"type": "string", "description": "Directory path to search in (recursive)"}
            },
            "required": ["pattern", "dir_path"]
        }
    }
}

_FIND_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_file",
        "description": (
            "Find files by name pattern. Recursively searches a directory for "
            "files matching a glob pattern (e.g. 'epistaxis.md', '*.py', "
            "'spoke/**/attractors/*.md'). Returns matching paths with size and "
            "modification date. Much faster than listing directories one by one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match filenames (e.g. '*.md', 'epistaxis.md', '**/*.py')"},
                "dir_path": {"type": "string", "description": "Root directory to search from (recursive)"},
            },
            "required": ["pattern", "dir_path"]
        }
    }
}
_RUN_EPISTAXIS_OPS_SCHEMA = epistaxis_tool_schema()
_SEARCH_WEB_SCHEMA = brave_search_tool_schema()
_QUERY_GMAIL_SCHEMA = gmail_tool_schema()
_RUN_TERMINAL_COMMAND_SCHEMA = terminal_tool_schema()
_LAUNCH_SUBAGENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "launch_subagent",
        "description": (
            "Launch an operator-owned background subagent job. The current "
            "supported kind is 'search' for bounded local file/code search."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["search"],
                    "description": "Type of subagent to launch. Currently only 'search' is supported.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Concrete search task for the background subagent.",
                },
            },
            "required": ["kind", "prompt"],
        },
    },
}
_LIST_SUBAGENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_subagents",
        "description": "List background subagent jobs and their current states.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    },
}
_GET_SUBAGENT_RESULT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_subagent_result",
        "description": "Fetch status or final output for a specific background subagent job.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "The subagent id returned by launch_subagent.",
                },
            },
            "required": ["subagent_id"],
        },
    },
}
_CANCEL_SUBAGENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cancel_subagent",
        "description": "Request cancellation for a running background subagent job.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "The subagent id returned by launch_subagent.",
                },
            },
            "required": ["subagent_id"],
        },
    },
}
_COMPACT_HISTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compact_history",
        "description": (
            "Compact the conversation history to reduce context size. "
            "Three modes:\n"
            "- drop_tool_results: strip tool call/result messages from "
            "the oldest N turns, keeping user and assistant text.\n"
            "- summarize: replace the oldest N turns with a summary you "
            "provide.\n"
            "- guided: attractor-aware compaction. The tool reads the "
            "full attractor set, cross-references against the conversation "
            "history being compacted, and returns retention flags — a short "
            "list of things you must preserve because they connect to "
            "durable intent. Call this first, then call again with "
            "mode='summarize' using the flags to guide your summary. "
            "The flags are the safety net; your conversational judgment "
            "handles everything else."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["drop_tool_results", "summarize", "guided"],
                    "description": (
                        "drop_tool_results: strip tool messages from oldest N turns. "
                        "summarize: replace oldest N turns with a summary. "
                        "guided: return attractor-aware retention flags for the "
                        "oldest N turns, then follow up with summarize."
                    ),
                },
                "n": {
                    "type": "integer",
                    "description": (
                        "Number of oldest turns to compact. 0 means all turns."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Required when mode='summarize'. A brief summary of "
                        "the compacted turns. When following a guided call, "
                        "incorporate the retention flags."
                    ),
                },
            },
            "required": ["mode", "n"],
        },
    },
}


def get_tool_schemas() -> list[dict]:
    """Return the tool schemas for the assistant."""
    return [
        _CAPTURE_CONTEXT_SCHEMA,
        _READ_ALOUD_SCHEMA,
        _ADD_TO_TRAY_SCHEMA,
        _LIST_DIRECTORY_SCHEMA,
        _READ_FILE_SCHEMA,
        _WRITE_FILE_SCHEMA,
        _EDIT_FILE_SCHEMA,
        _SEARCH_FILE_SCHEMA,
        _FIND_FILE_SCHEMA,
        _RUN_EPISTAXIS_OPS_SCHEMA,
        _SEARCH_WEB_SCHEMA,
        _QUERY_GMAIL_SCHEMA,
        _RUN_TERMINAL_COMMAND_SCHEMA,
        _LAUNCH_SUBAGENT_SCHEMA,
        _LIST_SUBAGENTS_SCHEMA,
        _GET_SUBAGENT_RESULT_SCHEMA,
        _CANCEL_SUBAGENT_SCHEMA,
        _COMPACT_HISTORY_SCHEMA,
    ]


def get_search_subagent_tool_schemas() -> list[dict]:
    """Return the bounded read-only tool subset for search subagents."""
    return [
        _LIST_DIRECTORY_SCHEMA,
        _READ_FILE_SCHEMA,
        _SEARCH_FILE_SCHEMA,
        _FIND_FILE_SCHEMA,
    ]


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
    *,
    skip_ocr: bool | None = None,
) -> Any:
    """Execute capture_context and return the SceneCapture."""
    from spoke.scene_capture import capture_context

    scope = arguments.get("scope", "active_window")
    return capture_context(scope=scope, cache=scene_cache, skip_ocr=skip_ocr)


def _capture_context_result_dict(
    capture: Any,
    *,
    include_image: bool = False,
) -> dict[str, Any]:
    result = {
        "scene_ref": capture.scene_ref,
        "scope": capture.scope,
        "app_name": capture.app_name,
        "bundle_id": capture.bundle_id,
        "window_title": capture.window_title,
        "image_size": list(capture.image_size),
        "model_image_size": list(capture.model_image_size),
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
    if include_image and getattr(capture, "model_image_path", None):
        result["model_image"] = {
            "path": capture.model_image_path,
            "media_type": capture.model_image_media_type or "image/png",
            "size": list(capture.model_image_size),
        }
    return result


def _capture_context_multimodal_result(
    capture: Any,
    *,
    include_image: bool = True,
) -> dict[str, Any]:
    summary = _capture_context_result_dict(capture, include_image=False)
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(summary)}
    ]
    model_image_path = getattr(capture, "model_image_path", None)
    if include_image and model_image_path:
        try:
            with open(model_image_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{getattr(capture, 'model_image_media_type', None) or 'image/png'};"
                            f"base64,{encoded}"
                        )
                    },
                }
            )
        except OSError:
            logger.warning(
                "Failed to inline capture_context model image from %s",
                model_image_path,
                exc_info=True,
            )
    logger.info(
        "capture_context: multimodal payload scene_ref=%s image_attached=%s image_path=%r",
        summary.get("scene_ref"),
        len(parts) > 1,
        model_image_path,
    )
    return {
        "content": parts,
        "log_text": json.dumps(summary),
    }


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
        model_id = getattr(tts_client, "_model_id", "unknown")
        cancelled = getattr(tts_client, "_cancelled", "unknown")
        model_loaded = getattr(tts_client, "_model", None) is not None
        logger.info("read_aloud: tts_client present, model=%s, model_loaded=%s, cancelled=%s, text=%d chars",
                     model_id, model_loaded, cancelled, len(text))
        if _is_local_omnivoice_cold_tts(tts_client):
            wait_until_ready = getattr(tts_client, "wait_until_ready", None)
            if _omnivoice_warmup_inflight(tts_client) and callable(wait_until_ready):
                logger.info(
                    "read_aloud: waiting for local OmniVoice warmup to finish"
                )
                if wait_until_ready(timeout=15.0):
                    model_loaded = getattr(tts_client, "_model", None) is not None
                    logger.info(
                        "read_aloud: local OmniVoice warmup finished during command turn"
                    )
                else:
                    logger.warning(
                        "read_aloud: local OmniVoice warmup still in flight after timeout"
                    )
                    return (
                        "Error speaking text: Local OmniVoice TTS is still warming. "
                        "Retry in a moment once the background load finishes."
                    )
            if _is_local_omnivoice_cold_tts(tts_client):
                warm = getattr(tts_client, "warm", None)
                if callable(warm):
                    logger.info(
                        "read_aloud: starting background warmup for local OmniVoice"
                    )
                    warm()
                if callable(wait_until_ready) and wait_until_ready(timeout=5.0):
                    model_loaded = getattr(tts_client, "_model", None) is not None
                    logger.info(
                        "read_aloud: local OmniVoice warmup completed after cold start"
                    )
                if not _is_local_omnivoice_cold_tts(tts_client):
                    logger.info(
                        "read_aloud: local OmniVoice became ready after cold warm start"
                    )
                else:
                    logger.warning(
                        "read_aloud: local OmniVoice still cold after background warm start"
                    )
                    return (
                        "Error speaking text: Local OmniVoice TTS is loading in the background. "
                        "Try read_aloud again in a moment."
                    )
        try:
            logger.info("read_aloud: launching speak_async (non-blocking)")
            tts_client.speak_async(text)
            logger.info("read_aloud: speak_async launched")
            return f"Speaking: {text}"
        except Exception as exc:
            detail = str(exc)
            if hasattr(exc, "read"):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            elif hasattr(exc, "reason"):
                detail = f"{exc.reason} ({detail})"
            logger.warning("TTS playback failed: %s", detail, exc_info=True)
            return (
                f"Error speaking text: TTS playback failed. "
                f"model={model_id}, model_loaded={model_loaded}, "
                f"cancelled={cancelled}, error={type(exc).__name__}: {detail}"
            )
    else:
        logger.warning("read_aloud: no tts_client available")
        return (
            "Error: TTS client is not available. "
            "This usually means no TTS voice/backend is configured, "
            "or the TTS client failed to initialize at startup. "
            "Tell the user: TTS is not configured."
        )


def _execute_add_to_tray(
    arguments: dict,
    tray_writer: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Execute add_to_tray by handing exact text to the live tray surface."""
    text = arguments.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return {"error": "No tray text provided"}
    if tray_writer is None:
        return {"error": "Tray writer unavailable"}
    return tray_writer(text)













def _resolve_tool_path(p: str) -> str:
    """Resolve a path for filesystem tools.

    Expands ``~``, and resolves relative paths against ``_TOOLS_HOME``
    (~/dev) so the model can use short paths like ``epistaxis/...``.
    """
    expanded = os.path.expanduser(p)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(_TOOLS_HOME, expanded)


def _execute_list_directory(arguments: dict) -> dict[str, Any]:
    import fnmatch
    from datetime import datetime, timezone

    dir_path = _resolve_tool_path(arguments.get("dir_path") or ".")
    pattern = arguments.get("pattern")
    try:
        if not os.path.isdir(dir_path):
            return {"error": f"Not a valid directory: {dir_path}"}
        names = os.listdir(dir_path)
        if pattern:
            names = [n for n in names if fnmatch.fnmatch(n, pattern)]
        names.sort()
        entries = []
        max_entries = 50
        for name in names[:max_entries]:
            full = os.path.join(dir_path, name)
            try:
                st = os.stat(full)
                is_dir = os.path.isdir(full)
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                entries.append({
                    "name": name,
                    "type": "dir" if is_dir else "file",
                    "size": st.st_size if not is_dir else None,
                    "modified": mtime,
                })
            except OSError:
                entries.append({"name": name, "type": "unknown"})
        result: dict[str, Any] = {"dir_path": dir_path, "entries": entries}
        if len(names) > max_entries:
            result["truncated"] = f"Showing {max_entries} of {len(names)} entries"
        return result
    except Exception as e:
        return {"error": str(e)}

def _execute_read_file(arguments: dict) -> dict[str, Any]:
    import ast
    import re
    from itertools import islice

    raw_path = arguments.get("file_path")
    if not raw_path:
        return {"error": "file_path is required"}
    file_path = _resolve_tool_path(raw_path)

    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")

    if end_line == 0:
        return {"error": "end_line must be >= 1 if provided."}

    try:
        if not os.path.isfile(file_path):
            # Suggest similar files in the same directory
            suggestions = []
            parent = os.path.dirname(file_path) or "."
            basename = os.path.basename(file_path)
            if os.path.isdir(parent):
                from difflib import get_close_matches
                siblings = os.listdir(parent)
                suggestions = get_close_matches(basename, siblings, n=3, cutoff=0.4)
            msg = f"File not found: {file_path}"
            if suggestions:
                msg += f". Did you mean: {', '.join(os.path.join(parent, s) for s in suggestions)}?"
            return {"error": msg}

        with open(file_path, "rb") as f_bin:
            total_lines = sum(1 for _ in f_bin)

        if start_line is not None and end_line is not None and start_line > end_line:
            return {"error": "start_line cannot be greater than end_line"}

        if start_line is not None or end_line is not None:
            start = max(0, (start_line or 1) - 1)
            end = min(total_lines, (end_line or total_lines))
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                slice_lines = list(islice(f, start, end))
            return {
                "file_path": file_path,
                "lines_returned": f"{start + 1}-{end} of {total_lines}",
                "content": "".join(slice_lines),
            }

        max_lines_threshold = int(os.environ.get("SPOKE_MAX_FILE_LINES", "2000"))

        if total_lines > max_lines_threshold:
            outline = []
            max_ast_bytes = 500_000
            file_size = os.path.getsize(file_path)
            if file_path.endswith(".py") and file_size < max_ast_bytes:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        tree = ast.parse(f.read())
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            outline.append(f"Line {node.lineno}: {node.__class__.__name__.replace('Def', '')} {node.name}")
                except Exception:
                    pass

            if not outline:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if re.match(r'^(export )?(default )?(class|function|const|let|var) [a-zA-Z0-9_]+', line.lstrip()):
                            outline.append(f"Line {i+1}: {line.strip()[:100]}")

            return {
                "file_path": file_path,
                "error": f"File is too large ({total_lines} lines). Please use start_line and end_line.",
                "outline": outline if outline else "No clear outline extracted.",
            }

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"file_path": file_path, "content": content}
    except Exception as e:
        return {"error": str(e)}

def _validate_write_target(file_path: str) -> str | None:
    abs_path = os.path.abspath(file_path)
    home_dir = os.path.expanduser("~")
    for sensitive in [".ssh", ".gnupg", ".aws", "Library/Keychains"]:
        if abs_path.startswith(os.path.join(home_dir, sensitive)):
            return f"Write access denied to sensitive directory: {sensitive}"

    if not (
        abs_path.startswith(home_dir)
        or abs_path.startswith("/private/tmp")
        or abs_path.startswith("/tmp")
        or abs_path.startswith("/var/folders")
        or abs_path.startswith("/private/var/folders")
    ):
        return "Write access denied outside of user home or tmp directories."

    return None


def _contains_lazy_edit_placeholder(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "...",
        "rest of code",
        "existing code",
        "same as above",
        "same as before",
    )
    return any(pattern in lowered for pattern in patterns)


def _execute_write_file(arguments: dict) -> dict[str, Any]:
    raw_path = arguments.get("file_path")
    if not raw_path:
        return {"error": "file_path is required"}
    file_path = _resolve_tool_path(raw_path)
    content = arguments.get("content", "")
    if not file_path:
        return {"error": "file_path is required"}
    if content is None:
        content = ""
    try:
        access_error = _validate_write_target(file_path)
        if access_error:
            return {"error": access_error}

        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}


def _preferred_newline_style(text: str) -> str:
    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    cr_count = text.count("\r") - crlf_count
    if crlf_count >= lf_count and crlf_count >= cr_count and crlf_count > 0:
        return "\r\n"
    if lf_count > 0:
        return "\n"
    if cr_count > 0:
        return "\r"
    return "\n"


def _should_normalize_trailing_whitespace(file_path: str) -> bool:
    suffix = os.path.splitext(file_path)[1].lower()
    return suffix not in {".md", ".markdown"}


def _normalize_match_text_with_map(
    text: str, *, normalize_trailing_whitespace: bool = True
) -> tuple[str, list[int]]:
    normalized_parts: list[str] = []
    position_map = [0]
    raw_pos = 0

    for raw_line in text.splitlines(keepends=True):
        line_ending = ""
        body = raw_line
        if raw_line.endswith("\r\n"):
            line_ending = "\r\n"
            body = raw_line[:-2]
        elif raw_line.endswith("\n") or raw_line.endswith("\r"):
            line_ending = raw_line[-1]
            body = raw_line[:-1]

        trimmed_body = body.rstrip(" \t") if normalize_trailing_whitespace else body
        for idx, char in enumerate(trimmed_body, start=1):
            normalized_parts.append(char)
            position_map.append(raw_pos + idx)

        raw_pos += len(body)
        if line_ending:
            raw_pos += len(line_ending)
            normalized_parts.append("\n")
            position_map.append(raw_pos)

    if text and not text.endswith(("\n", "\r")):
        normalized_parts.append("\n")
        position_map.append(raw_pos)

    return "".join(normalized_parts), position_map


def _normalize_match_text(text: str, *, normalize_trailing_whitespace: bool = True) -> str:
    normalized, _ = _normalize_match_text_with_map(
        text,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
    )
    return normalized


def _normalize_text_for_comparison(
    text: str,
    *,
    normalize_line_endings: bool,
    normalize_trailing_whitespace: bool,
    normalize_missing_final_newline: bool,
) -> str:
    parts: list[str] = []
    for line in _split_lines_with_offsets(text):
        body = line["body"].rstrip(" \t") if normalize_trailing_whitespace else line["body"]
        parts.append(body)
        if line["has_line_ending"]:
            parts.append("\n" if normalize_line_endings else line["line_ending"])
    if text and not text.endswith(("\n", "\r")) and normalize_missing_final_newline:
        parts.append("\n")
    return "".join(parts)


def _normalization_was_needed(
    *,
    left: str,
    right: str,
    normalize_line_endings: bool,
    normalize_trailing_whitespace: bool,
    normalize_missing_final_newline: bool,
) -> bool:
    if left == right:
        return False
    return _normalize_text_for_comparison(
        left,
        normalize_line_endings=normalize_line_endings,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
        normalize_missing_final_newline=normalize_missing_final_newline,
    ) == _normalize_text_for_comparison(
        right,
        normalize_line_endings=normalize_line_endings,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
        normalize_missing_final_newline=normalize_missing_final_newline,
    )


def _apply_newline_style(text: str, newline_style: str) -> str:
    return text.replace("\n", newline_style)


def _canonicalize_final_newline(text: str, newline_style: str) -> str:
    if not text:
        return text
    return text.rstrip("\r\n") + newline_style


def _normalize_newlines_for_counting(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _line_number_for_offset(text: str, offset: int) -> int:
    prefix = _normalize_newlines_for_counting(text[:offset])
    return prefix.count("\n") + 1


def _logical_line_count(text: str) -> int:
    if text == "":
        return 1
    normalized = _normalize_newlines_for_counting(text)
    if normalized.endswith("\n"):
        return normalized.count("\n")
    return normalized.count("\n") + 1


def _edited_range_from_diff(old_text: str, new_text: str) -> dict[str, int]:
    old_lines = _normalize_newlines_for_counting(old_text).splitlines()
    new_lines = _normalize_newlines_for_counting(new_text).splitlines()

    prefix = 0
    while (
        prefix < len(old_lines)
        and prefix < len(new_lines)
        and old_lines[prefix] == new_lines[prefix]
    ):
        prefix += 1

    suffix = 0
    while (
        suffix < (len(old_lines) - prefix)
        and suffix < (len(new_lines) - prefix)
        and old_lines[len(old_lines) - 1 - suffix] == new_lines[len(new_lines) - 1 - suffix]
    ):
        suffix += 1

    start_line = prefix + 1
    end_line = max(start_line, len(new_lines) - suffix)
    return {"start_line": start_line, "end_line": end_line}


def _normalization_applied(
    *,
    matched_text: str,
    old_string: str,
    replacement_text: str,
    new_string: str,
    normalize_trailing_whitespace: bool,
    indentation_used: bool,
    final_newline_used: bool,
) -> list[str]:
    applied: list[str] = []
    line_endings_used = _normalization_was_needed(
        left=matched_text,
        right=old_string,
        normalize_line_endings=True,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
        normalize_missing_final_newline=False,
    ) and not _normalization_was_needed(
        left=matched_text,
        right=old_string,
        normalize_line_endings=False,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
        normalize_missing_final_newline=False,
    )
    line_endings_used = line_endings_used or (
        _normalization_was_needed(
            left=replacement_text,
            right=new_string,
            normalize_line_endings=True,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
            normalize_missing_final_newline=False,
        )
        and not _normalization_was_needed(
            left=replacement_text,
            right=new_string,
            normalize_line_endings=False,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
            normalize_missing_final_newline=False,
        )
    )
    if line_endings_used:
        applied.append("line_endings")
    trailing_used = normalize_trailing_whitespace and (
        _normalization_was_needed(
            left=matched_text,
            right=old_string,
            normalize_line_endings=True,
            normalize_trailing_whitespace=True,
            normalize_missing_final_newline=False,
        )
        and not _normalization_was_needed(
            left=matched_text,
            right=old_string,
            normalize_line_endings=True,
            normalize_trailing_whitespace=False,
            normalize_missing_final_newline=False,
        )
    )
    trailing_used = trailing_used or (
        normalize_trailing_whitespace
        and _normalization_was_needed(
            left=replacement_text,
            right=new_string,
            normalize_line_endings=True,
            normalize_trailing_whitespace=True,
            normalize_missing_final_newline=False,
        )
        and not _normalization_was_needed(
            left=replacement_text,
            right=new_string,
            normalize_line_endings=True,
            normalize_trailing_whitespace=False,
            normalize_missing_final_newline=False,
        )
    )
    if trailing_used:
        applied.append("trailing_whitespace")
    if indentation_used:
        applied.append("indentation")
    if final_newline_used:
        applied.append("final_newline")
    return applied


def _edit_result(
    *,
    status: str,
    file_path: str,
    match_count: int,
    failure_reason: str | None,
    normalization_applied: list[str],
    edited_range: dict[str, int] | None,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "applied": status == "success",
        "file": file_path,
        "file_path": file_path,
        "edited_range": edited_range,
        "normalization_applied": normalization_applied,
        "failure_reason": failure_reason,
        "match_count": match_count,
    }
    if error is not None:
        result["error"] = error
    return result


def _latest_edit_file_counters(path: Path) -> dict[str, int]:
    counters = {key: 0 for key in _EDIT_FILE_TELEMETRY_COUNTER_KEYS}
    try:
        if not path.is_file():
            return counters
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            entry = json.loads(line)
            saved = entry.get("counters")
            if not isinstance(saved, dict):
                continue
            for key in counters:
                value = saved.get(key)
                if isinstance(value, int):
                    counters[key] = value
            return counters
    except Exception:
        logger.debug("Failed to load edit_file telemetry counters", exc_info=True)
    return counters


def _append_edit_file_telemetry(result: dict[str, Any]) -> None:
    try:
        counters = _latest_edit_file_counters(_EDIT_FILE_TELEMETRY_PATH)
        outcome = (
            "success"
            if result.get("status") == "success"
            else result.get("failure_reason") or "error"
        )
        counters["total"] += 1
        if outcome in counters:
            counters[outcome] += 1
        normalization_applied = list(result.get("normalization_applied") or [])
        if normalization_applied:
            counters["normalization_assisted"] += 1
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": "edit_file",
            "outcome": outcome,
            "applied": bool(result.get("applied")),
            "file": result.get("file", ""),
            "failure_reason": result.get("failure_reason"),
            "match_count": result.get("match_count", 0),
            "normalization_applied": normalization_applied,
            "counters": counters,
        }
        _EDIT_FILE_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_EDIT_FILE_TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        logger.debug("Failed to append edit_file telemetry", exc_info=True)


def _split_lines_with_offsets(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    raw_pos = 0
    for raw_line in text.splitlines(keepends=True):
        line_ending = ""
        body = raw_line
        if raw_line.endswith("\r\n"):
            line_ending = "\r\n"
            body = raw_line[:-2]
        elif raw_line.endswith("\n") or raw_line.endswith("\r"):
            line_ending = raw_line[-1]
            body = raw_line[:-1]
        line_start = raw_pos
        raw_pos += len(raw_line)
        lines.append(
            {
                "body": body,
                "line_ending": line_ending,
                "start": line_start,
                "end": raw_pos,
                "has_line_ending": bool(line_ending),
            }
        )
    if not lines and text == "":
        return []
    return lines


def _leading_whitespace(text: str) -> str:
    prefix_len = len(text) - len(text.lstrip(" \t"))
    return text[:prefix_len]


def _indent_width(prefix: str) -> int:
    return len(prefix.expandtabs(4))


def _normalized_line_records(
    text: str, *, normalize_trailing_whitespace: bool
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in _split_lines_with_offsets(text):
        normalized_body = (
            line["body"].rstrip(" \t") if normalize_trailing_whitespace else line["body"]
        )
        leading = _leading_whitespace(normalized_body)
        content = normalized_body[len(leading) :]
        records.append(
            {
                **line,
                "normalized_body": normalized_body,
                "leading": leading,
                "content": content,
                "indent_width": _indent_width(leading),
            }
        )
    return records


def _relative_indent_levels(lines: list[dict[str, Any]]) -> list[int | None] | None:
    nonblank = [line for line in lines if line["content"] != ""]
    if not nonblank:
        return [None for _ in lines]

    base_width = nonblank[0]["indent_width"]
    relative_widths: list[int | None] = []
    positive_widths: list[int] = []
    for line in lines:
        if line["content"] == "":
            relative_widths.append(None)
            continue
        relative = line["indent_width"] - base_width
        if relative < 0:
            return None
        relative_widths.append(relative)
        if relative > 0:
            positive_widths.append(relative)

    indent_unit = math.gcd(*positive_widths) if positive_widths else 1
    levels: list[int | None] = []
    for relative in relative_widths:
        if relative is None:
            levels.append(None)
            continue
        levels.append(relative // indent_unit)
    return levels


def _line_block_matches(
    file_lines: list[dict[str, Any]],
    old_lines: list[dict[str, Any]],
) -> tuple[bool, list[int | None] | None]:
    if len(file_lines) != len(old_lines):
        return False, None
    if [line["content"] for line in file_lines] != [line["content"] for line in old_lines]:
        return False, None

    old_levels = _relative_indent_levels(old_lines)
    file_levels = _relative_indent_levels(file_lines)
    if old_levels is None or file_levels is None or old_levels != file_levels:
        return False, None
    return True, file_levels


def _indent_template(
    matched_lines: list[dict[str, Any]], levels: list[int | None]
) -> tuple[str, dict[int, str], str]:
    nonblank = [
        (line, level)
        for line, level in zip(matched_lines, levels, strict=False)
        if level is not None and line["content"] != ""
    ]
    if not nonblank:
        return "", {0: ""}, ""

    base_prefix = nonblank[0][0]["leading"]
    prefixes_by_level: dict[int, str] = {}
    for line, level in nonblank:
        prefixes_by_level.setdefault(level, line["leading"])

    unit_prefix = ""
    positive_levels = sorted(level for level in prefixes_by_level if level > 0)
    if positive_levels:
        level_one = positive_levels[0]
        level_one_prefix = prefixes_by_level[level_one]
        if level_one_prefix.startswith(base_prefix):
            unit_prefix = level_one_prefix[len(base_prefix) :] or unit_prefix
        elif base_prefix == "":
            unit_prefix = level_one_prefix

    return base_prefix, prefixes_by_level, unit_prefix


def _prefix_for_level(
    level: int | None,
    *,
    base_prefix: str,
    prefixes_by_level: dict[int, str],
    unit_prefix: str,
) -> str:
    if level is None:
        return ""
    if level in prefixes_by_level:
        return prefixes_by_level[level]
    if unit_prefix:
        return base_prefix + (unit_prefix * level)
    return prefixes_by_level.get(0, base_prefix)


def _render_indentation_aware_replacement(
    new_string: str,
    *,
    matched_lines: list[dict[str, Any]],
    matched_levels: list[int | None],
    normalize_trailing_whitespace: bool,
    newline_style: str,
) -> str:
    new_lines = _normalized_line_records(
        new_string,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
    )
    new_levels = _relative_indent_levels(new_lines)
    if new_levels is None:
        new_levels = [None for _ in new_lines]

    base_prefix, prefixes_by_level, unit_prefix = _indent_template(matched_lines, matched_levels)

    rendered_parts: list[str] = []
    for line, level in zip(new_lines, new_levels, strict=False):
        if line["content"] == "":
            body = ""
        else:
            prefix = _prefix_for_level(
                level,
                base_prefix=base_prefix,
                prefixes_by_level=prefixes_by_level,
                unit_prefix=unit_prefix,
            )
            body = prefix + line["content"]
        rendered_parts.append(body)
        if line["has_line_ending"]:
            rendered_parts.append("\n")

    return _apply_newline_style("".join(rendered_parts), newline_style)


def _find_indentation_aware_matches(
    raw_content: str,
    old_string: str,
    *,
    normalize_trailing_whitespace: bool,
) -> list[dict[str, Any]]:
    old_lines = _normalized_line_records(
        old_string,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
    )
    if len(old_lines) < 2:
        return []

    file_lines = _normalized_line_records(
        raw_content,
        normalize_trailing_whitespace=normalize_trailing_whitespace,
    )
    if len(file_lines) < len(old_lines):
        return []

    matches: list[dict[str, Any]] = []
    window = len(old_lines)
    for start_idx in range(len(file_lines) - window + 1):
        candidate = file_lines[start_idx : start_idx + window]
        matched, levels = _line_block_matches(candidate, old_lines)
        if not matched or levels is None:
            continue
        matches.append(
            {
                "raw_start": candidate[0]["start"],
                "raw_end": candidate[-1]["end"],
                "matched_lines": candidate,
                "levels": levels,
            }
        )
    return matches


def _execute_edit_file(arguments: dict) -> dict[str, Any]:
    def finish(result: dict[str, Any]) -> dict[str, Any]:
        _append_edit_file_telemetry(result)
        return result

    raw_path = arguments.get("file")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")

    if not raw_path or not isinstance(raw_path, str):
        return finish(_edit_result(
            status="error",
            file_path="",
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error="file is required",
        ))
    if not isinstance(old_string, str) or old_string == "":
        return finish(_edit_result(
            status="error",
            file_path=raw_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error="old_string is required",
        ))
    if not isinstance(new_string, str):
        return finish(_edit_result(
            status="error",
            file_path=raw_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error="new_string is required",
        ))
    if old_string == new_string:
        return finish(_edit_result(
            status="error",
            file_path=raw_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error="old_string and new_string must differ",
        ))
    if _contains_lazy_edit_placeholder(new_string):
        return finish(_edit_result(
            status="error",
            file_path=raw_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error="new_string contains lazy placeholder text",
        ))

    file_path = _resolve_tool_path(raw_path)
    access_error = _validate_write_target(file_path)
    if access_error:
        return finish(_edit_result(
            status="error",
            file_path=file_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error=access_error,
        ))

    if os.path.exists(file_path) and not os.path.isfile(file_path):
        return finish(_edit_result(
            status="error",
            file_path=file_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error="file is not a regular file",
        ))

    if not os.path.isfile(file_path):
        return finish(_edit_result(
            status="error",
            file_path=file_path,
            match_count=0,
            failure_reason="not_found",
            normalization_applied=[],
            edited_range=None,
        ))

    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        try:
            raw_content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return finish(_edit_result(
                status="error",
                file_path=file_path,
                match_count=0,
                failure_reason="malformed_request",
                normalization_applied=[],
                edited_range=None,
                error="file is not valid UTF-8 text",
            ))

        normalize_trailing_whitespace = _should_normalize_trailing_whitespace(file_path)
        newline_style = _preferred_newline_style(raw_content)
        indent_matches = _find_indentation_aware_matches(
            raw_content,
            old_string,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
        )
        if indent_matches:
            match_count = len(indent_matches)
            if match_count > 1:
                return finish(_edit_result(
                    status="error",
                    file_path=file_path,
                    match_count=match_count,
                    failure_reason="not_unique",
                    normalization_applied=[],
                    edited_range=None,
                ))

            match = indent_matches[0]
            replacement = _render_indentation_aware_replacement(
                new_string,
                matched_lines=match["matched_lines"],
                matched_levels=match["levels"],
                normalize_trailing_whitespace=normalize_trailing_whitespace,
                newline_style=newline_style,
            )
            updated_pre_canonicalization = (
                raw_content[: match["raw_start"]] + replacement + raw_content[match["raw_end"] :]
            )
            updated = _canonicalize_final_newline(updated_pre_canonicalization, newline_style)
            final_newline_used = (
                updated != updated_pre_canonicalization
                or (match["raw_end"] == len(raw_content) and not raw_content.endswith(("\n", "\r")))
            )
            applied = _normalization_applied(
                matched_text=raw_content[match["raw_start"] : match["raw_end"]],
                old_string=old_string,
                replacement_text=replacement,
                new_string=new_string,
                normalize_trailing_whitespace=normalize_trailing_whitespace,
                indentation_used=True,
                final_newline_used=final_newline_used,
            )
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(updated)
            return finish(_edit_result(
                status="success",
                file_path=file_path,
                match_count=1,
                failure_reason=None,
                normalization_applied=applied,
                edited_range=_edited_range_from_diff(raw_content, updated),
            ))

        normalized_content, position_map = _normalize_match_text_with_map(
            raw_content,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
        )
        normalized_old = _normalize_match_text(
            old_string,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
        )
        normalized_new = _normalize_match_text(
            new_string,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
        )

        match_count = normalized_content.count(normalized_old)
        if match_count == 0:
            return finish(_edit_result(
                status="error",
                file_path=file_path,
                match_count=0,
                failure_reason="not_found",
                normalization_applied=[],
                edited_range=None,
            ))
        if match_count > 1:
            return finish(_edit_result(
                status="error",
                file_path=file_path,
                match_count=match_count,
                failure_reason="not_unique",
                normalization_applied=[],
                edited_range=None,
            ))

        match_start = normalized_content.find(normalized_old)
        match_end = match_start + len(normalized_old)
        raw_start = position_map[match_start]
        raw_end = position_map[match_end]
        replacement = _apply_newline_style(normalized_new, newline_style)
        updated_pre_canonicalization = raw_content[:raw_start] + replacement + raw_content[raw_end:]
        updated = _canonicalize_final_newline(updated_pre_canonicalization, newline_style)
        final_newline_used = (
            updated != updated_pre_canonicalization
            or (raw_end == len(raw_content) and not raw_content.endswith(("\n", "\r")))
        )
        applied = _normalization_applied(
            matched_text=raw_content[raw_start:raw_end],
            old_string=old_string,
            replacement_text=replacement,
            new_string=new_string,
            normalize_trailing_whitespace=normalize_trailing_whitespace,
            indentation_used=False,
            final_newline_used=final_newline_used,
        )
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(updated)
        return finish(_edit_result(
            status="success",
            file_path=file_path,
            match_count=1,
            failure_reason=None,
            normalization_applied=applied,
            edited_range=_edited_range_from_diff(raw_content, updated),
        ))
    except Exception as e:
        return finish(_edit_result(
            status="error",
            file_path=file_path,
            match_count=0,
            failure_reason="malformed_request",
            normalization_applied=[],
            edited_range=None,
            error=str(e),
        ))

def _execute_search_file(arguments: dict) -> dict[str, Any]:
    import subprocess
    pattern = arguments.get("pattern", "")
    dir_path = _resolve_tool_path(arguments.get("dir_path", "."))
    if not pattern:
        return {"error": "pattern is required"}
    try:
        # Simple grep via subprocess with timeout and safe flags
        # -m 5 per file keeps total output bounded across large trees
        result = subprocess.run(
            ["grep", "-rnm", "5", "--", pattern, dir_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.rstrip("\n").split("\n") if result.stdout.strip() else []
        max_total = 30
        truncated = len(lines) > max_total
        lines = lines[:max_total]
        out: dict[str, Any] = {"matches": "\n".join(lines), "dir_path": dir_path, "pattern": pattern}
        if truncated:
            out["truncated"] = f"Showing first {max_total} matches"
        return out
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out after 10 seconds"}
    except Exception as e:
        return {"error": str(e)}


def _execute_find_file(arguments: dict) -> dict[str, Any]:
    from datetime import datetime, timezone
    from pathlib import Path

    pattern = arguments.get("pattern", "")
    dir_path = _resolve_tool_path(arguments.get("dir_path", "."))
    if not pattern:
        return {"error": "pattern is required"}
    try:
        root = Path(dir_path).expanduser()
        if not root.is_dir():
            return {"error": f"Not a valid directory: {dir_path}"}
        # Use ** prefix if the pattern doesn't already have path separators
        if "/" not in pattern and "**" not in pattern:
            glob_pattern = f"**/{pattern}"
        else:
            glob_pattern = pattern
        matches = []
        for p in root.glob(glob_pattern):
            try:
                st = p.stat()
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                matches.append({
                    "path": str(p),
                    "type": "dir" if p.is_dir() else "file",
                    "size": st.st_size if not p.is_dir() else None,
                    "modified": mtime,
                })
            except OSError:
                matches.append({"path": str(p), "type": "unknown"})
            if len(matches) >= 30:
                break
        result: dict[str, Any] = {"pattern": pattern, "dir_path": dir_path, "matches": matches}
        if len(matches) >= 30:
            result["truncated"] = "Showing first 30 matches"
        return result
    except Exception as e:
        return {"error": str(e)}


def _execute_epistaxis_ops(arguments: dict) -> str:
    """Execute a bounded Epistaxis operation plan and return JSON."""
    epistaxis_root = arguments.get("epistaxis_root", "")
    target_repo = arguments.get("target_repo", "")
    operations = arguments.get("operations", [])
    try:
        if not isinstance(operations, list):
            raise EpistaxisOperatorError("operations must be a list")
        operator = EpistaxisOperator(epistaxis_root, target_repo)
        return json.dumps(
            {
                "target_repo": target_repo,
                "operations": operator.execute_plan(operations),
            }
        )
    except EpistaxisOperatorError as exc:
        return json.dumps({"error": str(exc)})


def _execute_search_web(arguments: dict) -> str:
    """Execute the bounded web search surface and return JSON."""
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    try:
        normalized_max_results = int(max_results)
        operator = BraveSearchOperator()
        return json.dumps(
            operator.execute_search(query, max_results=normalized_max_results)
        )
    except (TypeError, ValueError, BraveSearchOperatorError) as exc:
        return json.dumps({"error": str(exc)})


def _execute_query_gmail(arguments: dict) -> str:
    """Execute the bounded Gmail query surface and return JSON."""
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    try:
        normalized_max_results = int(max_results)
        operator = GmailOperator()
        return json.dumps(
            operator.execute_query(query, max_results=normalized_max_results)
        )
    except (TypeError, ValueError, GmailOperatorError) as exc:
        return json.dumps({"error": str(exc)})


def _execute_run_terminal_command(
    arguments: dict,
    *,
    approval_granted: bool = False,
    session_approval_rules: list[dict[str, Any]] | None = None,
) -> str:
    """Execute the bounded terminal command surface and return JSON."""
    argv = arguments.get("argv")
    cwd = arguments.get("cwd")
    timeout_seconds = arguments.get("timeout_seconds", 10)
    try:
        operator = TerminalOperator(session_approval_rules=session_approval_rules)
        return json.dumps(
            operator.execute_command(
                argv,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                approval_granted=approval_granted,
            )
        )
    except TerminalOperatorError as exc:
        return json.dumps({"error": str(exc)})


def _execute_launch_subagent(arguments: dict, subagent_manager: Any | None = None) -> dict[str, Any]:
    if subagent_manager is None:
        return {"error": "Subagent manager unavailable"}
    kind = arguments.get("kind", "")
    prompt = arguments.get("prompt", "")
    try:
        return subagent_manager.launch(kind, prompt)
    except ValueError as exc:
        return {"error": str(exc)}


def _execute_list_subagents(subagent_manager: Any | None = None) -> dict[str, Any]:
    if subagent_manager is None:
        return {"error": "Subagent manager unavailable"}
    return {"jobs": subagent_manager.list_jobs()}


def _execute_get_subagent_result(
    arguments: dict,
    subagent_manager: Any | None = None,
) -> dict[str, Any]:
    if subagent_manager is None:
        return {"error": "Subagent manager unavailable"}
    subagent_id = arguments.get("subagent_id", "")
    if not subagent_id:
        return {"error": "subagent_id is required"}
    return subagent_manager.get_job(subagent_id)


def _execute_cancel_subagent(
    arguments: dict,
    subagent_manager: Any | None = None,
) -> dict[str, Any]:
    if subagent_manager is None:
        return {"error": "Subagent manager unavailable"}
    subagent_id = arguments.get("subagent_id", "")
    if not subagent_id:
        return {"error": "subagent_id is required"}
    return subagent_manager.cancel(subagent_id)



def execute_tool(
    name: str,
    arguments: dict,
    *,
    scene_cache: SceneCaptureCache | None = None,
    last_response: str | None = None,
    tts_client: Any | None = None,
    tray_writer: Callable[[str], Any] | None = None,
    tool_output_mode: str = "text",
    approval_granted: bool = False,
    session_approval_rules: list[dict[str, Any]] | None = None,
    subagent_manager: Any | None = None,
    history_compactor: Callable[[dict], str] | None = None,
) -> Any:
    """Execute a tool by name and return the result as a JSON string.

    Returns a JSON-encoded result for the model to consume.
    """
    if name == "capture_context":
        requested_include_image = arguments.get("include_image")
        wants_image = (
            tool_output_mode == "multimodal"
            if requested_include_image is None
            else bool(requested_include_image)
        )
        include_image = wants_image and tool_output_mode == "multimodal"
        skip_ocr = include_image and (
            os.environ.get("SPOKE_SKIP_OCR", "").lower() in ("1", "true", "yes")
        )
        capture = _execute_capture_context(
            arguments,
            scene_cache=scene_cache,
            skip_ocr=skip_ocr,
        )
        if capture is None:
            return json.dumps({"error": "Capture failed"})
        logger.info(
            "capture_context: tool_output_mode=%s requested_include_image=%r wants_image=%s include_image=%s skip_ocr=%s scene_ref=%s app=%r title=%r",
            tool_output_mode,
            requested_include_image,
            wants_image,
            include_image,
            skip_ocr,
            getattr(capture, "scene_ref", None),
            getattr(capture, "app_name", None),
            getattr(capture, "window_title", None),
        )
        if tool_output_mode == "multimodal":
            return _capture_context_multimodal_result(
                capture,
                include_image=include_image,
            )
        return json.dumps(
            _capture_context_result_dict(capture, include_image=include_image)
        )

    elif name == "read_aloud":
        return _execute_read_aloud(
            arguments,
            scene_cache=scene_cache,
            last_response=last_response,
            tts_client=tts_client,
        )

    elif name == "add_to_tray":
        return json.dumps(
            _execute_add_to_tray(
                arguments,
                tray_writer=tray_writer,
            )
        )


    elif name == "list_directory":
        return json.dumps(_execute_list_directory(arguments))
    elif name == "read_file":
        return json.dumps(_execute_read_file(arguments))
    elif name == "write_file":
        return json.dumps(_execute_write_file(arguments))
    elif name == "edit_file":
        return json.dumps(_execute_edit_file(arguments))
    elif name == "search_file":
        return json.dumps(_execute_search_file(arguments))
    elif name == "find_file":
        return json.dumps(_execute_find_file(arguments))
    elif name == "run_epistaxis_ops":
        return _execute_epistaxis_ops(arguments)
    elif name == "search_web":
        return _execute_search_web(arguments)
    elif name == "query_gmail":
        return _execute_query_gmail(arguments)
    elif name == "run_terminal_command":
        return _execute_run_terminal_command(
            arguments,
            approval_granted=approval_granted,
            session_approval_rules=session_approval_rules,
        )
    elif name == "launch_subagent":
        return json.dumps(
            _execute_launch_subagent(
                arguments,
                subagent_manager=subagent_manager,
            )
        )
    elif name == "list_subagents":
        return json.dumps(_execute_list_subagents(subagent_manager=subagent_manager))
    elif name == "get_subagent_result":
        return json.dumps(
            _execute_get_subagent_result(
                arguments,
                subagent_manager=subagent_manager,
            )
        )
    elif name == "cancel_subagent":
        return json.dumps(
            _execute_cancel_subagent(
                arguments,
                subagent_manager=subagent_manager,
            )
        )
    elif name == "compact_history":
        if history_compactor is None:
            return json.dumps({"error": "History compactor unavailable"})
        return history_compactor(arguments)
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})
