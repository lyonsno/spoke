"""Tool dispatch for screen context and bounded local operator actions.

Defines the tool schemas for capture_context and read_aloud, handles
accumulation of streamed tool call deltas, and executes tools locally.

See docs/screen-context-v1.md for the design.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from spoke.epistaxis_operator import EpistaxisOperator, EpistaxisOperatorError, tool_schema as epistaxis_tool_schema
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
        "description": "List contents of a directory. Returns files and subdirectories.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "Absolute or relative directory path"}
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
_RUN_EPISTAXIS_OPS_SCHEMA = epistaxis_tool_schema()


def get_tool_schemas() -> list[dict]:
    """Return the tool schemas for the assistant."""
    return [_CAPTURE_CONTEXT_SCHEMA, _READ_ALOUD_SCHEMA, _ADD_TO_TRAY_SCHEMA, _LIST_DIRECTORY_SCHEMA, _READ_FILE_SCHEMA, _WRITE_FILE_SCHEMA, _SEARCH_FILE_SCHEMA, _RUN_EPISTAXIS_OPS_SCHEMA]


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
        logger.info("read_aloud: tts_client present, model=%s, text=%d chars",
                     getattr(tts_client, "_model_id", "?"), len(text))
        try:
            logger.info("read_aloud: calling speak (blocking)")
            tts_client.speak(text)
            logger.info("read_aloud: speak finished")
            return f"Speaking: {text}"
        except Exception:
            logger.warning("TTS playback failed", exc_info=True)
            return "Error speaking text: TTS playback failed"
    else:
        logger.warning("read_aloud: no tts_client available")

    return f"Spoke: {text}"


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













def _execute_list_directory(arguments: dict) -> dict[str, Any]:
    import os
    dir_path = arguments.get("dir_path")
    if dir_path is None:
        dir_path = "."
    try:
        if not os.path.isdir(dir_path):
            return {"error": f"Not a valid directory: {dir_path}"}
        contents = os.listdir(dir_path)
        if len(contents) > 1000:
            contents = contents[:1000] + ["... (truncated, over 1000 items)"]
        return {"dir_path": dir_path, "contents": contents}
    except Exception as e:
        return {"error": str(e)}

def _execute_read_file(arguments: dict) -> dict[str, Any]:
    import os
    import ast
    import re
    from itertools import islice

    file_path = arguments.get("file_path")
    if not file_path:
        return {"error": "file_path is required"}

    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")

    if end_line == 0:
        return {"error": "end_line must be >= 1 if provided."}

    try:
        if not os.path.isfile(file_path):
            return {"error": f"File not found: {file_path}"}

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

def _execute_write_file(arguments: dict) -> dict[str, Any]:
    import os
    file_path = arguments.get("file_path", "")
    content = arguments.get("content", "")
    if not file_path:
        return {"error": "file_path is required"}
    if content is None:
        content = ""
    try:
        abs_path = os.path.abspath(file_path)
        home_dir = os.path.expanduser("~")
        for sensitive in [".ssh", ".gnupg", ".aws", "Library/Keychains"]:
            if abs_path.startswith(os.path.join(home_dir, sensitive)):
                return {"error": f"Write access denied to sensitive directory: {sensitive}"}

        # Guard against system roots
        if not abs_path.startswith(home_dir) and not abs_path.startswith("/private/tmp") and not abs_path.startswith("/tmp") and not abs_path.startswith("/var/folders") and not abs_path.startswith("/private/var/folders"):
            return {"error": "Write access denied outside of user home or tmp directories."}

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

def _execute_search_file(arguments: dict) -> dict[str, Any]:
    import subprocess
    pattern = arguments.get("pattern", "")
    dir_path = arguments.get("dir_path", ".")
    if not pattern:
        return {"error": "pattern is required"}
    try:
        # Simple grep via subprocess with timeout and safe flags
        result = subprocess.run(
            ["grep", "-rnm", "100", "--", pattern, dir_path], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return {"matches": result.stdout, "dir_path": dir_path, "pattern": pattern}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out after 10 seconds"}
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


def execute_tool(
    name: str,
    arguments: dict,
    *,
    scene_cache: SceneCaptureCache | None = None,
    last_response: str | None = None,
    tts_client: Any | None = None,
    tray_writer: Callable[[str], Any] | None = None,
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
    elif name == "search_file":
        return json.dumps(_execute_search_file(arguments))
    elif name == "run_epistaxis_ops":
        return _execute_epistaxis_ops(arguments)

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})
