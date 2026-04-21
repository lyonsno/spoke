"""Bounded terminal command operator for the spoken command path.

The operator accepts parsed argv rather than raw shell text, classifies the
request against a small policy lattice, and only executes allowlisted commands.
Anything outside the allowlist is surfaced as approval-required or denied
without invoking a shell.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


_DEFAULT_CWD = Path.home() / "dev"
_MAX_TIMEOUT_SECONDS = 30
_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_OUTPUT_CHARS = 4000
_SHELL_CONTROL_TOKENS = frozenset({"|", "||", "&&", ";", "<", ">", ">>", "&", "2>", "2>>"})
_ALLOWED_CWD_ROOTS = (
    Path.home(),
    Path("/private/tmp"),
    Path("/tmp"),
    Path("/var/folders"),
    Path("/private/var/folders"),
)
_DENY_PREFIXES = (
    ("rm",),
    ("sudo",),
    ("su",),
    ("doas",),
    ("shutdown",),
    ("reboot",),
    ("launchctl",),
    ("killall",),
    ("pkill",),
    ("git", "reset", "--hard"),
    ("git", "clean"),
    ("git", "push", "--force"),
    ("git", "push", "-f"),
)
_ALLOW_PREFIXES = (
    ("git", "status"),
    ("git", "log"),
    ("git", "diff"),
    ("git", "show"),
    ("git", "branch", "--list"),
    ("git", "branch", "--show-current"),
    ("git", "blame"),
    ("git", "rev-parse"),
    ("git", "cherry"),
    ("rg",),
    ("ls",),
    ("pwd",),
    ("date",),
    ("find",),
    ("cat",),
    ("head",),
    ("tail",),
    ("sed", "-n"),
    ("ps",),
)
_APPROVAL_PREFIXES = (
    ("git",),
    ("pytest",),
    ("uv",),
    ("python",),
    ("python3",),
    ("bash",),
    ("sh",),
    ("zsh",),
    ("make",),
    ("npm",),
    ("pnpm",),
    ("cargo",),
    ("curl",),
    ("wget",),
    ("kill",),
    ("open",),
    ("osascript",),
)


class TerminalOperatorError(RuntimeError):
    """Raised when the bounded terminal contract cannot be satisfied."""


def tool_schema() -> dict[str, Any]:
    """Return the OpenAI tool schema for the bounded terminal surface."""

    return {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": (
                "Run a bounded local terminal command using parsed argv, not raw "
                "shell text. Pass each token as a separate argv entry. Do not "
                "use shell syntax like pipes, redirects, chaining, or shell "
                "interpolation. The tool applies allow/deny/approval policy and "
                "returns compact stdout/stderr, exit code, and timeout/truncation state."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "Command argv as separate tokens, e.g. "
                            "['git', 'status', '--short']."
                        ),
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Optional working directory. Relative paths resolve "
                            "against ~/dev. Defaults to ~/dev."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_TIMEOUT_SECONDS,
                        "description": (
                            "Optional timeout in seconds. Defaults to 10 and is "
                            f"capped at {_MAX_TIMEOUT_SECONDS}."
                        ),
                    },
                },
                "required": ["argv"],
            },
        },
    }


class TerminalOperator:
    """Execute bounded terminal commands without exposing shell syntax."""

    def __init__(self, *, max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS):
        self._max_output_chars = max_output_chars

    def execute_command(
        self,
        argv: Any,
        *,
        cwd: str | None = None,
        timeout_seconds: Any = _DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        normalized_cwd = self._resolve_cwd(cwd)
        normalized_timeout = self._normalize_timeout(timeout_seconds)
        decision, reason = self._classify(argv)
        result: dict[str, Any] = {
            "decision": decision,
            "executed": False,
            "argv": argv,
            "cwd": normalized_cwd,
            "timed_out": False,
        }
        if reason:
            result["reason"] = reason
        if decision != "allow":
            return result

        result["executed"] = True
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                cwd=normalized_cwd,
                text=True,
                timeout=normalized_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_truncated = self._truncate(exc.stdout)
            stderr, stderr_truncated = self._truncate(exc.stderr)
            result.update(
                {
                    "exit_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                }
            )
            return result

        stdout, stdout_truncated = self._truncate(proc.stdout)
        stderr, stderr_truncated = self._truncate(proc.stderr)
        result.update(
            {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        )
        return result

    def _resolve_cwd(self, cwd: str | None) -> str:
        raw = (cwd or "").strip()
        path = _DEFAULT_CWD if not raw else Path(raw).expanduser()
        if not path.is_absolute():
            path = (_DEFAULT_CWD / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise TerminalOperatorError(f"cwd does not exist: {path}")
        if not path.is_dir():
            raise TerminalOperatorError(f"cwd is not a directory: {path}")
        if not any(self._is_within(path, root) for root in _ALLOWED_CWD_ROOTS):
            raise TerminalOperatorError(f"cwd is outside allowed local roots: {path}")
        return str(path)

    def _normalize_timeout(self, timeout_seconds: Any) -> int:
        if timeout_seconds is None:
            return _DEFAULT_TIMEOUT_SECONDS
        try:
            timeout = int(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise TerminalOperatorError("timeout_seconds must be an integer") from exc
        if not 1 <= timeout <= _MAX_TIMEOUT_SECONDS:
            raise TerminalOperatorError(
                f"timeout_seconds must be between 1 and {_MAX_TIMEOUT_SECONDS}"
            )
        return timeout

    def _classify(self, argv: Any) -> tuple[str, str | None]:
        if not isinstance(argv, list) or not argv:
            return "deny", "command denied: argv must be a non-empty list of strings"
        if not all(isinstance(token, str) for token in argv):
            return "deny", "command denied: argv entries must all be strings"
        if any(not token for token in argv):
            return "deny", "command denied: argv entries must not be empty"
        if any(token in _SHELL_CONTROL_TOKENS for token in argv):
            return "deny", "command denied: shell syntax tokens are not supported; pass plain argv only"
        if any("\n" in token or "\x00" in token for token in argv):
            return "deny", "command denied: argv tokens must not contain newlines or NUL bytes"

        if self._matches_any_prefix(argv, _DENY_PREFIXES):
            return "deny", f"command denied by terminal policy: {' '.join(argv[:3])}"
        if self._matches_any_prefix(argv, _ALLOW_PREFIXES):
            return "allow", None
        if self._matches_any_prefix(argv, _APPROVAL_PREFIXES):
            return "approval_required", f"command requires approval: {' '.join(argv[:3])}"
        return "approval_required", f"command requires approval: {' '.join(argv[:3])}"

    def _truncate(self, text: Any) -> tuple[str, bool]:
        if not isinstance(text, str):
            if text is None:
                return "", False
            text = str(text)
        if len(text) <= self._max_output_chars:
            return text, False
        return text[: self._max_output_chars], True

    @staticmethod
    def _matches_any_prefix(argv: list[str], prefixes: tuple[tuple[str, ...], ...]) -> bool:
        return any(TerminalOperator._starts_with(argv, prefix) for prefix in prefixes)

    @staticmethod
    def _starts_with(argv: list[str], prefix: tuple[str, ...]) -> bool:
        return len(argv) >= len(prefix) and tuple(argv[: len(prefix)]) == prefix

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
