"""Bounded terminal command operator for the spoken command path.

The operator accepts parsed argv rather than raw shell text, classifies the
request against a small policy lattice, and only executes allowlisted commands.
Anything outside the allowlist is surfaced as approval-required or denied
without invoking a shell.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


_DEFAULT_CWD = Path.home() / "dev"
_MAX_TIMEOUT_SECONDS = 30
_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_OUTPUT_CHARS = 4000
_USER_LOCAL_BIN = str(Path.home() / ".local" / "bin")
_EXECUTABLE_SEARCH_PATH = os.pathsep.join(
    (
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        "/opt/homebrew/bin",
        "/usr/local/bin",
        _USER_LOCAL_BIN,
    )
)
_SHELL_CONTROL_TOKENS = frozenset({"|", "||", "&&", ";", "<", ">", ">>", "&", "2>", "2>>"})
_ALLOWED_CWD_ROOTS = (
    _DEFAULT_CWD,
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
    ("cat",),
    ("head",),
    ("tail",),
)
_APPROVAL_PREFIXES = (
    ("git",),
    ("find",),
    ("sed",),
    ("ps",),
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
    ("epistaxis",),
    ("epistaxis-create-worktree",),
    ("epistaxis-commit",),
)
_PATH_SCOPED_ALLOW_COMMANDS = frozenset({"cat", "head", "tail", "ls", "rg"})
_EPISTAXIS_EXECUTABLES = frozenset(
    {
        "epistaxis",
        "epistaxis-create-worktree",
        "epistaxis-commit",
    }
)
_BASE_EXECUTION_ENV = {
    "PATH": _EXECUTABLE_SEARCH_PATH,
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
}
_GIT_ENV_PASSTHROUGH = (
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "GH_CONFIG_DIR",
    "SSH_AUTH_SOCK",
)
_EPISTAXIS_ENV_PASSTHROUGH = (
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "GH_CONFIG_DIR",
    "SSH_AUTH_SOCK",
    "CODEX_THREAD_ID",
    "EPISTAXIS_LIVE_TOOLS_ROOT",
    "EPISTAXIS_BOOTSTRAP_SOURCE_REPO",
    "EPISTAXIS_BIN_DIR",
)
_RG_SHORT_VALUE_FLAGS = frozenset({"A", "B", "C", "e", "f", "g", "j", "m", "M", "t", "T"})
_EXECUTION_ENV_OVERRIDES = {
    "git": {
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "PAGER": "cat",
    },
    "rg": {
        "RIPGREP_CONFIG_PATH": os.devnull,
    },
}


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
        approval_granted: bool = False,
    ) -> dict[str, Any]:
        normalized_cwd = self._resolve_cwd(cwd)
        normalized_timeout = self._normalize_timeout(timeout_seconds)
        decision, reason = self._classify(argv, cwd=normalized_cwd)
        approval_state = "not_needed"
        if decision == "approval_required":
            approval_state = "granted" if approval_granted else "pending"
        elif decision == "deny":
            approval_state = "not_applicable"
        result: dict[str, Any] = {
            "decision": decision,
            "policy_decision": decision,
            "executed": False,
            "argv": argv,
            "cwd": normalized_cwd,
            "timed_out": False,
            "approval_state": approval_state,
        }
        if reason:
            result["reason"] = reason
            result["policy_reason"] = reason
        if decision == "approval_required" and not approval_granted:
            result["pending_approval"] = True
            result["approval_request"] = {
                "kind": "terminal_command",
                "argv": list(argv) if isinstance(argv, list) else argv,
                "cwd": normalized_cwd,
                "reason": reason,
                "message": self._format_approval_message(argv, normalized_cwd, reason),
            }
            return result
        if decision != "allow" and not (decision == "approval_required" and approval_granted):
            return result

        normalized_argv = self._normalized_argv(argv)
        executable_name = normalized_argv[0]
        resolved_executable = self._resolve_executable(executable_name)
        execution_argv = list(argv)
        execution_argv[0] = resolved_executable
        result["decision"] = "allow"
        result["executed"] = True
        if approval_granted:
            result["approved_by_user"] = True
            result.pop("reason", None)
        try:
            proc = subprocess.run(
                execution_argv,
                capture_output=True,
                cwd=normalized_cwd,
                env=self._execution_env(executable_name),
                text=False,
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
                    "output_complete": not (stdout_truncated or stderr_truncated),
                }
            )
            truncation_message = self._format_truncation_message(
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
            if truncation_message is not None:
                result["truncation_message"] = truncation_message
            return result
        except OSError as exc:
            raise TerminalOperatorError(
                f"failed to start command: {exc}"
            ) from exc

        stdout, stdout_truncated = self._truncate(proc.stdout)
        stderr, stderr_truncated = self._truncate(proc.stderr)
        result.update(
            {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "output_complete": not (stdout_truncated or stderr_truncated),
            }
        )
        truncation_message = self._format_truncation_message(
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        if truncation_message is not None:
            result["truncation_message"] = truncation_message
        return result

    def _format_approval_message(self, argv: Any, cwd: str, reason: str | None) -> str:
        command_text = self._display_command(argv)
        lines = [
            "Approval needed",
            "",
            command_text,
            f"cwd: {cwd}",
        ]
        if reason:
            lines.append(f"reason: {reason}")
        lines.extend(
            [
                "",
                "Enter to run  ·  Delete to cancel  ·  speak or type to revise",
            ]
        )
        return "\n".join(lines)

    def _display_command(self, argv: Any) -> str:
        normalized = self._normalized_argv(argv)
        return shlex.join(normalized)

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

    def _resolve_executable(self, executable_name: str) -> str:
        resolved = shutil.which(executable_name, path=_EXECUTABLE_SEARCH_PATH)
        if not resolved:
            raise TerminalOperatorError(
                f"allowed executable not found on bounded PATH: {executable_name}"
            )
        return resolved

    def _execution_env(self, executable_name: str) -> dict[str, str]:
        env = dict(_BASE_EXECUTION_ENV)
        if executable_name == "git":
            for key in _GIT_ENV_PASSTHROUGH:
                value = os.environ.get(key)
                if value:
                    env[key] = value
        if executable_name in _EPISTAXIS_EXECUTABLES:
            for key in _EPISTAXIS_ENV_PASSTHROUGH:
                value = os.environ.get(key)
                if value:
                    env[key] = value
        env.update(_EXECUTION_ENV_OVERRIDES.get(executable_name, {}))
        return env

    def _classify(self, argv: Any, *, cwd: str) -> tuple[str, str | None]:
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
        if "/" in argv[0]:
            return "deny", "command denied: pass a bare executable name, not an explicit path"
        normalized_argv = self._normalized_argv(argv)
        git_flag_reason = self._git_flag_approval_reason(normalized_argv)
        if git_flag_reason:
            return "approval_required", git_flag_reason

        rg_flag_reason = self._rg_flag_approval_reason(normalized_argv)
        if rg_flag_reason:
            return "approval_required", rg_flag_reason

        path_scope_reason = self._path_scope_reason(normalized_argv, cwd=cwd)
        if path_scope_reason:
            return "approval_required", path_scope_reason

        if self._matches_any_prefix(normalized_argv, _DENY_PREFIXES):
            return "deny", f"command denied by terminal policy: {' '.join(normalized_argv[:3])}"
        if self._matches_any_prefix(normalized_argv, _ALLOW_PREFIXES):
            return "allow", None
        if self._matches_any_prefix(normalized_argv, _APPROVAL_PREFIXES):
            return "approval_required", f"command requires approval: {' '.join(normalized_argv[:3])}"
        return "approval_required", f"command requires approval: {' '.join(normalized_argv[:3])}"

    def _truncate(self, text: Any) -> tuple[str, bool]:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        elif not isinstance(text, str):
            if text is None:
                return "", False
            text = str(text)
        if len(text) <= self._max_output_chars:
            return text, False
        return text[: self._max_output_chars], True

    def _format_truncation_message(
        self,
        *,
        stdout_truncated: bool,
        stderr_truncated: bool,
    ) -> str | None:
        if stdout_truncated and stderr_truncated:
            return (
                "tool output truncated before it reached the assistant: "
                f"stdout and stderr truncated to {self._max_output_chars} chars each"
            )
        if stdout_truncated:
            return (
                "tool output truncated before it reached the assistant: "
                f"stdout truncated to {self._max_output_chars} chars"
            )
        if stderr_truncated:
            return (
                "tool output truncated before it reached the assistant: "
                f"stderr truncated to {self._max_output_chars} chars"
            )
        return None

    @staticmethod
    def _matches_any_prefix(argv: list[str], prefixes: tuple[tuple[str, ...], ...]) -> bool:
        return any(TerminalOperator._starts_with(argv, prefix) for prefix in prefixes)

    @staticmethod
    def _normalized_argv(argv: list[str]) -> list[str]:
        normalized = list(argv)
        normalized[0] = Path(normalized[0]).name
        return normalized

    @staticmethod
    def _git_flag_approval_reason(argv: list[str]) -> str | None:
        if len(argv) < 2 or argv[0] != "git":
            return None
        subcommand = argv[1]
        if subcommand == "diff" and "--no-index" in argv[2:]:
            return "command requires approval: git diff --no-index"
        for index, token in enumerate(argv[2:], start=2):
            if token == "--ext-diff":
                return f"command requires approval: git {subcommand} --ext-diff"
            if token == "--textconv":
                return f"command requires approval: git {subcommand} --textconv"
            if subcommand == "blame" and token == "--contents":
                target = argv[index + 1] if index + 1 < len(argv) else "<missing>"
                return f"command requires approval: git blame --contents {target}"
            if subcommand == "blame" and token.startswith("--contents="):
                return f"command requires approval: git blame {token}"
            if token == "--output":
                target = argv[index + 1] if index + 1 < len(argv) else "<missing>"
                return f"command requires approval: git {subcommand} --output {target}"
            if token.startswith("--output="):
                return f"command requires approval: git {subcommand} {token}"
        return None

    @staticmethod
    def _rg_flag_approval_reason(argv: list[str]) -> str | None:
        if not argv or argv[0] != "rg":
            return None
        for index, token in enumerate(argv[1:], start=1):
            if token in {"-f", "--file"}:
                target = argv[index + 1] if index + 1 < len(argv) else "<missing>"
                return f"command requires approval: rg {token} {target}"
            if TerminalOperator._rg_attached_short_value(token, "f") is not None:
                return f"command requires approval: rg {token}"
            if token.startswith(("-f=", "--file=")):
                return f"command requires approval: rg {token}"
            if token == "--ignore-file":
                target = argv[index + 1] if index + 1 < len(argv) else "<missing>"
                return f"command requires approval: rg --ignore-file {target}"
            if token.startswith("--ignore-file="):
                return f"command requires approval: rg {token}"
            if token == "--pre":
                target = argv[index + 1] if index + 1 < len(argv) else "<missing>"
                return f"command requires approval: rg --pre {target}"
            if token.startswith("--pre="):
                return f"command requires approval: rg {token}"
            if token == "--pre-glob":
                target = argv[index + 1] if index + 1 < len(argv) else "<missing>"
                return f"command requires approval: rg --pre-glob {target}"
            if token.startswith("--pre-glob="):
                return f"command requires approval: rg {token}"
        return None

    @staticmethod
    def _path_scope_reason(argv: list[str], *, cwd: str) -> str | None:
        if not argv or argv[0] not in _PATH_SCOPED_ALLOW_COMMANDS:
            return None
        base = Path(cwd)
        for token in TerminalOperator._iter_path_operands(argv):
            resolved = Path(token).expanduser()
            if not resolved.is_absolute():
                resolved = (base / resolved).resolve()
            else:
                resolved = resolved.resolve()
            if not any(TerminalOperator._is_within(resolved, root) for root in _ALLOWED_CWD_ROOTS):
                return f"command requires approval: path escapes allowed local roots ({token})"
        return None

    @staticmethod
    def _iter_path_operands(argv: list[str]) -> list[str]:
        command = argv[0]
        if command in {"cat", "head", "tail", "ls"}:
            return TerminalOperator._path_operands_after_options(argv[1:])
        if command == "rg":
            path_tokens: list[str] = []
            expects_positional_pattern = True
            pattern_supplied_by_flag = False
            skip_next = False
            options_terminated = False
            for token in argv[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if options_terminated:
                    if expects_positional_pattern and not pattern_supplied_by_flag:
                        pattern_supplied_by_flag = True
                        continue
                    path_tokens.append(token)
                    continue
                if token == "--":
                    options_terminated = True
                    continue
                if token in {"-e", "--regexp", "-f", "--file"}:
                    pattern_supplied_by_flag = True
                    skip_next = True
                    continue
                if TerminalOperator._rg_attached_short_value(token, "e") is not None:
                    pattern_supplied_by_flag = True
                    continue
                if TerminalOperator._rg_attached_short_value(token, "f") is not None:
                    pattern_supplied_by_flag = True
                    continue
                if token in {"-g", "--glob", "--pre", "--pre-glob", "--ignore-file"}:
                    skip_next = True
                    continue
                if token.startswith(("-e=", "--regexp=", "-f=", "--file=")):
                    pattern_supplied_by_flag = True
                    continue
                if token.startswith(("-g=", "--glob=", "--pre=", "--pre-glob=", "--ignore-file=")):
                    continue
                if token in {"--files", "--type-list"}:
                    expects_positional_pattern = False
                    continue
                if token.startswith("-"):
                    continue
                if expects_positional_pattern and not pattern_supplied_by_flag:
                    pattern_supplied_by_flag = True
                    continue
                path_tokens.append(token)
            return path_tokens
        return []

    @staticmethod
    def _rg_attached_short_value(token: str, flag: str) -> str | None:
        if not token.startswith("-") or token.startswith("--") or len(token) <= 2:
            return None
        cluster = token[1:]
        for index, char in enumerate(cluster):
            if char == flag:
                value = cluster[index + 1 :]
                return value or None
            if char in _RG_SHORT_VALUE_FLAGS:
                return None
        return None

    @staticmethod
    def _is_path_operand(token: str) -> bool:
        if not token or token == "-":
            return False
        return not token.startswith("-")

    @staticmethod
    def _path_operands_after_options(tokens: list[str]) -> list[str]:
        operands: list[str] = []
        options_terminated = False
        for token in tokens:
            if options_terminated:
                if token != "-":
                    operands.append(token)
                continue
            if token == "--":
                options_terminated = True
                continue
            if TerminalOperator._is_path_operand(token):
                operands.append(token)
        return operands

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
