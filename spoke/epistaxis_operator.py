"""Experimental bounded executor for Epistaxis-only operations.

This is intentionally narrower than a shell broker. The caller supplies a JSON
plan containing a small set of allowed operations, and this module enforces the
git/worktree guardrails before mutating the Epistaxis repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


_INERT_EPISTAXIS_MAIN = Path.home() / "dev" / "epistaxis"
_EPISTAXIS_POLICY_PATH = Path("policy/codex/agents.md")
_REVIEW_NOTES_HEADER = "## Review Notes"
_ALLOWED_OPS = frozenset(
    {
        "read_repo_note",
        "list_reviews",
        "write_review_ticket",
        "append_review_pointer",
        "stage_review_artifacts",
        "git_status",
        "git_commit",
        "git_push_current_branch",
    }
)


class EpistaxisOperatorError(RuntimeError):
    """Raised when a requested operation violates the bounded contract."""


def tool_schema() -> dict[str, Any]:
    """Return a tool-call-friendly schema for the bounded operation surface."""

    operation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "op": {"type": "string", "enum": sorted(_ALLOWED_OPS)},
            "ticket_name": {"type": "string"},
            "content": {"type": "string"},
            "entry_heading": {"type": "string"},
            "bullets": {"type": "array", "items": {"type": "string"}},
            "ticket_names": {"type": "array", "items": {"type": "string"}},
            "commit_message": {"type": "string"},
        },
        "required": ["op"],
    }
    return {
        "type": "function",
        "function": {
            "name": "run_epistaxis_ops",
            "description": (
                "Run a bounded set of Epistaxis-only operations from a dedicated "
                "Epistaxis worktree. Not for authoritative current-state reads. "
                "No arbitrary shell or non-Epistaxis writes."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "epistaxis_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to a dedicated Epistaxis worktree. "
                            "Do not use the inert ~/dev/epistaxis main checkout."
                        ),
                    },
                    "target_repo": {"type": "string"},
                    "operations": {"type": "array", "items": operation},
                },
                "required": ["epistaxis_root", "target_repo", "operations"],
            },
        },
    }


class EpistaxisOperator:
    """Execute a tiny, deterministic operation set inside the Epistaxis repo."""

    def __init__(self, epistaxis_root: str | Path, target_repo: str):
        self._root = Path(epistaxis_root).expanduser().resolve()
        self._target_repo = target_repo.strip()
        if not self._target_repo:
            raise EpistaxisOperatorError("target_repo must not be empty")
        self._repo_note = self._root / f"{self._target_repo}_epistaxis.md"
        self._reviews_dir = self._root / "reviews"

    def execute_plan(self, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_worktree_guardrails()
        results: list[dict[str, Any]] = []
        for op in operations:
            name = op.get("op")
            if name not in _ALLOWED_OPS:
                raise EpistaxisOperatorError(f"unsupported op: {name!r}")
            handler = getattr(self, f"_op_{name}")
            results.append(handler(op))
        return results

    def _ensure_worktree_guardrails(self) -> None:
        if self._root == _INERT_EPISTAXIS_MAIN.resolve():
            raise EpistaxisOperatorError(
                f"refusing to mutate inert Epistaxis main checkout: {self._root}"
            )
        if not (self._root / ".git").exists():
            raise EpistaxisOperatorError(f"not an Epistaxis git checkout: {self._root}")
        if not (self._root / _EPISTAXIS_POLICY_PATH).exists():
            raise EpistaxisOperatorError(
                f"missing {str(_EPISTAXIS_POLICY_PATH)!r}; refusing to run outside an Epistaxis checkout"
            )
        branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD").strip()
        if branch == "main":
            raise EpistaxisOperatorError(
                "refusing to run from branch 'main'; use a dedicated Epistaxis worktree branch"
            )

    def _git_output(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self._root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout

    def _git_run(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(self._root), *args],
            capture_output=True,
            text=True,
            check=True,
        )

    def _ticket_path(self, ticket_name: str) -> Path:
        if not ticket_name.endswith(".md"):
            raise EpistaxisOperatorError("ticket_name must end with .md")
        if "/" in ticket_name or ticket_name.startswith("."):
            raise EpistaxisOperatorError("ticket_name must be a bare filename")
        if not ticket_name.startswith(f"{self._target_repo}_"):
            raise EpistaxisOperatorError(
                f"ticket_name must start with '{self._target_repo}_'"
            )
        return self._reviews_dir / ticket_name

    def _read_repo_note(self) -> str:
        if not self._repo_note.exists():
            raise EpistaxisOperatorError(f"repo note does not exist: {self._repo_note.name}")
        return self._repo_note.read_text(encoding="utf-8")

    def _write_repo_note(self, text: str) -> None:
        self._repo_note.write_text(text, encoding="utf-8")

    def _insert_review_pointer(self, text: str, block: str) -> str:
        header_idx = text.find(_REVIEW_NOTES_HEADER)
        if header_idx == -1:
            raise EpistaxisOperatorError(
                f"repo note is missing the '{_REVIEW_NOTES_HEADER}' section"
            )
        after_header = text.find("\n", header_idx)
        if after_header == -1:
            after_header = len(text)
        next_section = text.find("\n## ", after_header + 1)
        if next_section == -1:
            before = text.rstrip()
            return f"{before}\n\n{block.rstrip()}\n"
        section_prefix = text[:next_section].rstrip()
        section_suffix = text[next_section:]
        return f"{section_prefix}\n\n{block.rstrip()}\n{section_suffix}"

    def _op_read_repo_note(self, op: dict[str, Any]) -> dict[str, Any]:
        return {
            "op": "read_repo_note",
            "path": str(self._repo_note),
            "content": self._read_repo_note(),
        }

    def _op_list_reviews(self, op: dict[str, Any]) -> dict[str, Any]:
        reviews = sorted(
            path.name
            for path in self._reviews_dir.glob(f"{self._target_repo}_*.md")
            if path.is_file()
        )
        return {"op": "list_reviews", "reviews": reviews}

    def _op_write_review_ticket(self, op: dict[str, Any]) -> dict[str, Any]:
        ticket_name = str(op.get("ticket_name", "")).strip()
        content = str(op.get("content", ""))
        if not content.strip():
            raise EpistaxisOperatorError("write_review_ticket requires non-empty content")
        ticket_path = self._ticket_path(ticket_name)
        ticket_path.parent.mkdir(parents=True, exist_ok=True)
        ticket_path.write_text(content, encoding="utf-8")
        return {"op": "write_review_ticket", "path": str(ticket_path)}

    def _op_append_review_pointer(self, op: dict[str, Any]) -> dict[str, Any]:
        heading = str(op.get("entry_heading", "")).strip()
        bullets = op.get("bullets")
        if not heading:
            raise EpistaxisOperatorError("append_review_pointer requires entry_heading")
        if not isinstance(bullets, list) or not bullets or not all(
            isinstance(item, str) and item.strip() for item in bullets
        ):
            raise EpistaxisOperatorError(
                "append_review_pointer requires a non-empty bullets list of strings"
            )
        block = "\n".join([f"### {heading}", *[f"- {item.strip()}" for item in bullets]])
        updated = self._insert_review_pointer(self._read_repo_note(), block)
        self._write_repo_note(updated)
        return {
            "op": "append_review_pointer",
            "path": str(self._repo_note),
            "heading": heading,
        }

    def _op_stage_review_artifacts(self, op: dict[str, Any]) -> dict[str, Any]:
        ticket_names = op.get("ticket_names", [])
        if not isinstance(ticket_names, list) or not all(isinstance(item, str) for item in ticket_names):
            raise EpistaxisOperatorError("stage_review_artifacts requires ticket_names as a list of strings")
        rel_paths = [self._repo_note.name]
        for ticket_name in ticket_names:
            rel_paths.append(str(self._ticket_path(ticket_name).relative_to(self._root)))
        self._git_run("add", *rel_paths)
        return {"op": "stage_review_artifacts", "paths": rel_paths}

    def _op_git_status(self, op: dict[str, Any]) -> dict[str, Any]:
        return {"op": "git_status", "status": self._git_output("status", "--short")}

    def _op_git_commit(self, op: dict[str, Any]) -> dict[str, Any]:
        message = str(op.get("commit_message", "")).strip()
        if not message:
            raise EpistaxisOperatorError("git_commit requires commit_message")
        self._git_run("commit", "-m", message)
        sha = self._git_output("rev-parse", "--short", "HEAD").strip()
        return {"op": "git_commit", "commit": sha, "message": message}

    def _op_git_push_current_branch(self, op: dict[str, Any]) -> dict[str, Any]:
        branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD").strip()
        self._git_run("push", "-u", "origin", branch)
        return {"op": "git_push_current_branch", "branch": branch}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Epistaxis operation plan")
    parser.add_argument("--epistaxis-root", default=".", help="Epistaxis worktree root")
    parser.add_argument("--target-repo", required=True, help="Target repo slug, e.g. spoke")
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to a JSON file containing {'operations': [...]}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    operations = plan.get("operations")
    if not isinstance(operations, list):
        raise EpistaxisOperatorError("plan JSON must contain an 'operations' list")
    operator = EpistaxisOperator(args.epistaxis_root, args.target_repo)
    result = {
        "target_repo": args.target_repo,
        "operations": operator.execute_plan(operations),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
