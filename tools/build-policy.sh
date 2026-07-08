#!/usr/bin/env bash
# build-policy.sh — concatenate shared core + tool stubs into deliverable policy files.
# Mirrors the operator-memory pattern. Run from repo root or any location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CORE="$REPO_ROOT/policy/shared/core.md"

if [[ ! -f "$CORE" ]]; then
  echo "error: shared core not found at $CORE" >&2
  exit 1
fi

build_one() {
  local tool="$1"
  local stub="$2"
  local output="$3"

  if [[ ! -f "$stub" ]]; then
    echo "error: stub not found at $stub" >&2
    return 1
  fi

  {
    echo "# ${tool} Instructions"
    echo ""
    cat "$CORE"
    printf '\n---\n\n'
    cat "$stub"
  } > "$output"

  echo "built: $output ($(wc -l < "$output") lines)"
}

build_one "Claude Code" \
  "$REPO_ROOT/policy/claude/stub.md" \
  "$REPO_ROOT/CLAUDE.md"

build_one "Codex" \
  "$REPO_ROOT/policy/codex/stub.md" \
  "$REPO_ROOT/AGENTS.md"

echo "done."
