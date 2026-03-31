"""Helpers for registry-backed launch-target switching."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)

_DEFAULT_LAUNCH_TARGETS_PATH = Path.home() / ".config" / "spoke" / "launch_targets.json"
_DEFAULT_SHARED_LAUNCH_ENV_PATH = Path.home() / ".config" / "spoke" / "launch-env.sh"
_ENV_EXPR_RE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<op>[-+])(?P<arg>[^}]*))?\}"
    r"|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)"
)


def launch_targets_path() -> Path:
    override = os.environ.get("SPOKE_LAUNCH_TARGETS_PATH")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_LAUNCH_TARGETS_PATH


def shared_launch_env_path() -> Path:
    override = os.environ.get("SPOKE_SHARED_LAUNCH_ENV_PATH")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_SHARED_LAUNCH_ENV_PATH


def load_launch_target_registry(path: Path | None = None) -> dict:
    registry_path = path or launch_targets_path()
    try:
        payload = json.loads(registry_path.read_text())
    except FileNotFoundError:
        return {"selected": None, "targets": []}
    except Exception:
        logger.warning(
            "Failed to read launch target registry from %s", registry_path, exc_info=True
        )
        return {"selected": None, "targets": []}
    if not isinstance(payload, dict):
        return {"selected": None, "targets": []}
    targets = payload.get("targets")
    if not isinstance(targets, list):
        targets = []
    return {
        "selected": payload.get("selected"),
        "targets": targets,
    }


def iter_launch_targets(path: Path | None = None) -> list[dict]:
    payload = load_launch_target_registry(path)
    resolved_targets: list[dict] = []
    for raw_target in payload.get("targets", []):
        if not isinstance(raw_target, dict):
            continue
        target_id = str(raw_target.get("id", "")).strip()
        target_path_raw = str(raw_target.get("path", "")).strip()
        if not target_id or not target_path_raw:
            continue
        target_path = Path(target_path_raw).expanduser()
        resolved_targets.append(
            {
                "id": target_id,
                "label": str(raw_target.get("label") or target_id),
                "path": target_path,
                "enabled": target_path.is_dir(),
            }
        )
    return resolved_targets


def resolve_launch_target(target_id: str, path: Path | None = None) -> dict | None:
    for target in iter_launch_targets(path):
        if target["id"] == target_id:
            return target
    return None


def current_launch_target_id(
    current_checkout: Path,
    path: Path | None = None,
) -> str | None:
    current_checkout = current_checkout.resolve()
    payload = load_launch_target_registry(path)
    for target in iter_launch_targets(path):
        try:
            if target["path"].resolve() == current_checkout:
                return target["id"]
        except FileNotFoundError:
            continue
    selected = payload.get("selected")
    return str(selected) if selected else None


def save_selected_launch_target(target_id: str, path: Path | None = None) -> bool:
    registry_path = path or launch_targets_path()
    payload = load_launch_target_registry(registry_path)
    if resolve_launch_target(target_id, registry_path) is None:
        return False
    payload["selected"] = target_id
    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.rename(registry_path)
        return True
    except Exception:
        logger.warning(
            "Failed to save launch target registry to %s", registry_path, exc_info=True
        )
        return False


def _expand_env_value(value: str, base_env: dict[str, str] | None = None) -> str:
    env = dict(base_env or os.environ)

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group("braced") or match.group("plain")
        current = env.get(var_name, "")
        op = match.group("op")
        arg = match.group("arg") or ""
        if op == "-":
            return current or _ENV_EXPR_RE.sub(_replace, arg)
        if op == "+":
            return _ENV_EXPR_RE.sub(_replace, arg) if current else ""
        return current

    expanded = value
    for _ in range(8):
        updated = _ENV_EXPR_RE.sub(_replace, expanded)
        if updated == expanded:
            break
        expanded = updated
    return expanded


def parse_env_overrides(env_file: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    try:
        lines = env_file.read_text().splitlines()
    except FileNotFoundError:
        return overrides
    except Exception:
        logger.warning("Failed to read env overrides from %s", env_file, exc_info=True)
        return overrides

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            merged_env = os.environ.copy()
            merged_env.update(overrides)
            overrides[key] = _expand_env_value(value, merged_env)
    return overrides


def load_launch_env_overrides(repo_root: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for env_file in (shared_launch_env_path(), repo_root / ".spoke-smoke-env"):
        overrides.update(parse_env_overrides(env_file))
    return overrides
