"""Helpers for registry-backed launch-target switching."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_LAUNCH_TARGETS_PATH = Path.home() / ".config" / "spoke" / "launch_targets.json"
_DEFAULT_MAIN_TARGET_PATH = Path.home() / ".config" / "spoke" / "main-target"


def launch_targets_path() -> Path:
    override = os.environ.get("SPOKE_LAUNCH_TARGETS_PATH")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_LAUNCH_TARGETS_PATH


def main_target_path() -> Path:
    override = os.environ.get("SPOKE_MAIN_TARGET_PATH")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_MAIN_TARGET_PATH


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


def current_launch_target(
    current_checkout: Path,
    path: Path | None = None,
) -> dict | None:
    current_checkout = current_checkout.resolve()
    for target in iter_launch_targets(path):
        try:
            if target["path"].resolve() == current_checkout:
                return target
        except FileNotFoundError:
            continue
    return None


def current_launch_target_id(
    current_checkout: Path,
    path: Path | None = None,
) -> str | None:
    payload = load_launch_target_registry(path)
    target = current_launch_target(current_checkout, path)
    if target is not None:
        return target["id"]
    selected = payload.get("selected")
    return str(selected) if selected else None


def save_selected_launch_target(target_id: str, path: Path | None = None) -> bool:
    registry_path = path or launch_targets_path()
    payload = load_launch_target_registry(registry_path)
    target = resolve_launch_target(target_id, registry_path)
    if target is None:
        return False
    payload["selected"] = target_id
    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.rename(registry_path)
    except Exception:
        logger.warning(
            "Failed to save launch target registry to %s", registry_path, exc_info=True
        )
        return False
    try:
        target_file = main_target_path()
        target_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_target_file = target_file.with_suffix(".tmp")
        tmp_target_file.write_text(f"{target['path']}\n")
        tmp_target_file.rename(target_file)
        return True
    except Exception:
        logger.warning(
            "Failed to save main launcher target to %s", main_target_path(), exc_info=True
        )
        return False


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
            overrides[key] = value
    return overrides
