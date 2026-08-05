"""Pure model and Epistaxis client for the live Diaulos switcher."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import unquote, urlparse


class DiaulosInventoryError(RuntimeError):
    pass


class DiaulosActivationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiaulosCandidate:
    handle: str
    diaulos_id: str
    aliases: tuple[str, ...]
    pane_id: int
    tab_id: int
    window_id: int
    title: str
    cwd: str
    thread_id: str
    match_basis: tuple[str, ...]
    observed_at: str
    discovery_authority: str

    @property
    def searchable_text(self) -> str:
        return " ".join((self.handle, *self.aliases, self.title, self.cwd))


def parse_live_inventory(payload: Any) -> list[DiaulosCandidate]:
    if not isinstance(payload, dict):
        raise DiaulosInventoryError("live Diaulos inventory is not an object")
    if payload.get("status") != "complete":
        raise DiaulosInventoryError(
            f"live Diaulos inventory is not complete: {payload.get('status') or 'missing'}"
        )
    observed_at = str(payload.get("observed_at") or "").strip()
    if not observed_at:
        raise DiaulosInventoryError("live Diaulos inventory has no observation timestamp")
    authority = str(payload.get("discovery_authority") or "").strip()
    if authority != "complete-live-pane-enumeration":
        raise DiaulosInventoryError(
            f"live Diaulos inventory has non-authoritative discovery route: {authority or 'missing'}"
        )
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise DiaulosInventoryError("live Diaulos inventory entries are missing")

    candidates: list[DiaulosCandidate] = []
    identities: set[tuple[str, int]] = set()
    panes_by_handle: dict[str, set[int]] = {}
    handles_by_pane: dict[int, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DiaulosInventoryError(f"live Diaulos row {index} is not an object")
        handle = str(row.get("handle") or "").strip()
        pane_id = _required_int(row.get("pane_id"), f"row {index} pane_id")
        if not handle:
            raise DiaulosInventoryError(f"live Diaulos row {index} has no handle")
        identity = (handle, pane_id)
        handle_panes = panes_by_handle.setdefault(handle, set())
        if handle_panes and pane_id not in handle_panes:
            raise DiaulosInventoryError(
                f"live Diaulos inventory gives {handle} authority over multiple panes"
            )
        handle_panes.add(pane_id)
        pane_handles = handles_by_pane.setdefault(pane_id, set())
        if pane_handles and handle not in pane_handles:
            raise DiaulosInventoryError(
                f"live Diaulos inventory gives pane {pane_id} authority over multiple handles"
            )
        pane_handles.add(handle)
        if identity in identities:
            raise DiaulosInventoryError(
                f"live Diaulos inventory repeats {handle} on pane {pane_id}"
            )
        identities.add(identity)
        aliases_raw = row.get("aliases") or []
        if not isinstance(aliases_raw, list):
            raise DiaulosInventoryError(f"live Diaulos row {index} aliases are not a list")
        tab_id = _required_int(row.get("tab_id"), f"row {index} tab_id")
        window_id = _required_int(row.get("window_id"), f"row {index} window_id")
        cwd = str(row.get("cwd") or "").strip()
        if not cwd:
            raise DiaulosInventoryError(f"live Diaulos row {index} cwd is missing")
        candidates.append(
            DiaulosCandidate(
                handle=handle,
                diaulos_id=str(row.get("diaulos_id") or "").strip(),
                aliases=tuple(
                    str(value).strip()
                    for value in aliases_raw
                    if str(value).strip()
                ),
                pane_id=pane_id,
                tab_id=tab_id,
                window_id=window_id,
                title=str(row.get("title") or "").strip(),
                cwd=cwd,
                thread_id=str(row.get("thread_id") or "").strip(),
                match_basis=tuple(
                    str(value).strip()
                    for value in row.get("match_basis") or []
                    if str(value).strip()
                ),
                observed_at=observed_at,
                discovery_authority=authority,
            )
        )
    return candidates


class DiaulosSwitcherModel:
    def __init__(self, candidates: Sequence[DiaulosCandidate]) -> None:
        self.all_candidates = list(candidates)
        self.filtered = list(candidates)
        self.query = ""
        self.selected_index = 0

    @property
    def selected(self) -> DiaulosCandidate | None:
        if not self.filtered:
            return None
        return self.filtered[self.selected_index]

    def set_query(self, query: str) -> None:
        selected_identity = (
            (self.selected.handle, self.selected.pane_id)
            if self.selected is not None
            else None
        )
        self.query = query
        terms = _search_terms(query)
        self.filtered = [
            candidate
            for candidate in self.all_candidates
            if _candidate_matches(candidate, terms)
        ]
        self.selected_index = 0
        if selected_identity is not None:
            for index, candidate in enumerate(self.filtered):
                if (candidate.handle, candidate.pane_id) == selected_identity:
                    self.selected_index = index
                    break

    def move(self, delta: int) -> None:
        if not self.filtered:
            self.selected_index = 0
            return
        self.selected_index = min(
            len(self.filtered) - 1,
            max(0, self.selected_index + delta),
        )


class EpistaxisDiaulosClient:
    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout_seconds: float | None = None,
        epistaxis_executable: str | None = None,
        wezterm_executable: str | None = None,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._runner = runner
        self._timeout_seconds = timeout_seconds
        self._epistaxis_executable = epistaxis_executable
        self._wezterm_executable = (
            wezterm_executable
            or os.environ.get("WEZTERM_CLI", "").strip()
            or "/Applications/WezTerm.app/Contents/MacOS/wezterm"
        )
        self._snapshot_path = Path(
            snapshot_path
            or Path.home()
            / ".local"
            / "state"
            / "epistaxis"
            / "live-diauloi.json"
        ).expanduser()

    def load(self) -> list[DiaulosCandidate]:
        try:
            payload = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DiaulosInventoryError(
                f"live Diaulos snapshot is missing at {self._snapshot_path}"
            ) from exc
        except OSError as exc:
            raise DiaulosInventoryError(
                f"live Diaulos snapshot could not be read: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DiaulosInventoryError(
                f"live Diaulos snapshot returned invalid JSON: {exc}"
            ) from exc
        return parse_live_inventory(payload)

    def refresh(self) -> list[DiaulosCandidate]:
        command = ["epistaxis", "diaulos", "live", "--json"]
        result = self._run_epistaxis(command)
        if result.returncode:
            raise DiaulosInventoryError(
                result.stderr.strip() or f"live Diaulos inventory exited {result.returncode}"
            )
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DiaulosInventoryError(f"live Diaulos inventory returned invalid JSON: {exc}") from exc
        candidates = parse_live_inventory(payload)
        self._replace_snapshot(payload)
        return candidates

    def activate(self, candidate: DiaulosCandidate) -> dict[str, Any]:
        list_command = self._wezterm_command("list", "--format", "json")
        result = self._run_process(list_command, DiaulosActivationError)
        if result.returncode:
            raise DiaulosActivationError(
                result.stderr.strip()
                or f"WezTerm pane enumeration exited {result.returncode}"
            )
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DiaulosActivationError(
                f"WezTerm pane enumeration returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise DiaulosActivationError(
                "WezTerm pane enumeration returned a non-list payload"
            )

        matches: list[dict[str, Any]] = []
        for row in payload:
            try:
                pane_id = _optional_int(row.get("pane_id"), "live pane_id")
            except DiaulosInventoryError as exc:
                raise DiaulosActivationError(str(exc)) from exc
            if pane_id == candidate.pane_id:
                matches.append(row)
        if len(matches) != 1:
            raise DiaulosActivationError(
                "selected pane is not present exactly once in the current WezTerm observation"
            )
        live = matches[0]
        self._verify_live_route(candidate, live)

        activate_command = self._wezterm_command(
            "activate-pane",
            "--pane-id",
            str(candidate.pane_id),
        )
        result = self._run_process(activate_command, DiaulosActivationError)
        if result.returncode:
            raise DiaulosActivationError(
                result.stderr.strip() or f"WezTerm activation exited {result.returncode}"
            )
        return {
            "diaulos": candidate.handle,
            "pane_id": candidate.pane_id,
            "expected_pane_id": candidate.pane_id,
            "tab_id": candidate.tab_id,
            "window_id": candidate.window_id,
            "verification": "direct-wezterm-pane-enumeration",
        }

    def _run_epistaxis(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        effective_command = list(command)
        executable = self._epistaxis_executable or shutil.which(
            "epistaxis",
            path=_epistaxis_search_path(),
        )
        if not executable:
            raise DiaulosInventoryError(
                "Epistaxis command is unavailable; searched the GUI-safe operator path"
            )
        effective_command[0] = executable
        return self._run_process(effective_command, DiaulosInventoryError)

    def _run_process(
        self,
        command: list[str],
        error: type[DiaulosInventoryError] | type[DiaulosActivationError],
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise error(f"command failed before a receipt: {exc}") from exc

    def _replace_snapshot(self, payload: dict[str, Any]) -> None:
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._snapshot_path.name}.",
            dir=self._snapshot_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._snapshot_path)
        except OSError as exc:
            raise DiaulosInventoryError(
                f"live Diaulos snapshot could not be replaced: {exc}"
            ) from exc
        finally:
            temporary_path.unlink(missing_ok=True)

    def _wezterm_command(self, *args: str) -> list[str]:
        command = [self._wezterm_executable, "cli"]
        extra = os.environ.get(
            "EPISTAXIS_WEZTERM_CLI_ARGS",
            "--no-auto-start",
        ).strip()
        command.extend(shlex.split(extra))
        command.extend(args)
        return command

    @staticmethod
    def _verify_live_route(
        candidate: DiaulosCandidate,
        live: dict[str, Any],
    ) -> None:
        comparisons = (
            ("tab_id", candidate.tab_id),
            ("window_id", candidate.window_id),
        )
        for field, expected in comparisons:
            if expected is None:
                continue
            try:
                observed = _optional_int(live.get(field), f"live {field}")
            except DiaulosInventoryError as exc:
                raise DiaulosActivationError(str(exc)) from exc
            if observed != expected:
                raise DiaulosActivationError(
                    f"selected pane {field} changed from {expected} to {observed}"
                )
        expected_cwd = _normalize_pane_cwd(candidate.cwd)
        observed_cwd = _normalize_pane_cwd(str(live.get("cwd") or ""))
        if expected_cwd and observed_cwd != expected_cwd:
            raise DiaulosActivationError(
                f"selected pane cwd changed from {expected_cwd} to {observed_cwd or 'missing'}"
            )


def _epistaxis_search_path() -> str:
    return os.pathsep.join(
        (
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
            str(Path.home() / ".local" / "bin"),
        )
    )


def _normalize_pane_cwd(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "://" not in value:
        return str(Path(value).expanduser())
    parsed = urlparse(value)
    return unquote(parsed.path)


def _candidate_matches(candidate: DiaulosCandidate, terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    haystack = _normalize_search(candidate.searchable_text)
    return all(term in haystack for term in terms)


def _search_terms(value: str) -> tuple[str, ...]:
    return tuple(part for part in _normalize_search(value).split() if part)


def _normalize_search(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _required_int(value: Any, field: str) -> int:
    parsed = _optional_int(value, field)
    if parsed is None:
        raise DiaulosInventoryError(f"{field} is missing")
    return parsed


def _optional_int(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DiaulosInventoryError(f"{field} is not an integer") from exc
