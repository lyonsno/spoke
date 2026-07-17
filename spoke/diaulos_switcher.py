"""Pure model and Epistaxis client for the live Diaulos switcher."""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Sequence


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
    tab_id: int | None
    window_id: int | None
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
        if identity in identities:
            raise DiaulosInventoryError(
                f"live Diaulos inventory repeats {handle} on pane {pane_id}"
            )
        identities.add(identity)
        aliases_raw = row.get("aliases") or []
        if not isinstance(aliases_raw, list):
            raise DiaulosInventoryError(f"live Diaulos row {index} aliases are not a list")
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
                tab_id=_optional_int(row.get("tab_id"), f"row {index} tab_id"),
                window_id=_optional_int(row.get("window_id"), f"row {index} window_id"),
                title=str(row.get("title") or "").strip(),
                cwd=str(row.get("cwd") or "").strip(),
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
        timeout_seconds: float = 4.0,
    ) -> None:
        self._runner = runner
        self._timeout_seconds = timeout_seconds

    def load(self) -> list[DiaulosCandidate]:
        command = ["epistaxis", "diaulos", "live", "--json"]
        result = self._run(command)
        if result.returncode:
            raise DiaulosInventoryError(
                result.stderr.strip() or f"live Diaulos inventory exited {result.returncode}"
            )
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DiaulosInventoryError(f"live Diaulos inventory returned invalid JSON: {exc}") from exc
        return parse_live_inventory(payload)

    def activate(self, candidate: DiaulosCandidate) -> dict[str, Any]:
        command = [
            "epistaxis",
            "focus-pane",
            "--diaulos",
            candidate.handle,
            "--expected-pane-id",
            str(candidate.pane_id),
            "--json",
        ]
        result = self._run(command)
        if result.returncode:
            raise DiaulosActivationError(
                result.stderr.strip() or f"Diaulos activation exited {result.returncode}"
            )
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DiaulosActivationError(f"Diaulos activation returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise DiaulosActivationError("Diaulos activation receipt is not an object")
        if str(payload.get("diaulos") or "") != candidate.handle:
            raise DiaulosActivationError("Diaulos activation receipt names a different handle")
        try:
            pane_id = _optional_int(payload.get("pane_id"), "activation pane_id")
            expected_pane_id = _optional_int(
                payload.get("expected_pane_id"),
                "activation expected_pane_id",
            )
        except DiaulosInventoryError as exc:
            raise DiaulosActivationError(str(exc)) from exc
        if pane_id != candidate.pane_id:
            raise DiaulosActivationError("Diaulos activation receipt names a different pane")
        if expected_pane_id != candidate.pane_id:
            raise DiaulosActivationError("Diaulos activation receipt did not preserve expected pane")
        return payload

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = (
                DiaulosInventoryError
                if command[1:3] == ["diaulos", "live"]
                else DiaulosActivationError
            )
            raise error(f"Epistaxis command failed before a receipt: {exc}") from exc


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
