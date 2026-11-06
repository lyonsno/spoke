"""Read-only sidebar projection of Greenroom's operator attention snapshot."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def attention_snapshot_path(override: str | Path | None = None) -> Path:
    explicit = override or os.environ.get("SPOKE_GREENROOM_ATTENTION_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("GPU_GREENROOM_DIR", "~/.local/state/gpu-greenroom")).expanduser() / "operator-attention.json"


def _web_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("missing monitor URL")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("expected HTTP(S) monitor URL")
    return value


def _timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("expected RFC3339 timestamp")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("timestamp needs timezone")
    return date.timestamp()


@dataclass(frozen=True)
class GreenroomFooter:
    queue_text: str
    attention_text: str
    tooltip: str
    monitor_url: str | None = None
    attention_url: str | None = None
    freshness: str = "unavailable"


class GreenroomAttentionReader:
    """Stat on the existing sidebar cadence; decode only when the file changes.

    The producer supplies its freshness deadline and the complete unresolved
    item set. This view neither owns requests nor infers status transitions.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = attention_snapshot_path(path)
        self._stamp = None
        self._snapshot = None
        self._error = "snapshot unavailable"

    def _decode(self, raw):
        if not isinstance(raw, dict) or raw.get("schema") != "gpu-greenroom.operator-attention.v1":
            raise ValueError("unsupported attention schema")
        observed = _timestamp(raw.get("observed_at"))
        deadline = _timestamp(raw.get("valid_until"))
        if deadline < observed:
            raise ValueError("freshness deadline precedes observation")
        monitor = _web_url(raw.get("monitor_url"))
        queue = raw.get("queue")
        if not isinstance(queue, dict):
            raise ValueError("missing queue summary")
        for key in ("running", "waiting"):
            if type(queue.get(key)) is not int or queue[key] < 0:
                raise ValueError("invalid queue counts")
        items = raw.get("items")
        if not isinstance(items, list):
            raise ValueError("missing unresolved items")
        ids = set()
        for item in items:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("id", "title", "status")):
                raise ValueError("invalid attention item")
            if item["id"] in ids:
                raise ValueError("duplicate attention identity")
            ids.add(item["id"])
            _web_url(item.get("target_url"))
        return raw, observed, deadline, monitor

    def read(self, now: float | None = None) -> GreenroomFooter:
        try:
            stat = self.path.stat()
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if stamp != self._stamp:
                self._snapshot = None
                self._stamp = stamp
                try:
                    self._snapshot = self._decode(json.loads(self.path.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError, OverflowError):
                    self._error = "invalid attention snapshot"
        except OSError:
            self._stamp = None
            self._snapshot = None
            self._error = "snapshot unavailable"
        if self._snapshot is None:
            return GreenroomFooter("Greenroom · unavailable", "Status unavailable", f"{self._error}\n{self.path}")
        raw, observed, deadline, monitor = self._snapshot
        clock = datetime.now(timezone.utc).timestamp() if now is None else now
        tooltip = f"Observed {raw['observed_at']}\nValid until {raw['valid_until']}\n{self.path}"
        if clock < observed or clock >= deadline:
            state = "stale" if clock >= deadline else "clock mismatch"
            return GreenroomFooter(f"Greenroom · {state}", f"Last seen: {len(raw['items'])} requests", tooltip, monitor_url=monitor, freshness=state)
        queue = raw["queue"]
        items = raw["items"]
        title = f"Greenroom · {queue['running']} running · {queue['waiting']} waiting"
        attention = f"⚠ {len(items)} · {items[0]['title']}" if items else "✓ No requests"
        return GreenroomFooter(title, attention, attention + "\n" + tooltip, monitor, items[0]["target_url"] if items else None, "fresh")
