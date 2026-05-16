"""Typed coordination surface stack — the tray reimagined.

The tray holds typed coordination surfaces, not raw text. Each surface is
a structured entry (agent thread, metadosis, zetesis result, finding,
Perceptasia view, etc.) that the operator can rock through with shift+space.

The primary (topmost) surface is expanded; all others render compact one-line
summaries. Voice acts on the primary surface, classified against that surface
type's action vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4


class SurfaceKind(str, Enum):
    """Known typed surface kinds in the coordination stack."""

    AGENT_THREAD = "agent_thread"
    METADOSIS = "metadosis"
    ZETESIS_RESULT = "zetesis_result"
    FINDING = "finding"
    PERCEPTASIA_VIEW = "perceptasia_view"
    METAMORPHOSIS_RESULT = "metamorphosis_result"
    TEXT = "text"  # legacy fallback — raw text entry


@dataclass
class SurfaceIdentity:
    """Stable identity for a coordination surface.

    Enough to address the surface across sessions: kind + a kind-scoped id.
    """

    kind: SurfaceKind
    surface_id: str  # kind-scoped unique id (e.g. provider_session_id, finding path)
    label: str = ""  # short human-readable label for compact display


@dataclass
class SurfaceEntry:
    """A single entry in the coordination surface stack.

    Each entry is a typed surface with identity, content payload, and
    display state. The payload is kind-specific opaque data consumed by
    the surface type's renderers.
    """

    identity: SurfaceIdentity
    payload: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = True
    priority: int = 0  # lower = more important, for insertion ordering

    @property
    def kind(self) -> SurfaceKind:
        return self.identity.kind

    @property
    def surface_id(self) -> str:
        return self.identity.surface_id

    @property
    def label(self) -> str:
        return self.identity.label or self.identity.surface_id


class SurfaceRenderer(Protocol):
    """Protocol for surface type renderers."""

    def compact(self, entry: SurfaceEntry) -> str:
        """One-line summary for non-primary display."""
        ...

    def expanded(self, entry: SurfaceEntry) -> str:
        """Multi-line expanded view for primary display."""
        ...


@dataclass
class SurfaceAction:
    """A voice action available on a surface type."""

    name: str  # internal action identifier
    phrases: list[str] = field(default_factory=list)  # example trigger phrases
    description: str = ""


@dataclass
class SurfaceTypeRegistration:
    """Registration entry for a surface type in the registry."""

    kind: SurfaceKind
    actions: list[SurfaceAction] = field(default_factory=list)
    renderer: SurfaceRenderer | None = None


class SurfaceTypeRegistry:
    """Extensible registry of surface type definitions.

    Each surface type registers its action vocabulary and renderer.
    New surface types can be added at runtime.
    """

    def __init__(self) -> None:
        self._types: dict[SurfaceKind, SurfaceTypeRegistration] = {}

    def register(self, registration: SurfaceTypeRegistration) -> None:
        self._types[registration.kind] = registration

    def get(self, kind: SurfaceKind) -> SurfaceTypeRegistration | None:
        return self._types.get(kind)

    def actions_for(self, kind: SurfaceKind) -> list[SurfaceAction]:
        reg = self._types.get(kind)
        return reg.actions if reg else []

    def renderer_for(self, kind: SurfaceKind) -> SurfaceRenderer | None:
        reg = self._types.get(kind)
        return reg.renderer if reg else None

    @property
    def registered_kinds(self) -> list[SurfaceKind]:
        return list(self._types.keys())


class CoordinationStack:
    """The typed coordination surface stack.

    Manages a list of SurfaceEntry objects with a current index (primary).
    The primary entry is expanded; others are compact. Navigation preserves
    the existing shift+space rocking semantics.
    """

    def __init__(self, registry: SurfaceTypeRegistry | None = None) -> None:
        self._entries: list[SurfaceEntry] = []
        self._index: int = 0
        self._active: bool = False
        self._registry = registry or SurfaceTypeRegistry()

    @property
    def entries(self) -> list[SurfaceEntry]:
        return self._entries

    @property
    def index(self) -> int:
        return self._index

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value

    @property
    def primary(self) -> SurfaceEntry | None:
        """The currently focused (expanded) surface, or None if empty."""
        if not self._entries:
            return None
        if self._index >= len(self._entries):
            self._index = len(self._entries) - 1
        return self._entries[self._index]

    @property
    def size(self) -> int:
        return len(self._entries)

    def push(self, entry: SurfaceEntry, *, to_top: bool = True) -> None:
        """Push a new surface into the stack.

        Args:
            entry: The surface to add.
            to_top: If True, insert at index 0 (newest on top).
                    If False, append to end.
        """
        if to_top:
            self._entries.insert(0, entry)
            if self._active:
                self._index += 1
            else:
                self._index = 0
        else:
            self._entries.append(entry)
            if not self._entries[:-1]:  # was empty before
                self._index = 0

    def push_by_priority(self, entry: SurfaceEntry) -> None:
        """Insert a surface at a position determined by its priority.

        Lower priority values sort toward the top (index 0).
        """
        insert_at = 0
        for i, existing in enumerate(self._entries):
            if existing.priority <= entry.priority:
                insert_at = i + 1
            else:
                break
        self._entries.insert(insert_at, entry)
        if self._active and insert_at <= self._index:
            self._index += 1

    def rock_up(self) -> SurfaceEntry | None:
        """Navigate toward index 0 (newer/higher priority)."""
        if not self._entries or not self._active:
            return None
        if self._index > 0:
            self._index -= 1
        return self.primary

    def rock_down(self) -> SurfaceEntry | None:
        """Navigate toward end (older/lower priority)."""
        if not self._entries or not self._active:
            return None
        if self._index < len(self._entries) - 1:
            self._index += 1
        return self.primary

    def rock_wrap_up(self) -> SurfaceEntry | None:
        """Navigate up with wraparound."""
        if not self._entries or not self._active:
            return None
        self._index = (self._index - 1) % len(self._entries)
        return self.primary

    def remove_current(self) -> SurfaceEntry | None:
        """Remove the current primary entry. Returns the removed entry."""
        if not self._entries:
            return None
        removed = self._entries.pop(self._index)
        if not self._entries:
            self._active = False
            self._index = 0
        elif self._index >= len(self._entries):
            self._index = len(self._entries) - 1
        return removed

    def remove_by_id(self, surface_id: str) -> SurfaceEntry | None:
        """Remove an entry by its surface_id. Returns removed or None."""
        for i, entry in enumerate(self._entries):
            if entry.surface_id == surface_id:
                removed = self._entries.pop(i)
                if self._index >= len(self._entries) and self._entries:
                    self._index = len(self._entries) - 1
                elif not self._entries:
                    self._active = False
                    self._index = 0
                elif i < self._index:
                    self._index -= 1
                return removed
        return None

    def find_by_id(self, surface_id: str) -> SurfaceEntry | None:
        """Find an entry by its surface_id."""
        for entry in self._entries:
            if entry.surface_id == surface_id:
                return entry
        return None

    def find_by_kind(self, kind: SurfaceKind) -> list[SurfaceEntry]:
        """Find all entries of a given kind."""
        return [e for e in self._entries if e.kind == kind]

    def activate(self) -> SurfaceEntry | None:
        """Activate the stack (show it). Returns current primary."""
        if not self._entries:
            return None
        self._active = True
        if self._index >= len(self._entries):
            self._index = len(self._entries) - 1
        return self.primary

    def deactivate(self) -> None:
        """Deactivate the stack (hide it)."""
        self._active = False

    def compact_summary(self, entry: SurfaceEntry) -> str:
        """Get compact one-line summary for an entry."""
        renderer = self._registry.renderer_for(entry.kind)
        if renderer:
            return renderer.compact(entry)
        return entry.label

    def expanded_view(self, entry: SurfaceEntry) -> str:
        """Get expanded multi-line view for an entry."""
        renderer = self._registry.renderer_for(entry.kind)
        if renderer:
            return renderer.expanded(entry)
        return entry.label

    def action_vocabulary(self) -> list[SurfaceAction]:
        """Get the action vocabulary for the current primary surface type."""
        primary = self.primary
        if not primary:
            return []
        return self._registry.actions_for(primary.kind)


# ---------------------------------------------------------------------------
# Legacy bridge: convert old TrayEntry/str items to SurfaceEntry
# ---------------------------------------------------------------------------


def text_surface_from_str(text: str, *, owner: str = "user") -> SurfaceEntry:
    """Create a TEXT surface entry from a raw string (legacy tray compat)."""
    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.TEXT,
            surface_id=f"text-{uuid4().hex[:8]}",
            label=text[:60] if text else "",
        ),
        payload={"text": text, "owner": owner},
        acknowledged=(owner != "assistant"),
    )
