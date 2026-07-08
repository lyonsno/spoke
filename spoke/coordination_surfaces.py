"""Typed coordination surface stack — the tray reimagined.

The tray holds typed coordination surfaces, not raw text. Each surface is
a structured entry (agent thread, metadosis, zetesis result, finding,
Perceptasia view, etc.) that the operator can rock through with shift+space.

The primary (topmost) surface is expanded; all others render compact one-line
summaries. Voice acts on the primary surface, classified against that surface
type's action vocabulary.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4


class SurfaceKind(str, Enum):
    """Known typed surface kinds in the coordination stack."""

    AGENT_THREAD = "agent_thread"
    METADOSIS = "metadosis"
    ZETESIS_RESULT = "zetesis_result"
    FINDING = "finding"
    PERCEPTASIA_VIEW = "perceptasia_view"
    METAMORPHOSIS_RESULT = "metamorphosis_result"
    DIAULOS = "diaulos"
    TEXT = "text"  # legacy fallback — raw text entry


class StackOrderingMode(str, Enum):
    """How the stack currently orders its working set."""

    ARRIVAL = "arrival"
    PRIORITY = "priority"


class SurfaceDestinationKind(str, Enum):
    """Where a stack-focused act is routed for interpretation.

    The stack supplies address and cargo; it is not itself the durable writer
    for source-owned state.
    """

    NONE = "none"
    STACK = "stack"
    DIAULOS = "diaulos"
    LANE = "lane"
    RESOLVER = "resolver"
    SOURCE_ORGAN = "source_organ"


@dataclass
class SurfaceIdentity:
    """Stable identity for a coordination surface.

    Enough to address the surface across sessions: kind + a globally unique id.
    """

    kind: SurfaceKind
    surface_id: str  # globally unique across the stack (e.g. provider_session_id, finding path)
    label: str = ""  # short human-readable label for compact display


@dataclass
class SurfaceRoutingContext:
    """Address/cargo metadata for agency-mediated stack actions."""

    destination_kind: SurfaceDestinationKind = SurfaceDestinationKind.NONE
    destination_id: str = ""
    reread_refs: list[str] = field(default_factory=list)
    scope: dict[str, list[str]] = field(default_factory=dict)
    cargo: dict[str, Any] = field(default_factory=dict)
    writeback_target: str = ""


@dataclass
class SurfaceEntry:
    """A single entry in the coordination surface stack.

    Each entry is a typed surface with identity, content payload, and
    display state. The payload is kind-specific opaque data consumed by
    the surface type's renderers.
    """

    identity: SurfaceIdentity
    payload: dict[str, Any] = field(default_factory=dict)
    routing: SurfaceRoutingContext | None = None
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
    interlocutor_act: str = ""
    requires_interlocutor: bool = False
    source_owned: bool = False
    writeback_allowed: bool = False


@dataclass
class SurfaceTypeRegistration:
    """Registration entry for a surface type in the registry."""

    kind: SurfaceKind
    actions: list[SurfaceAction] = field(default_factory=list)
    renderer: SurfaceRenderer | None = None


class DiaulosCardRenderer:
    """Read-only renderer for Diaulos identity cards."""

    def compact(self, entry: SurfaceEntry) -> str:
        status = str(entry.payload.get("status") or "").strip()
        if status:
            return f"Diaulos: {entry.label} · {status}"
        return f"Diaulos: {entry.label}"

    def expanded(self, entry: SurfaceEntry) -> str:
        diaulos = str(entry.payload.get("diaulos") or "").strip()
        diaulos_id = str(entry.payload.get("diaulos_id") or "").strip()
        topos = str(entry.payload.get("topos") or "").strip()
        source_topoi = _string_list(entry.payload.get("source_topoi"))
        custody_refs = _string_list(entry.payload.get("custody_refs"))
        warnings = _string_list(entry.payload.get("warnings"))
        status = str(entry.payload.get("status") or "").strip()
        summary = str(entry.payload.get("summary") or "").strip()

        heading = f"Diaulos: {entry.label}"
        if status:
            heading = f"{heading} · {status}"
        lines = [heading]
        if diaulos:
            lines.append(f"Handle: {diaulos}")
        if diaulos_id:
            lines.append(f"ID: {diaulos_id}")
        custody_labels = [_ref_label(ref) for ref in (custody_refs or ([topos] if topos else []))]
        custody_labels = [label for label in custody_labels if label]
        if custody_labels:
            lines.append(f"Custody: {', '.join(custody_labels[:2])}")
        if source_topoi:
            source_labels = [_ref_label(ref) for ref in source_topoi]
            caveat = ""
            if "current_topos_not_registry_source_topos" in warnings:
                caveat = " (currentness caveat)"
            lines.append(f"Registry source: {', '.join(source_labels[:2])}{caveat}")
        if summary and not (source_topoi or custody_refs or warnings):
            lines.append(summary)
        lines.append("Read-only card: selection/expansion only; no focus/send/writeback.")
        return "\n".join(lines)


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
        self._ordering_mode = StackOrderingMode.ARRIVAL
        self._arrival_counter = 0
        self._arrival_order: dict[int, int] = {}

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
    def ordering_mode(self) -> StackOrderingMode:
        return self._ordering_mode

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

    def _record_arrival(self, entry: SurfaceEntry) -> None:
        key = id(entry)
        if key in self._arrival_order:
            return
        self._arrival_order[key] = self._arrival_counter
        self._arrival_counter += 1

    def _priority_key(self, entry: SurfaceEntry) -> tuple[int, int]:
        return (entry.priority, self._arrival_order.get(id(entry), 0))

    def _set_index_to_entry(self, pivot: SurfaceEntry | None) -> None:
        if not self._entries:
            self._index = 0
            return
        if pivot is None:
            self._index = 0
            return
        try:
            self._index = self._entries.index(pivot)
        except ValueError:
            self._index = min(self._index, len(self._entries) - 1)

    def set_ordering_mode(self, mode: StackOrderingMode) -> None:
        """Switch stack ordering while preserving the current primary surface."""
        pivot = self.primary if self._active else None
        self._ordering_mode = mode
        if mode == StackOrderingMode.PRIORITY:
            self._entries.sort(key=self._priority_key)
        self._set_index_to_entry(pivot)

    def push(self, entry: SurfaceEntry, *, to_top: bool = True) -> None:
        """Push a new surface into the stack.

        Args:
            entry: The surface to add.
            to_top: If True, insert at index 0 (newest on top).
                    If False, append to end.
        """
        self._record_arrival(entry)
        if self._ordering_mode == StackOrderingMode.PRIORITY:
            pivot = self.primary if self._active else None
            self._entries.append(entry)
            self._entries.sort(key=self._priority_key)
            self._set_index_to_entry(pivot)
            return

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
        self._record_arrival(entry)
        pivot = self.primary if self._active else None
        self._entries.append(entry)
        self.set_ordering_mode(StackOrderingMode.PRIORITY)
        self._set_index_to_entry(pivot)

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

    def focus_by_id(self, surface_id: str) -> SurfaceEntry | None:
        """Focus an entry by surface_id without changing active state."""
        for i, entry in enumerate(self._entries):
            if entry.surface_id == surface_id:
                self._index = i
                return entry
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
        """Get the action vocabulary for the current primary surface type.

        Returns empty when the stack is inactive — voice routing should
        not resolve against a surface the operator isn't looking at.
        """
        if not self._active:
            return []
        primary = self.primary
        if not primary:
            return []
        return self._registry.actions_for(primary.kind)


# ---------------------------------------------------------------------------
# Legacy bridge: convert old TrayEntry/str items to SurfaceEntry
# ---------------------------------------------------------------------------


def surface_actions_to_resolver_intents(actions: list[SurfaceAction]) -> list[dict]:
    """Convert surface actions to the shape expected by ConsensusResolver.

    Returns a list of dicts with keys: id, description, examples, plus
    routing metadata consumed by agency-aware resolvers.
    This avoids a hard import dependency on consensus_resolver (which may
    not be on main yet) while producing compatible data.
    """
    return [
        {
            "id": action.name,
            "description": action.description or action.name,
            "examples": tuple(action.phrases),
            "interlocutor_act": action.interlocutor_act,
            "requires_interlocutor": action.requires_interlocutor,
            "source_owned": action.source_owned,
            "writeback_allowed": action.writeback_allowed,
        }
        for action in actions
    ]


def build_default_registry() -> SurfaceTypeRegistry:
    """Build the default surface type registry with action vocabularies.

    Action vocabularies define what voice commands are available when each
    surface type is primary. The consensus resolver classifies utterances
    against these.
    """
    reg = SurfaceTypeRegistry()

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.AGENT_THREAD,
        actions=[
            SurfaceAction(
                "start",
                ["start this", "go", "run it"],
                "Route start/resume request to the agent session",
                "route_agent_start",
                True,
                True,
                True,
            ),
            SurfaceAction(
                "status",
                ["what did it just do", "status", "what's happening"],
                "Ask the agent session for current status",
                "ask_agent_status",
                True,
                False,
                False,
            ),
            SurfaceAction(
                "output",
                ["show me the output", "show output", "read it"],
                "Ask the agent session for recent output",
                "ask_agent_output",
                True,
                False,
                False,
            ),
            SurfaceAction(
                "cancel",
                ["cancel", "stop", "kill it"],
                "Route cancellation request to the agent session",
                "route_agent_cancel",
                True,
                True,
                True,
            ),
            SurfaceAction("dismiss", ["dismiss", "close", "done"], "Remove from stack"),
        ],
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.METADOSIS,
        actions=[
            SurfaceAction(
                "update",
                ["update the thesis", "update"],
                "Route update request to the artifact custodian",
                "route_update_to_custodian",
                True,
                True,
                True,
            ),
            SurfaceAction(
                "blocking",
                ["what's blocking this", "blockers"],
                "Ask the artifact custodian for blockers",
                "ask_artifact_blockers",
                True,
                False,
                False,
            ),
            SurfaceAction(
                "broadcast",
                ["send this to all lanes", "broadcast"],
                "Route broadcast request through the registered lane surface",
                "route_broadcast_to_lanes",
                True,
                True,
                True,
            ),
            SurfaceAction("dismiss", ["dismiss", "close"], "Remove from stack"),
        ],
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.ZETESIS_RESULT,
        actions=[
            SurfaceAction("read", ["read this", "read it"], "Read the result aloud"),
            SurfaceAction(
                "act",
                ["act on this", "do it"],
                "Route result to the current source-owned actor",
                "route_result_to_actor",
                True,
                True,
                True,
            ),
            SurfaceAction("dismiss", ["dismiss", "close", "done"], "Remove from stack"),
        ],
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.FINDING,
        actions=[
            SurfaceAction(
                "accept",
                ["accept", "ok", "acknowledge"],
                "Route finding disposition through the finding owner",
                "route_finding_disposition",
                True,
                True,
                True,
            ),
            SurfaceAction(
                "defer",
                ["defer", "later", "not now"],
                "Route finding deferral through the finding owner",
                "route_finding_disposition",
                True,
                True,
                True,
            ),
            SurfaceAction("navigate", ["navigate to the commit", "show me", "go to"], "Navigate to source"),
            SurfaceAction("dismiss", ["dismiss", "close"], "Remove from stack"),
        ],
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.PERCEPTASIA_VIEW,
        actions=[
            SurfaceAction(
                "show_attractors",
                ["show me the attractors", "attractors"],
                "Ask the Perceptasia context for attractors",
                "ask_perceptasia_context",
                True,
                False,
                False,
            ),
            SurfaceAction("zoom", ["zoom in", "zoom in on this"], "Zoom into selection"),
            SurfaceAction(
                "hot",
                ["what's hot", "what's active"],
                "Ask the Perceptasia context for hot items",
                "ask_perceptasia_context",
                True,
                False,
                False,
            ),
            SurfaceAction("dismiss", ["dismiss", "close"], "Remove from stack"),
        ],
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.METAMORPHOSIS_RESULT,
        actions=[
            SurfaceAction(
                "confirm",
                ["confirm", "ok", "looks good"],
                "Route mutation confirmation through the source-owned surface",
                "route_mutation_confirmation",
                True,
                True,
                True,
            ),
            SurfaceAction(
                "revert",
                ["revert", "undo", "roll back"],
                "Route mutation revert through the source-owned surface",
                "route_mutation_revert",
                True,
                True,
                True,
            ),
            SurfaceAction("dismiss", ["dismiss", "close"], "Remove from stack"),
        ],
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.DIAULOS,
        actions=[
            SurfaceAction("dismiss", ["dismiss", "close"], "Remove from stack"),
        ],
        renderer=DiaulosCardRenderer(),
    ))

    reg.register(SurfaceTypeRegistration(
        kind=SurfaceKind.TEXT,
        actions=[
            SurfaceAction("send", ["send", "send this", "to assistant"], "Send to assistant"),
            SurfaceAction("paste", ["paste", "insert"], "Paste at cursor"),
            SurfaceAction("dismiss", ["dismiss", "delete", "close"], "Remove from stack"),
        ],
    ))

    return reg


def text_surface_from_str(text: str, *, owner: str = "user") -> SurfaceEntry:
    """Create a TEXT surface entry from a raw string (legacy tray compat)."""
    label = " ".join(text.split())[:60] if text else ""
    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.TEXT,
            surface_id=f"text-{uuid4().hex[:8]}",
            label=label,
        ),
        payload={"text": text, "owner": owner},
        acknowledged=(owner != "assistant"),
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _ref_label(value: str) -> str:
    ref = str(value).strip()
    if not ref:
        return ""
    if "#" in ref:
        anchor = ref.rsplit("#", 1)[1].strip()
        if anchor:
            return anchor
    return Path(ref).name or ref


def _diaulos_refs(record: dict[str, Any]) -> dict[str, list[str]]:
    raw_refs = record.get("refs")
    if not isinstance(raw_refs, dict):
        return {}
    return {
        str(key): _string_list(value)
        for key, value in raw_refs.items()
        if _string_list(value)
    }


def diaulos_surface_from_record(record: dict[str, Any]) -> SurfaceEntry:
    """Project a Diaulos identity record into a read-only Stack card.

    The input is intentionally dict-shaped for now because the authoritative
    Diaulos resolver surfaces are still settling. This function preserves only
    identity/display facts and refuses to imply live pane focus, directive
    sending, or operator-memory write authority.
    """
    diaulos = str(record.get("diaulos") or "").strip()
    diaulos_id = str(record.get("diaulos_id") or "").strip()
    display_name = str(record.get("display_name") or "").strip()
    topos = str(record.get("topos") or "").strip()
    source_topoi = _string_list(record.get("source_topoi"))
    custody_refs = _string_list(record.get("custody_refs"))
    warnings = _string_list(record.get("warnings"))
    status = str(record.get("status") or "").strip()
    summary = str(record.get("summary") or "").strip()
    refs = _diaulos_refs(record)

    stable_id = diaulos_id or diaulos
    if not stable_id:
        raise ValueError("diaulos record requires diaulos_id or diaulos")

    label = display_name or diaulos or diaulos_id
    reread_refs = [topos] if topos else []
    for ref in source_topoi:
        if ref not in reread_refs:
            reread_refs.append(ref)
    for ref in custody_refs:
        if ref not in reread_refs:
            reread_refs.append(ref)
    for ref in refs.get("topoi", []):
        if ref not in reread_refs:
            reread_refs.append(ref)

    payload = {
        "diaulos": diaulos,
        "diaulos_id": diaulos_id,
        "display_name": display_name,
        "topos": topos,
        "source_topoi": source_topoi,
        "custody_refs": custody_refs,
        "warnings": warnings,
        "status": status,
        "summary": summary,
        "refs": refs,
    }

    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.DIAULOS,
            surface_id=f"diaulos:{stable_id}",
            label=label,
        ),
        payload=payload,
        routing=SurfaceRoutingContext(
            destination_kind=SurfaceDestinationKind.DIAULOS,
            destination_id=stable_id,
            reread_refs=reread_refs,
            scope={"diauloi": [stable_id]},
            cargo={
                "diaulos": diaulos,
                "diaulos_id": diaulos_id,
                "topos": topos,
                "source_topoi": source_topoi,
                "custody_refs": custody_refs,
                "warnings": warnings,
                "refs": refs,
                "authority": "read_only_identity_fact",
                "may_focus_pane": False,
                "may_write_state": False,
                "may_send_directive": False,
            },
            writeback_target="",
        ),
        acknowledged=True,
    )


# ---------------------------------------------------------------------------
# Message bus: async delivery of surfaces into the stack
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMessage:
    """A message delivering a surface into the stack from an async source."""

    entry: SurfaceEntry
    source: str = ""  # identifier for the producer (e.g. "agent_shell", "zetesis")
    activate: bool = False  # whether to activate/show the stack on delivery
    position: str = "top"  # "top" or "priority"


class SurfaceMessageBus:
    """Thread-safe message bus for async surface delivery into a CoordinationStack.

    Producers post SurfaceMessages from any thread. The main thread drains
    pending messages into the stack on its event loop (or via explicit drain).

    Usage:
        bus = SurfaceMessageBus(stack)
        # From background thread:
        bus.post(SurfaceMessage(entry=my_surface, source="zetesis"))
        # From main thread event loop:
        delivered = bus.drain()
    """

    def __init__(
        self,
        stack: CoordinationStack,
        *,
        on_delivery: Callable[[SurfaceEntry], None] | None = None,
    ) -> None:
        self._stack = stack
        self._queue: deque[SurfaceMessage] = deque()
        self._lock = threading.Lock()
        self._on_delivery = on_delivery

    def post(self, message: SurfaceMessage) -> None:
        """Post a surface message from any thread. Thread-safe."""
        with self._lock:
            self._queue.append(message)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def drain(self) -> list[SurfaceEntry]:
        """Drain all pending messages into the stack. Call from main thread.

        Returns the list of entries that were delivered.
        """
        with self._lock:
            messages = list(self._queue)
            self._queue.clear()

        delivered: list[SurfaceEntry] = []
        for msg in messages:
            if msg.position == "priority":
                self._stack.push_by_priority(msg.entry)
            else:
                self._stack.push(msg.entry, to_top=True)

            if msg.activate:
                self._stack.activate()

            delivered.append(msg.entry)

            if self._on_delivery:
                self._on_delivery(msg.entry)

        return delivered


# ---------------------------------------------------------------------------
# Operator-ping tokens: ephemeral source-linked attention near the Stack body
# ---------------------------------------------------------------------------


OPERATOR_PING_TOKEN_ANCHOR = "operator_stack_body"
OPERATOR_PING_TOKEN_GAP = 8.0
OPERATOR_PING_TOKEN_WIDTH = 248.0
OPERATOR_PING_TOKEN_HEIGHT = 28.0
OPERATOR_PING_TOKEN_CASCADE_X = 12.0
OPERATOR_PING_TOKEN_CASCADE_Y = 34.0


@dataclass(frozen=True)
class OperatorPingToken:
    """Ephemeral presentation token derived from a source-owned ping event.

    These tokens are not CoordinationStack rows. They carry enough source
    cargo for a gesture to expand or navigate back to the ping source while
    preserving the event as the only authority.
    """

    ping_id: str
    source_event_id: str
    source_signature: str
    label: str
    anchor: str = OPERATOR_PING_TOKEN_ANCHOR
    diaulos: str = ""
    topos: str = ""
    thread_id: str = ""
    pane_id: str = ""
    session_address: str = ""
    message: str = ""
    reason: str = ""
    reason_token: str = ""
    refs: dict[str, list[str]] = field(default_factory=dict)

    def activation_routing(self, *, gesture: str = "select") -> SurfaceRoutingContext:
        """Return read-only routing cargo for expanding or navigating the ping source."""
        reread_refs = _operator_ping_reread_refs(self.topos, self.refs)
        scope: dict[str, list[str]] = {"operator_pings": [self.ping_id]}
        if reread_refs:
            scope["topoi"] = reread_refs

        return SurfaceRoutingContext(
            destination_kind=SurfaceDestinationKind.SOURCE_ORGAN,
            destination_id=f"operator_ping:{self.ping_id}",
            reread_refs=reread_refs,
            scope=scope,
            cargo={
                "gesture": gesture,
                "ping_id": self.ping_id,
                "source_event_id": self.source_event_id,
                "source_signature": self.source_signature,
                "diaulos": self.diaulos,
                "topos": self.topos,
                "thread_id": self.thread_id,
                "pane_id": self.pane_id,
                "session_address": self.session_address,
                "message": self.message,
                "reason": self.reason,
                "reason_token": self.reason_token,
                "authority": "event_fact_only",
                "may_clear_ping": False,
                "may_focus_pane": False,
                "may_write_state": False,
            },
            writeback_target="",
        )


@dataclass(frozen=True)
class TokenVisualFrame:
    """Renderer-neutral rectangle for an ephemeral token visual."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class OperatorPingTokenVisual:
    """Renderer-neutral visual presentation for an operator-ping token."""

    ping_id: str
    visual_index: int
    diagnostic_count: int
    frame: TokenVisualFrame
    presentation_text: str
    accessibility_label: str
    anchor: str = OPERATOR_PING_TOKEN_ANCHOR
    style_role: str = "quiet_source_spark"
    authority: str = "event_fact_only"
    steals_primary_focus: bool = False


def _operator_ping_reread_refs(
    topos: str,
    refs: dict[str, list[str]] | None,
) -> list[str]:
    reread_refs: list[str] = []
    if topos:
        reread_refs.append(topos)
    if refs:
        for ref in refs.get("topoi", []):
            if ref and ref not in reread_refs:
                reread_refs.append(ref)
    return reread_refs


def _operator_ping_source_signature(ping: dict[str, Any]) -> str:
    diaulos = str(ping.get("diaulos") or "").strip()
    if diaulos:
        return f"Diaulos: {diaulos}"
    topos = str(ping.get("topos") or "").strip()
    if topos:
        return f"Topos: {topos}"
    thread_id = str(ping.get("thread_id") or "").strip()
    if thread_id:
        return f"Thread: {thread_id}"
    ping_id = str(ping.get("ping_id") or "").strip()
    return f"Ping: {ping_id}" if ping_id else "Operator ping"


def _operator_ping_token_label(ping: dict[str, Any]) -> str:
    reason_token = str(ping.get("reason_token") or "").strip()
    if reason_token:
        return reason_token
    message = " ".join(str(ping.get("message") or "").split())
    if message:
        return message[:60]
    return "operator ping"


def _operator_ping_visual_text(token: OperatorPingToken) -> str:
    parts = [token.source_signature]
    if token.label:
        parts.append(token.label)
    return " · ".join(parts)


def _operator_ping_visual_accessibility_label(token: OperatorPingToken) -> str:
    label = token.label or token.message or token.ping_id
    return f"Operator ping from {token.source_signature}: {label}"


def _operator_ping_created_at(ping: dict[str, Any], event: dict[str, Any]) -> str:
    return str(ping.get("created_at") or event.get("observed_at") or "")


def _operator_ping_sort_key(event: dict[str, Any]) -> tuple[str, str]:
    ping = event.get("operator_ping") if isinstance(event.get("operator_ping"), dict) else {}
    return (_operator_ping_created_at(ping, event), str(event.get("event_id") or ""))


def load_operator_ping_events_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load operator ping rows from a private coordination JSONL stream."""
    event_log = Path(path).expanduser()
    if not event_log.exists():
        return []

    events: list[dict[str, Any]] = []
    with event_log.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid operator ping event JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"invalid operator ping event JSON on line {line_number}: row is not an object"
                )
            if str(event.get("kind", "")).startswith("operator_ping."):
                events.append(event)
    return events


def derive_operator_ping_tokens(
    events: list[dict[str, Any]],
    *,
    stack: CoordinationStack | None = None,
) -> list[OperatorPingToken]:
    """Project unresolved operator pings into ephemeral source-linked tokens.

    Replays append-only coordination events: latest create for a ping id wins,
    clear removes that id from the unresolved set. The optional stack argument
    is accepted to make the non-mutation contract explicit; it is never
    inspected or mutated.
    """
    del stack

    unresolved: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = event.get("kind")
        ping = event.get("operator_ping")
        if not isinstance(ping, dict):
            continue
        ping_id = str(ping.get("ping_id") or "").strip()
        if not ping_id:
            continue
        if kind == "operator_ping.created":
            unresolved[ping_id] = event
        elif kind == "operator_ping.cleared":
            unresolved.pop(ping_id, None)

    tokens: list[OperatorPingToken] = []
    for event in sorted(unresolved.values(), key=_operator_ping_sort_key):
        ping = event["operator_ping"]
        tokens.append(
            OperatorPingToken(
                ping_id=str(ping["ping_id"]),
                source_event_id=str(event.get("event_id") or ""),
                source_signature=_operator_ping_source_signature(ping),
                label=_operator_ping_token_label(ping),
                diaulos=str(ping.get("diaulos") or ""),
                topos=str(ping.get("topos") or ""),
                thread_id=str(ping.get("thread_id") or ""),
                pane_id=str(ping.get("pane_id") or ""),
                session_address=str(ping.get("session_address") or ""),
                message=str(ping.get("message") or ""),
                reason=str(ping.get("reason") or ""),
                reason_token=str(ping.get("reason_token") or ""),
                refs={
                    str(key): [str(item) for item in value if item]
                    for key, value in (event.get("refs") or {}).items()
                    if isinstance(value, list)
                },
            )
        )
    return tokens


def layout_operator_ping_token_visuals(
    tokens: list[OperatorPingToken],
    *,
    stack_body_frame: tuple[float, float, float, float],
    stack: CoordinationStack | None = None,
) -> list[OperatorPingTokenVisual]:
    """Lay out operator-ping tokens near the Stack body without row mutation.

    The layout deliberately does not cap tokens. If a caller sees too many
    visuals, that is upstream salience pressure to diagnose, not data to drop.
    """
    del stack

    stack_x, stack_y, stack_width, stack_height = stack_body_frame
    base_x = stack_x + max(0.0, stack_width - OPERATOR_PING_TOKEN_WIDTH)
    base_y = stack_y + stack_height + OPERATOR_PING_TOKEN_GAP
    count = len(tokens)
    visuals: list[OperatorPingTokenVisual] = []
    for index, token in enumerate(tokens):
        frame = TokenVisualFrame(
            x=base_x + (index % 3) * OPERATOR_PING_TOKEN_CASCADE_X,
            y=base_y + index * OPERATOR_PING_TOKEN_CASCADE_Y,
            width=OPERATOR_PING_TOKEN_WIDTH,
            height=OPERATOR_PING_TOKEN_HEIGHT,
        )
        visuals.append(
            OperatorPingTokenVisual(
                ping_id=token.ping_id,
                visual_index=index,
                diagnostic_count=count,
                frame=frame,
                presentation_text=_operator_ping_visual_text(token),
                accessibility_label=_operator_ping_visual_accessibility_label(token),
            )
        )
    return visuals
