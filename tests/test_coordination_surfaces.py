"""Tests for typed coordination surface stack.

The stack replaces the raw-text tray with typed surfaces that the operator
rocks through using shift+space. Each surface type has its own action
vocabulary and compact/expanded renderers.
"""

import threading
import json

from spoke.coordination_surfaces import (
    CoordinationStack,
    StackOrderingMode,
    SurfaceAction,
    SurfaceDestinationKind,
    SurfaceEntry,
    SurfaceIdentity,
    SurfaceKind,
    SurfaceMessage,
    SurfaceMessageBus,
    SurfaceRenderer,
    SurfaceRoutingContext,
    SurfaceTypeRegistration,
    SurfaceTypeRegistry,
    build_default_registry,
    diaulos_surface_from_record,
    derive_operator_ping_tokens,
    layout_operator_ping_token_visuals,
    load_operator_ping_events_from_jsonl,
    surface_actions_to_resolver_intents,
    text_surface_from_str,
)


def _agent_entry(session_id: str = "sess-1", label: str = "Lane A") -> SurfaceEntry:
    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.AGENT_THREAD,
            surface_id=session_id,
            label=label,
        ),
        payload={"bearing": "Working on tests", "readiness": "working"},
    )


def _finding_entry(path: str = "findings/f1.md", label: str = "Finding 1") -> SurfaceEntry:
    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.FINDING,
            surface_id=path,
            label=label,
        ),
        payload={"severity": "material", "summary": "Test finding"},
    )


def _text_entry(text: str = "hello world") -> SurfaceEntry:
    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.TEXT,
            surface_id=f"text-{id(text)}",
            label=text[:40],
        ),
        payload={"text": text},
    )


def _text_entry_with_id(surface_id: str, text: str) -> SurfaceEntry:
    return SurfaceEntry(
        identity=SurfaceIdentity(
            kind=SurfaceKind.TEXT,
            surface_id=surface_id,
            label=text[:40],
        ),
        payload={"text": text},
    )


class TestSurfaceIdentity:
    def test_identity_fields(self):
        ident = SurfaceIdentity(
            kind=SurfaceKind.AGENT_THREAD,
            surface_id="codex-123",
            label="Codex lane",
        )
        assert ident.kind == SurfaceKind.AGENT_THREAD
        assert ident.surface_id == "codex-123"
        assert ident.label == "Codex lane"

    def test_label_defaults_empty(self):
        ident = SurfaceIdentity(kind=SurfaceKind.FINDING, surface_id="f/1")
        assert ident.label == ""


class TestSurfaceEntry:
    def test_kind_shortcut(self):
        e = _agent_entry()
        assert e.kind == SurfaceKind.AGENT_THREAD

    def test_surface_id_shortcut(self):
        e = _agent_entry("sess-99")
        assert e.surface_id == "sess-99"

    def test_label_falls_back_to_surface_id(self):
        e = SurfaceEntry(
            identity=SurfaceIdentity(kind=SurfaceKind.METADOSIS, surface_id="meta/1")
        )
        assert e.label == "meta/1"

    def test_payload_is_kind_specific(self):
        e = _agent_entry()
        assert e.payload["bearing"] == "Working on tests"

    def test_entry_can_carry_interlocutor_routing_context(self):
        e = SurfaceEntry(
            identity=SurfaceIdentity(
                kind=SurfaceKind.METADOSIS,
                surface_id="metadosis/example.md",
                label="Example metadosis",
            ),
            routing=SurfaceRoutingContext(
                destination_kind=SurfaceDestinationKind.DIAULOS,
                destination_id="codex-stack-diaulos-border-stamp-0519",
                reread_refs=["metadosis/example.md"],
                scope={"metadosis": ["metadosis/example.md"]},
                cargo={"selected_text": "operator focus"},
                writeback_target="metadosis/example.md",
            ),
        )

        assert e.routing is not None
        assert e.routing.destination_kind == SurfaceDestinationKind.DIAULOS
        assert e.routing.destination_id == "codex-stack-diaulos-border-stamp-0519"
        assert e.routing.reread_refs == ["metadosis/example.md"]
        assert e.routing.scope == {"metadosis": ["metadosis/example.md"]}
        assert e.routing.cargo == {"selected_text": "operator focus"}
        assert e.routing.writeback_target == "metadosis/example.md"


class TestSurfaceTypeRegistry:
    def test_register_and_retrieve(self):
        reg = SurfaceTypeRegistry()
        actions = [SurfaceAction(name="start", phrases=["start this", "go"])]
        registration = SurfaceTypeRegistration(
            kind=SurfaceKind.AGENT_THREAD, actions=actions
        )
        reg.register(registration)
        assert reg.get(SurfaceKind.AGENT_THREAD) is registration

    def test_actions_for_registered_type(self):
        reg = SurfaceTypeRegistry()
        actions = [
            SurfaceAction(name="accept", phrases=["accept"]),
            SurfaceAction(name="defer", phrases=["defer", "later"]),
        ]
        reg.register(SurfaceTypeRegistration(kind=SurfaceKind.FINDING, actions=actions))
        assert len(reg.actions_for(SurfaceKind.FINDING)) == 2
        assert reg.actions_for(SurfaceKind.FINDING)[0].name == "accept"

    def test_actions_for_unknown_type_returns_empty(self):
        reg = SurfaceTypeRegistry()
        assert reg.actions_for(SurfaceKind.PERCEPTASIA_VIEW) == []

    def test_registered_kinds(self):
        reg = SurfaceTypeRegistry()
        reg.register(SurfaceTypeRegistration(kind=SurfaceKind.AGENT_THREAD))
        reg.register(SurfaceTypeRegistration(kind=SurfaceKind.FINDING))
        assert set(reg.registered_kinds) == {
            SurfaceKind.AGENT_THREAD,
            SurfaceKind.FINDING,
        }


class TestCoordinationStack:
    def test_empty_stack(self):
        stack = CoordinationStack()
        assert stack.size == 0
        assert stack.primary is None
        assert stack.active is False

    def test_push_to_top(self):
        stack = CoordinationStack()
        e1 = _agent_entry("s1")
        e2 = _finding_entry("f1")
        stack.push(e1)
        stack.push(e2)
        # e2 pushed to top, so it's at index 0
        assert stack.entries[0] is e2
        assert stack.entries[1] is e1

    def test_push_to_bottom(self):
        stack = CoordinationStack()
        e1 = _agent_entry("s1")
        e2 = _finding_entry("f1")
        stack.push(e1)
        stack.push(e2, to_top=False)
        assert stack.entries[0] is e1
        assert stack.entries[1] is e2

    def test_push_by_priority(self):
        stack = CoordinationStack()
        low = _agent_entry("low")
        low.priority = 10
        high = _finding_entry("high")
        high.priority = 1
        mid = _text_entry_with_id("mid", "mid")
        mid.priority = 5

        stack.push(low)
        stack.push(high)
        stack.push_by_priority(mid)
        # Order should be: high(1), mid(5), low(10)
        assert stack.entries[0].surface_id == "high"
        assert stack.entries[1].priority == 5
        assert stack.entries[2].surface_id == "low"
        assert stack.ordering_mode == StackOrderingMode.PRIORITY

    def test_push_by_priority_sorts_mixed_arrival_stack_globally(self):
        stack = CoordinationStack()
        low = _agent_entry("low")
        low.priority = 10
        high = _finding_entry("high")
        high.priority = 1
        mid = _text_entry_with_id("mid", "mid")
        mid.priority = 5

        stack.push(low, to_top=False)
        stack.push(high, to_top=False)
        stack.push_by_priority(mid)

        assert [(e.surface_id, e.priority) for e in stack.entries] == [
            ("high", 1),
            ("mid", 5),
            ("low", 10),
        ]

    def test_priority_reorder_preserves_active_pivot_surface(self):
        stack = CoordinationStack()
        low = _agent_entry("low", "Low")
        low.priority = 10
        high = _finding_entry("high", "High")
        high.priority = 1
        mid = _text_entry_with_id("mid", "mid")
        mid.priority = 5

        stack.push(low, to_top=False)
        stack.push(high, to_top=False)
        stack.activate()
        stack.rock_down()
        assert stack.primary is high

        stack.push_by_priority(mid)

        assert [(e.surface_id, e.priority) for e in stack.entries] == [
            ("high", 1),
            ("mid", 5),
            ("low", 10),
        ]
        assert stack.primary is high

    def test_push_by_priority_inactive_resets_index(self):
        """When inactive, push_by_priority should reset index to 0."""
        stack = CoordinationStack()
        e1 = _agent_entry("s1")
        e1.priority = 5
        stack.push(e1)
        # Not activated — index should be 0 after priority push
        e2 = _finding_entry("f1")
        e2.priority = 1
        stack.push_by_priority(e2)
        assert stack.index == 0

    def test_activate_and_deactivate(self):
        stack = CoordinationStack()
        stack.push(_agent_entry())
        assert stack.activate() is not None
        assert stack.active is True
        stack.deactivate()
        assert stack.active is False

    def test_activate_empty_returns_none(self):
        stack = CoordinationStack()
        assert stack.activate() is None
        assert stack.active is False

    def test_rock_up_and_down(self):
        stack = CoordinationStack()
        e1 = _agent_entry("s1", "First")
        e2 = _agent_entry("s2", "Second")
        e3 = _agent_entry("s3", "Third")
        stack.push(e1, to_top=False)
        stack.push(e2, to_top=False)
        stack.push(e3, to_top=False)
        stack.activate()
        # Start at index 0
        assert stack.primary is e1
        # Rock down
        assert stack.rock_down() is e2
        assert stack.rock_down() is e3
        # At end, stays there
        assert stack.rock_down() is e3
        # Rock up
        assert stack.rock_up() is e2
        assert stack.rock_up() is e1
        # At start, stays there
        assert stack.rock_up() is e1

    def test_rock_wrap_up(self):
        stack = CoordinationStack()
        e1 = _agent_entry("s1")
        e2 = _agent_entry("s2")
        stack.push(e1, to_top=False)
        stack.push(e2, to_top=False)
        stack.activate()
        # At index 0, wrap up goes to end
        assert stack.rock_wrap_up() is e2

    def test_rock_inactive_returns_none(self):
        stack = CoordinationStack()
        stack.push(_agent_entry())
        assert stack.rock_up() is None
        assert stack.rock_down() is None

    def test_remove_current(self):
        stack = CoordinationStack()
        e1 = _agent_entry("s1")
        e2 = _agent_entry("s2")
        stack.push(e1, to_top=False)
        stack.push(e2, to_top=False)
        stack.activate()
        removed = stack.remove_current()
        assert removed is e1
        assert stack.size == 1
        assert stack.primary is e2

    def test_remove_current_deactivates_when_empty(self):
        stack = CoordinationStack()
        stack.push(_agent_entry())
        stack.activate()
        stack.remove_current()
        assert stack.active is False
        assert stack.size == 0

    def test_remove_by_id(self):
        stack = CoordinationStack()
        e1 = _agent_entry("s1")
        e2 = _agent_entry("s2")
        stack.push(e1, to_top=False)
        stack.push(e2, to_top=False)
        stack.activate()
        stack.rock_down()  # primary is e2
        removed = stack.remove_by_id("s1")
        assert removed is e1
        assert stack.size == 1
        # index adjusted since removed was before current
        assert stack.primary is e2

    def test_remove_by_id_not_found(self):
        stack = CoordinationStack()
        stack.push(_agent_entry("s1"))
        assert stack.remove_by_id("nonexistent") is None

    def test_find_by_id(self):
        stack = CoordinationStack()
        e = _agent_entry("target")
        stack.push(e)
        assert stack.find_by_id("target") is e
        assert stack.find_by_id("nope") is None

    def test_find_by_kind(self):
        stack = CoordinationStack()
        a = _agent_entry("a1")
        f = _finding_entry("f1")
        stack.push(a)
        stack.push(f)
        agents = stack.find_by_kind(SurfaceKind.AGENT_THREAD)
        assert len(agents) == 1
        assert agents[0] is a

    def test_action_vocabulary_from_registry(self):
        reg = SurfaceTypeRegistry()
        actions = [SurfaceAction(name="dismiss", phrases=["dismiss", "close"])]
        reg.register(
            SurfaceTypeRegistration(kind=SurfaceKind.ZETESIS_RESULT, actions=actions)
        )
        stack = CoordinationStack(registry=reg)
        stack.push(
            SurfaceEntry(
                identity=SurfaceIdentity(
                    kind=SurfaceKind.ZETESIS_RESULT,
                    surface_id="q1",
                    label="Query result",
                )
            )
        )
        stack.activate()
        vocab = stack.action_vocabulary()
        assert len(vocab) == 1
        assert vocab[0].name == "dismiss"

    def test_action_vocabulary_empty_when_inactive(self):
        """Active guard: vocabulary must be empty when stack is not active,
        even if entries exist and a registry has actions for their kind."""
        reg = build_default_registry()
        stack = CoordinationStack(registry=reg)
        stack.push(_agent_entry())
        # Not activated — should return empty despite registered actions
        assert stack.action_vocabulary() == []
        # Activate — now actions should appear
        stack.activate()
        assert len(stack.action_vocabulary()) > 0
        # Deactivate — empty again
        stack.deactivate()
        assert stack.action_vocabulary() == []


class TestLegacyBridge:
    def test_text_surface_from_str(self):
        entry = text_surface_from_str("hello world")
        assert entry.kind == SurfaceKind.TEXT
        assert entry.payload["text"] == "hello world"
        assert entry.payload["owner"] == "user"
        assert entry.acknowledged is True
        assert entry.label == "hello world"
        assert entry.routing is None

    def test_text_surface_label_is_compact_one_line(self):
        entry = text_surface_from_str("first line\nsecond line\tand more")
        stack = CoordinationStack()
        stack.push(entry)

        assert "\n" not in entry.label
        assert "\t" not in entry.label
        assert stack.compact_summary(entry) == "first line second line and more"

    def test_text_surface_from_str_assistant(self):
        entry = text_surface_from_str("response", owner="assistant")
        assert entry.payload["owner"] == "assistant"
        assert entry.acknowledged is False

    def test_text_surface_unique_ids(self):
        e1 = text_surface_from_str("a")
        e2 = text_surface_from_str("a")
        assert e1.surface_id != e2.surface_id

    def test_text_surface_in_stack(self):
        stack = CoordinationStack()
        stack.push(text_surface_from_str("first"))
        stack.push(text_surface_from_str("second"))
        stack.activate()
        assert stack.primary.payload["text"] == "second"
        stack.rock_down()
        assert stack.primary.payload["text"] == "first"


class TestVoiceActionRouting:
    def test_action_vocabulary_switches_with_primary(self):
        """When the primary surface changes, the action vocabulary changes."""
        reg = SurfaceTypeRegistry()
        reg.register(SurfaceTypeRegistration(
            kind=SurfaceKind.AGENT_THREAD,
            actions=[
                SurfaceAction(name="start", phrases=["start this", "go"]),
                SurfaceAction(name="cancel", phrases=["cancel", "stop"]),
            ],
        ))
        reg.register(SurfaceTypeRegistration(
            kind=SurfaceKind.FINDING,
            actions=[
                SurfaceAction(name="accept", phrases=["accept"]),
                SurfaceAction(name="defer", phrases=["defer", "later"]),
                SurfaceAction(name="navigate", phrases=["show me", "go to commit"]),
            ],
        ))
        stack = CoordinationStack(registry=reg)
        stack.push(_agent_entry("s1"), to_top=False)
        stack.push(_finding_entry("f1"), to_top=False)
        stack.activate()

        # Primary is agent thread (index 0)
        vocab = stack.action_vocabulary()
        assert len(vocab) == 2
        assert {a.name for a in vocab} == {"start", "cancel"}

        # Rock down to finding
        stack.rock_down()
        vocab = stack.action_vocabulary()
        assert len(vocab) == 3
        assert {a.name for a in vocab} == {"accept", "defer", "navigate"}

    def test_surface_actions_to_resolver_intents(self):
        actions = [
            SurfaceAction(
                name="accept",
                phrases=["accept", "ok"],
                description="Accept the finding",
                interlocutor_act="route_finding_disposition",
                requires_interlocutor=True,
                source_owned=True,
                writeback_allowed=True,
            ),
            SurfaceAction(name="defer", phrases=["defer"], description="Defer for later"),
        ]
        intents = surface_actions_to_resolver_intents(actions)
        assert len(intents) == 2
        assert intents[0]["id"] == "accept"
        assert intents[0]["description"] == "Accept the finding"
        assert intents[0]["examples"] == ("accept", "ok")
        assert intents[0]["interlocutor_act"] == "route_finding_disposition"
        assert intents[0]["requires_interlocutor"] is True
        assert intents[0]["source_owned"] is True
        assert intents[0]["writeback_allowed"] is True
        assert intents[1]["id"] == "defer"
        assert intents[1]["requires_interlocutor"] is False

    def test_resolver_intents_without_description_uses_name(self):
        actions = [SurfaceAction(name="dismiss", phrases=["dismiss"])]
        intents = surface_actions_to_resolver_intents(actions)
        assert intents[0]["description"] == "dismiss"

    def test_default_registry_all_kinds_have_actions(self):
        """Every surface kind in the default registry has at least one action."""
        reg = build_default_registry()
        for kind in SurfaceKind:
            actions = reg.actions_for(kind)
            assert len(actions) >= 1, f"{kind} has no actions in default registry"

    def test_default_registry_all_kinds_have_dismiss(self):
        """Every surface kind should support dismiss as a universal action."""
        reg = build_default_registry()
        for kind in SurfaceKind:
            actions = reg.actions_for(kind)
            names = {a.name for a in actions}
            assert "dismiss" in names, f"{kind} missing dismiss action"

    def test_default_registry_voice_routing_end_to_end(self):
        """Simulate: push agent thread, activate, get vocabulary, convert to intents."""
        reg = build_default_registry()
        stack = CoordinationStack(registry=reg)
        stack.push(_agent_entry("codex-1"))
        stack.activate()

        vocab = stack.action_vocabulary()
        intents = surface_actions_to_resolver_intents(vocab)

        # Should have agent thread actions
        intent_ids = {i["id"] for i in intents}
        assert "start" in intent_ids
        assert "cancel" in intent_ids
        assert "dismiss" in intent_ids
        # Each intent should have examples for the resolver
        for intent in intents:
            assert len(intent["examples"]) >= 1

    def test_default_registry_durable_actions_route_through_source_owned_interlocutors(self):
        reg = build_default_registry()

        update = {a.name: a for a in reg.actions_for(SurfaceKind.METADOSIS)}["update"]
        assert update.interlocutor_act == "route_update_to_custodian"
        assert update.requires_interlocutor is True
        assert update.source_owned is True
        assert update.writeback_allowed is True

        accept = {a.name: a for a in reg.actions_for(SurfaceKind.FINDING)}["accept"]
        assert accept.interlocutor_act == "route_finding_disposition"
        assert accept.requires_interlocutor is True
        assert accept.source_owned is True
        assert accept.writeback_allowed is True

        confirm = {a.name: a for a in reg.actions_for(SurfaceKind.METAMORPHOSIS_RESULT)}["confirm"]
        assert confirm.interlocutor_act == "route_mutation_confirmation"
        assert confirm.requires_interlocutor is True
        assert confirm.source_owned is True
        assert confirm.writeback_allowed is True

    def test_stack_local_dismiss_remains_non_writeback(self):
        reg = build_default_registry()
        for kind in SurfaceKind:
            dismiss = {a.name: a for a in reg.actions_for(kind)}["dismiss"]
            assert dismiss.requires_interlocutor is False
            assert dismiss.source_owned is False
            assert dismiss.writeback_allowed is False


class TestDiaulosCardSurface:
    def test_diaulos_record_projects_to_read_only_stack_card(self):
        entry = diaulos_surface_from_record(
            {
                "diaulos": "chairside-sparkwright",
                "diaulos_id": "dia-chair-1",
                "display_name": "Chairside Sparkwright",
                "topos": "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md",
                "status": "Κίνησις",
                "summary": "Read-only card slice in progress.",
                "refs": {
                    "topoi": [
                        "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md"
                    ],
                    "metadosis": [
                        "metadosis/source-signed-diaulos-switchboard_2026-05-20.md"
                    ],
                },
            }
        )

        assert entry.kind == SurfaceKind.DIAULOS
        assert entry.surface_id == "diaulos:dia-chair-1"
        assert entry.label == "Chairside Sparkwright"
        assert entry.acknowledged is True
        assert entry.routing is not None
        assert entry.routing.destination_kind == SurfaceDestinationKind.DIAULOS
        assert entry.routing.destination_id == "dia-chair-1"
        assert entry.routing.writeback_target == ""
        assert entry.routing.reread_refs == [
            "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md"
        ]
        assert entry.routing.cargo["authority"] == "read_only_identity_fact"
        assert entry.routing.cargo["may_focus_pane"] is False
        assert entry.routing.cargo["may_write_state"] is False
        assert entry.routing.cargo["may_send_directive"] is False
        assert entry.payload["diaulos"] == "chairside-sparkwright"
        assert entry.payload["diaulos_id"] == "dia-chair-1"
        assert entry.payload["summary"] == "Read-only card slice in progress."

    def test_diaulos_card_uses_handle_as_stable_fallback_without_topos_shadow(self):
        entry = diaulos_surface_from_record(
            {
                "diaulos": "kynormous-bastards",
                "topos": "projects/operator_memory/topoi/codex-kynormous-weight-of-staggering-kinesthesia-0514.md",
            }
        )

        assert entry.surface_id == "diaulos:kynormous-bastards"
        assert entry.label == "kynormous-bastards"
        assert entry.routing is not None
        assert entry.routing.destination_id == "kynormous-bastards"
        assert "codex-kynormous-weight" not in entry.surface_id

    def test_diaulos_card_renderer_keeps_compact_and_expanded_text_boring(self):
        reg = build_default_registry()
        stack = CoordinationStack(registry=reg)
        entry = diaulos_surface_from_record(
            {
                "diaulos": "opus-miserena-id-cartographer",
                "diaulos_id": "dia-b715a7f9-ec67-4dcd-80a3-12688844f177",
                "display_name": "Opus Miserena",
                "topos": "projects/operator_memory/topoi/codex-opus-miserena-id-cartographer-0521.md",
                "source_topoi": [
                    "projects/operator_memory/topoi/codex-opus-miserena-id-cartographer-0521.md"
                ],
                "status": "Κίνησις",
                "summary": "Diaulos ID coherence and switchboard routing custody.",
            }
        )
        stack.push(entry)

        compact = stack.compact_summary(entry)
        expanded = stack.expanded_view(entry)

        assert compact == "Diaulos: Opus Miserena · Κίνησις"
        assert "Diaulos: Opus Miserena · Κίνησις" in expanded
        assert "Handle: opus-miserena-id-cartographer" in expanded
        assert "ID: dia-b715a7f9-ec67-4dcd-80a3-12688844f177" in expanded
        assert "Registry source: codex-opus-miserena-id-cartographer-0521.md" in expanded
        assert "projects/operator_memory/topoi/" not in expanded
        assert "Read-only card" in expanded
        assert "Send directive" not in expanded
        assert "Focus pane" not in expanded

    def test_diaulos_card_preserves_registry_source_topoi_and_current_custody_refs(self):
        entry = diaulos_surface_from_record(
            {
                "diaulos": "chairside-sparkwright",
                "diaulos_id": "dia-f054023f-d93b-485c-af0c-942698434d11",
                "display_name": "Chairside Sparkwright",
                "topos": "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md",
                "source_topoi": [
                    "projects/spoke/operator_memory.md#codex-diaulos-stack-current-main-graft-0523"
                ],
                "custody_refs": [
                    "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md"
                ],
                "warnings": ["current_topos_not_registry_source_topos"],
            }
        )

        assert entry.surface_id == "diaulos:dia-f054023f-d93b-485c-af0c-942698434d11"
        assert entry.routing is not None
        assert entry.routing.reread_refs == [
            "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md",
            "projects/spoke/operator_memory.md#codex-diaulos-stack-current-main-graft-0523",
        ]
        assert entry.payload["source_topoi"] == [
            "projects/spoke/operator_memory.md#codex-diaulos-stack-current-main-graft-0523"
        ]
        assert entry.payload["custody_refs"] == [
            "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md"
        ]
        assert entry.payload["warnings"] == ["current_topos_not_registry_source_topos"]

    def test_diaulos_card_renderer_keeps_registry_backed_smoke_card_overlay_sized(self):
        reg = build_default_registry()
        stack = CoordinationStack(registry=reg)
        entry = diaulos_surface_from_record(
            {
                "diaulos": "chairside-sparkwright",
                "diaulos_id": "dia-f054023f-d93b-485c-af0c-942698434d11",
                "display_name": "Chairside Sparkwright",
                "topos": "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md",
                "source_topoi": [
                    "projects/spoke/operator_memory.md#codex-diaulos-stack-current-main-graft-0523"
                ],
                "custody_refs": [
                    "projects/spoke/topoi/codex-diaulos-card-carrying-bastards-0524.md"
                ],
                "warnings": ["current_topos_not_registry_source_topos"],
                "status": "Κίνησις",
                "summary": "Read-only Diaulos card smoke; no authority routing or writeback.",
            }
        )

        expanded = stack.expanded_view(entry)

        assert len(expanded.splitlines()) <= 6
        assert "Custody: codex-diaulos-card-carrying-bastards-0524.md" in expanded
        assert "Registry source: codex-diaulos-stack-current-main-graft-0523" in expanded
        assert "current_topos_not_registry_source_topos" not in expanded

    def test_default_diaulos_actions_do_not_claim_write_authority(self):
        reg = build_default_registry()
        actions = reg.actions_for(SurfaceKind.DIAULOS)

        assert [action.name for action in actions] == ["dismiss"]
        dismiss = actions[0]
        assert dismiss.requires_interlocutor is False
        assert dismiss.source_owned is False
        assert dismiss.writeback_allowed is False

    def test_operator_ping_events_do_not_project_to_diaulos_cards(self):
        stack = CoordinationStack(registry=build_default_registry())
        ping = {
            "kind": "operator_ping.created",
            "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-1",
            "operator_ping": {
                "ping_id": "ping-1",
                "created_at": "2026-05-24T14:00:00Z",
                "diaulos": "chairside-sparkwright",
                "reason_token": "pingy",
            },
        }

        tokens = derive_operator_ping_tokens([ping], stack=stack)

        assert [token.ping_id for token in tokens] == ["ping-1"]
        assert stack.find_by_kind(SurfaceKind.DIAULOS) == []


class TestRendererIntegration:
    def test_compact_and_expanded_with_renderer(self):
        class FakeRenderer:
            def compact(self, entry: SurfaceEntry) -> str:
                return f"[{entry.kind.value}] {entry.label}"

            def expanded(self, entry: SurfaceEntry) -> str:
                bearing = entry.payload.get("bearing", "")
                return f"{entry.label}\n  Bearing: {bearing}"

        reg = SurfaceTypeRegistry()
        reg.register(
            SurfaceTypeRegistration(
                kind=SurfaceKind.AGENT_THREAD, renderer=FakeRenderer()
            )
        )
        stack = CoordinationStack(registry=reg)
        e = _agent_entry("s1", "My Lane")
        stack.push(e)

        assert stack.compact_summary(e) == "[agent_thread] My Lane"
        assert "Bearing: Working on tests" in stack.expanded_view(e)

    def test_fallback_without_renderer(self):
        stack = CoordinationStack()
        e = _agent_entry("s1", "Fallback Lane")
        stack.push(e)
        assert stack.compact_summary(e) == "Fallback Lane"
        assert stack.expanded_view(e) == "Fallback Lane"


class TestSurfaceMessageBus:
    def test_post_and_drain(self):
        stack = CoordinationStack()
        bus = SurfaceMessageBus(stack)
        entry = _agent_entry("s1")
        bus.post(SurfaceMessage(entry=entry, source="test"))
        assert bus.pending_count == 1
        delivered = bus.drain()
        assert len(delivered) == 1
        assert delivered[0] is entry
        assert stack.size == 1
        assert bus.pending_count == 0

    def test_drain_empty(self):
        stack = CoordinationStack()
        bus = SurfaceMessageBus(stack)
        assert bus.drain() == []

    def test_activate_on_delivery(self):
        stack = CoordinationStack()
        bus = SurfaceMessageBus(stack)
        bus.post(SurfaceMessage(entry=_finding_entry("f1"), activate=True))
        bus.drain()
        assert stack.active is True

    def test_priority_insertion(self):
        stack = CoordinationStack()
        bus = SurfaceMessageBus(stack)
        # Pre-populate stack
        low = _agent_entry("low")
        low.priority = 10
        stack.push(low)

        high = _finding_entry("high")
        high.priority = 1
        bus.post(SurfaceMessage(entry=high, position="priority"))
        bus.drain()

        # High priority should be before low
        assert stack.entries[0].surface_id == "high"
        assert stack.entries[1].surface_id == "low"

    def test_on_delivery_callback(self):
        stack = CoordinationStack()
        delivered_entries = []
        bus = SurfaceMessageBus(stack, on_delivery=delivered_entries.append)
        entry = _agent_entry("s1")
        bus.post(SurfaceMessage(entry=entry))
        bus.drain()
        assert delivered_entries == [entry]

    def test_thread_safety(self):
        """Multiple threads posting concurrently should not lose messages."""
        stack = CoordinationStack()
        bus = SurfaceMessageBus(stack)
        n_threads = 10
        n_per_thread = 50

        def _poster(thread_id):
            for i in range(n_per_thread):
                entry = _agent_entry(f"t{thread_id}-{i}")
                bus.post(SurfaceMessage(entry=entry, source=f"thread-{thread_id}"))

        threads = [threading.Thread(target=_poster, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bus.pending_count == n_threads * n_per_thread
        delivered = bus.drain()
        assert len(delivered) == n_threads * n_per_thread
        assert stack.size == n_threads * n_per_thread


class TestOperatorPingTokenProjection:
    def test_operator_ping_event_log_reader_replays_jsonl_without_capping(self, tmp_path):
        event_log = tmp_path / "events.jsonl"
        events = [
            {
                "kind": "operator_ping.created",
                "event_id": f"operator_memory.event.v1:operator_ping.created:spoke:ping-{index}",
                "operator_ping": {
                    "ping_id": f"ping-{index}",
                    "created_at": f"2026-05-22T12:{index:02d}:00Z",
                    "diaulos": "chairside-sparkwright",
                    "reason_token": f"q{index}",
                },
            }
            for index in range(24)
        ]
        event_log.write_text(
            "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n\n",
            encoding="utf-8",
        )

        loaded = load_operator_ping_events_from_jsonl(event_log)

        assert [event["operator_ping"]["ping_id"] for event in loaded] == [
            f"ping-{index}" for index in range(24)
        ]

    def test_operator_ping_event_log_reader_missing_file_is_quiet(self, tmp_path):
        assert load_operator_ping_events_from_jsonl(tmp_path / "missing.jsonl") == []

    def test_unresolved_operator_pings_project_to_ephemeral_source_tokens(self):
        events = [
            {
                "kind": "operator_ping.created",
                "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-1",
                "source_tool": "operator_memory ping-operator",
                "operator_ping": {
                    "ping_id": "ping-1",
                    "created_at": "2026-05-22T12:00:00Z",
                    "diaulos": "chairside-sparkwright",
                    "topos": "projects/spoke/operator_memory.md#codex-source-sparks-near-the-chair-0522",
                    "thread_id": "019e50eb-1a5f-7cc1-96e0-ea9adb751ef8",
                    "pane_id": "77",
                    "message": "assay needs operator glance",
                    "reason": "Need operator choice before seating the token surface.",
                    "reason_token": "question",
                },
            },
            {
                "kind": "operator_ping.created",
                "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-2",
                "source_tool": "operator_memory ping-operator",
                "operator_ping": {
                    "ping_id": "ping-2",
                    "created_at": "2026-05-22T12:01:00Z",
                    "topos": "projects/spoke/operator_memory.md#other-lane",
                    "message": "already handled",
                    "reason_token": "done",
                },
            },
            {
                "kind": "operator_ping.cleared",
                "event_id": "operator_memory.event.v1:operator_ping.cleared:spoke:ping-2",
                "operator_ping": {
                    "ping_id": "ping-2",
                    "cleared_at": "2026-05-22T12:02:00Z",
                },
            },
        ]

        tokens = derive_operator_ping_tokens(events)

        assert [token.ping_id for token in tokens] == ["ping-1"]
        token = tokens[0]
        assert token.anchor == "operator_stack_body"
        assert token.source_signature == "Diaulos: chairside-sparkwright"
        assert token.label == "question"
        assert token.reason_token == "question"
        assert token.source_event_id == "operator_memory.event.v1:operator_ping.created:spoke:ping-1"
        assert token.message == "assay needs operator glance"

    def test_operator_ping_tokens_do_not_enter_or_focus_durable_stack_rows(self):
        stack = CoordinationStack()
        primary = _agent_entry("active-lane", "Active lane")
        stack.push(primary)
        stack.activate()

        tokens = derive_operator_ping_tokens(
            [
                {
                    "kind": "operator_ping.created",
                    "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-1",
                    "operator_ping": {
                        "ping_id": "ping-1",
                        "created_at": "2026-05-22T12:00:00Z",
                        "diaulos": "chairside-sparkwright",
                        "reason_token": "question",
                    },
                }
            ],
            stack=stack,
        )

        assert len(tokens) == 1
        assert stack.entries == [primary]
        assert stack.primary is primary
        assert stack.active is True

    def test_operator_ping_token_activation_routes_to_source_without_write_authority(self):
        token = derive_operator_ping_tokens(
            [
                {
                    "kind": "operator_ping.created",
                    "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-1",
                    "refs": {
                        "topoi": ["projects/spoke/operator_memory.md#codex-source-sparks-near-the-chair-0522"],
                    },
                    "operator_ping": {
                        "ping_id": "ping-1",
                        "created_at": "2026-05-22T12:00:00Z",
                        "diaulos": "chairside-sparkwright",
                        "topos": "projects/spoke/operator_memory.md#codex-source-sparks-near-the-chair-0522",
                        "thread_id": "019e50eb-1a5f-7cc1-96e0-ea9adb751ef8",
                        "pane_id": "77",
                        "session_address": "codex resume 019e50eb-1a5f-7cc1-96e0-ea9adb751ef8",
                        "reason": "Need operator choice before seating the token surface.",
                        "reason_token": "question",
                    },
                }
            ]
        )[0]

        routing = token.activation_routing(gesture="select")

        assert routing.destination_kind == SurfaceDestinationKind.SOURCE_ORGAN
        assert routing.destination_id == "operator_ping:ping-1"
        assert routing.reread_refs == [
            "projects/spoke/operator_memory.md#codex-source-sparks-near-the-chair-0522"
        ]
        assert routing.scope == {
            "operator_pings": ["ping-1"],
            "topoi": ["projects/spoke/operator_memory.md#codex-source-sparks-near-the-chair-0522"],
        }
        assert routing.writeback_target == ""
        assert routing.cargo["gesture"] == "select"
        assert routing.cargo["reason_token"] == "question"
        assert routing.cargo["source_signature"] == "Diaulos: chairside-sparkwright"
        assert routing.cargo["authority"] == "event_fact_only"
        assert routing.cargo["may_clear_ping"] is False
        assert routing.cargo["may_focus_pane"] is False
        assert routing.cargo["may_write_state"] is False


class TestOperatorPingTokenVisualAssay:
    def test_token_visuals_render_as_quiet_source_sparks_near_stack_body(self):
        token = derive_operator_ping_tokens(
            [
                {
                    "kind": "operator_ping.created",
                    "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-1",
                    "operator_ping": {
                        "ping_id": "ping-1",
                        "created_at": "2026-05-22T12:00:00Z",
                        "diaulos": "chairside-sparkwright",
                        "message": "assay needs operator glance",
                        "reason_token": "question",
                    },
                }
            ]
        )[0]

        visuals = layout_operator_ping_token_visuals(
            [token],
            stack_body_frame=(100.0, 80.0, 360.0, 96.0),
        )

        assert len(visuals) == 1
        visual = visuals[0]
        assert visual.ping_id == "ping-1"
        assert visual.anchor == "operator_stack_body"
        assert visual.presentation_text == "Diaulos: chairside-sparkwright · question"
        assert visual.accessibility_label == (
            "Operator ping from Diaulos: chairside-sparkwright: question"
        )
        assert visual.style_role == "quiet_source_spark"
        assert visual.authority == "event_fact_only"
        assert visual.steals_primary_focus is False
        assert visual.frame.x >= 100.0
        assert visual.frame.y >= 176.0

    def test_token_visual_layout_keeps_all_tokens_without_capping(self):
        tokens = derive_operator_ping_tokens(
            [
                {
                    "kind": "operator_ping.created",
                    "event_id": f"operator_memory.event.v1:operator_ping.created:spoke:ping-{index}",
                    "operator_ping": {
                        "ping_id": f"ping-{index}",
                        "created_at": f"2026-05-22T12:{index:02d}:00Z",
                        "diaulos": "chairside-sparkwright",
                        "reason_token": f"q{index}",
                    },
                }
                for index in range(18)
            ]
        )

        visuals = layout_operator_ping_token_visuals(
            tokens,
            stack_body_frame=(100.0, 80.0, 360.0, 96.0),
        )

        assert len(visuals) == 18
        assert [visual.ping_id for visual in visuals] == [
            f"ping-{index}" for index in range(18)
        ]
        assert [visual.visual_index for visual in visuals] == list(range(18))
        assert all(visual.diagnostic_count == 18 for visual in visuals)
        assert len({(visual.frame.x, visual.frame.y) for visual in visuals}) == 18

    def test_token_visual_layout_does_not_mutate_stack_focus_or_rows(self):
        stack = CoordinationStack()
        primary = _agent_entry("active-lane", "Active lane")
        stack.push(primary)
        stack.activate()
        token = derive_operator_ping_tokens(
            [
                {
                    "kind": "operator_ping.created",
                    "event_id": "operator_memory.event.v1:operator_ping.created:spoke:ping-1",
                    "operator_ping": {
                        "ping_id": "ping-1",
                        "created_at": "2026-05-22T12:00:00Z",
                        "diaulos": "chairside-sparkwright",
                        "reason_token": "question",
                    },
                }
            ]
        )[0]

        visuals = layout_operator_ping_token_visuals(
            [token],
            stack_body_frame=(100.0, 80.0, 360.0, 96.0),
            stack=stack,
        )

        assert len(visuals) == 1
        assert stack.entries == [primary]
        assert stack.primary is primary
        assert stack.active is True
