"""Tests for typed coordination surface stack.

The stack replaces the raw-text tray with typed surfaces that the operator
rocks through using shift+space. Each surface type has its own action
vocabulary and compact/expanded renderers.
"""

import threading

from spoke.coordination_surfaces import (
    CoordinationStack,
    SurfaceAction,
    SurfaceEntry,
    SurfaceIdentity,
    SurfaceKind,
    SurfaceMessage,
    SurfaceMessageBus,
    SurfaceRenderer,
    SurfaceTypeRegistration,
    SurfaceTypeRegistry,
    build_default_registry,
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
        mid = _text_entry("mid")
        mid.priority = 5

        stack.push(low)
        stack.push(high)
        stack.push_by_priority(mid)
        # Order should be: high(1), mid(5), low(10)
        assert stack.entries[0].surface_id == "high"
        assert stack.entries[1].priority == 5
        assert stack.entries[2].surface_id == "low"

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
            SurfaceAction(name="accept", phrases=["accept", "ok"], description="Accept the finding"),
            SurfaceAction(name="defer", phrases=["defer"], description="Defer for later"),
        ]
        intents = surface_actions_to_resolver_intents(actions)
        assert len(intents) == 2
        assert intents[0]["id"] == "accept"
        assert intents[0]["description"] == "Accept the finding"
        assert intents[0]["examples"] == ("accept", "ok")
        assert intents[1]["id"] == "defer"

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
