from unittest.mock import MagicMock, patch

import pytest

import spoke.smoke_inbox as smoke_inbox_module
from spoke.smoke_inbox import SmokeInbox, _request_list_text, _return_affordance


@pytest.mark.parametrize(("delivery", "title", "enabled", "message"), [
    (None, "Return Reply", True, "not yet returned"),
    ({"state": "sending"}, "Retry Return", True, "interrupted"),
    ({"state": "unconfirmed"}, "Retry Return", True, "unconfirmed"),
    ({"state": "delivered"}, "Send Reply", False, "handed to agent"),
])
def test_saved_response_always_exposes_truthful_return_state(delivery, title, enabled, message):
    row = {"status": "responded", "response": {"text": "Keep B"}, "delivery": delivery}
    actual_title, actual_enabled, actual_message = _return_affordance(row)
    assert (actual_title, actual_enabled) == (title, enabled)
    assert message in actual_message


def test_withdrawn_saved_response_has_no_return_affordance():
    row = {
        "status": "withdrawn",
        "response": {"text": "Keep B"},
        "delivery": None,
    }
    title, enabled, message = _return_affordance(row)
    assert title == "Send Reply"
    assert enabled is False
    assert "withdrawn" in message.lower()


def test_open_smoke_requires_effective_prepared_state():
    row = {
        "status": "pending",
        "request": {
            "id": "cf282a16-cc01-4877-a572-e7f84c6da95c",
            "availability": "preparation-needed",
            "availability_note": "GPU access still needed.",
            "url": "http://127.0.0.1:8156/kiln",
        },
    }
    inbox = SmokeInbox.__new__(SmokeInbox)
    inbox._selected = lambda: row
    inbox._queue = MagicMock()
    inbox._queue.get.return_value = row
    inbox._status = MagicMock()
    with patch.object(smoke_inbox_module, "NSWorkspace") as workspace:
        inbox.openSmoke_(None)
    workspace.sharedWorkspace.assert_not_called()
    inbox._queue.act.assert_not_called()
    message = inbox._status.setStringValue_.call_args.args[0]
    assert "not prepared" in message.lower()


def test_needs_you_count_survives_saved_and_unconfirmed_return_restart():
    rows = [
        {"status": "pending", "delivery": None},
        {"status": "responded", "delivery": None},
        {"status": "responded", "delivery": {"state": "sending"}},
        {"status": "responded", "delivery": {"state": "unconfirmed"}},
        {"status": "responded", "delivery": {"state": "delivered"}},
        {"status": "withdrawn", "delivery": None},
    ]
    inbox = SmokeInbox.__new__(SmokeInbox)
    inbox._closed = False
    inbox._last_snapshot = None
    inbox._menu = MagicMock()
    inbox._panel = None
    inbox.snapshotChanged_({"rows": rows, "errors": []})
    inbox._menu.set_smoke_pending_count.assert_called_once_with(4)


def test_request_list_keeps_long_titles_in_details_instead_of_clipping_them():
    row = {
        "status": "pending",
        "request": {
            "source": {"diaulos": "greenroom-floor-manager"},
            "title": "Comparator framing check with a deliberately long title",
        },
    }
    assert _request_list_text(row) == "greenroom-floor-manager\nWaiting"


@pytest.mark.parametrize(("delivery", "status"), [
    (None, "Saved"),
    ({"state": "sending"}, "Returning"),
    ({"state": "unconfirmed"}, "Return unconfirmed"),
    ({"state": "delivered"}, "Returned"),
])
def test_request_list_does_not_call_a_saved_response_returned(delivery, status):
    row = {
        "status": "responded",
        "delivery": delivery,
        "request": {"source": {"diaulos": "greenroom-floor-manager"}},
    }
    assert _request_list_text(row) == f"greenroom-floor-manager\n{status}"
