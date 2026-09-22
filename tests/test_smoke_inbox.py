import pytest

from spoke.smoke_inbox import _return_affordance


@pytest.mark.parametrize(("delivery", "title", "enabled", "message"), [
    (None, "Return Saved Reply", True, "not yet returned"),
    ({"state": "sending"}, "Retry Return", True, "interrupted"),
    ({"state": "unconfirmed"}, "Retry Return", True, "unconfirmed"),
    ({"state": "delivered"}, "Send Reply", False, "handed to agent"),
])
def test_saved_response_always_exposes_truthful_return_state(delivery, title, enabled, message):
    row = {"status": "responded", "response": {"text": "Keep B"}, "delivery": delivery}
    actual_title, actual_enabled, actual_message = _return_affordance(row)
    assert (actual_title, actual_enabled) == (title, enabled)
    assert message in actual_message
