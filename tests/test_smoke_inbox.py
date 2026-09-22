import pytest

from spoke.smoke_inbox import _request_list_text, _return_affordance


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


def test_request_list_keeps_long_titles_in_details_instead_of_clipping_them():
    row = {
        "status": "pending",
        "request": {
            "source": {"diaulos": "greenroom-floor-manager"},
            "title": "Comparator framing check with a deliberately long title",
        },
    }
    assert _request_list_text(row) == "greenroom-floor-manager\nWaiting"
