"""Tests for the thinking narrator sidecar."""

from unittest.mock import patch
import urllib.error


def _connection_refused() -> urllib.error.URLError:
    return urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))


def test_dispatch_falls_back_to_raw_excerpt_when_narrator_backend_refuses():
    """When the narrator model is unreachable, the user should still see reasoning."""
    from spoke.narrator import ThinkingNarrator

    summaries: list[str] = []
    narrator = ThinkingNarrator(
        on_summary=summaries.append,
        base_url="http://localhost:9999",
        model="test-narrator",
        api_key="key",
    )
    narrator.start()

    with patch.object(narrator, "_chat_completion", side_effect=_connection_refused()):
        narrator._dispatch(
            "  The model is comparing Step and Gemini latency tradeoffs across retries.  "
        )

    assert summaries == [
        "The model is comparing Step and Gemini latency tradeoffs across retries."
    ]


def test_dispatch_failure_does_not_pollute_narrator_history():
    """Failed narrator calls should not leave dangling prompt turns behind."""
    from spoke.narrator import ThinkingNarrator, _SYSTEM_PROMPT

    narrator = ThinkingNarrator(
        on_summary=lambda _summary: None,
        base_url="http://localhost:9999",
        model="test-narrator",
        api_key="key",
    )
    narrator.start()

    with patch.object(narrator, "_chat_completion", side_effect=_connection_refused()):
        narrator._dispatch("Comparing retry budgets against latency spikes.")

    assert narrator._messages == [{"role": "system", "content": _SYSTEM_PROMPT}]
