import io
import json
from unittest.mock import MagicMock, patch

import pytest


def _json_response(payload):
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=io.BytesIO(body))
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestGmailOperator:
    def test_tool_schema_exposes_bounded_query_mode(self):
        from spoke.gmail_operator import tool_schema

        schema = tool_schema()
        params = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "query_gmail"
        assert params["mode"]["enum"] == ["starred_recruiter_mail"]
        assert params["max_results"]["maximum"] == 10

    def test_query_starred_recruiter_mail_filters_and_scores_candidates(self, tmp_path, monkeypatch):
        from spoke.gmail_operator import GmailOperator

        creds_path = tmp_path / "gmail_credentials.json"
        creds_path.write_text(
            json.dumps(
                {
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                    "refresh_token": "refresh-token",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("SPOKE_GMAIL_CREDENTIALS_PATH", str(creds_path))

        def fake_urlopen(req, timeout=None):
            url = req.full_url
            if url == "https://oauth2.googleapis.com/token":
                assert req.data.decode("utf-8") == (
                    "client_id=client-id&client_secret=client-secret"
                    "&refresh_token=refresh-token&grant_type=refresh_token"
                )
                return _json_response({"access_token": "access-token"})
            if url.startswith("https://gmail.googleapis.com/gmail/v1/users/me/messages?"):
                assert "labelIds=STARRED" in url
                assert "maxResults=15" in url
                return _json_response(
                    {
                        "messages": [
                            {"id": "m-1", "threadId": "t-1"},
                            {"id": "m-2", "threadId": "t-2"},
                            {"id": "m-3", "threadId": "t-3"},
                        ]
                    }
                )
            if url.endswith("/messages/m-1?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date"):
                return _json_response(
                    {
                        "id": "m-1",
                        "threadId": "t-1",
                        "snippet": "Would love to talk about a senior staff role.",
                        "internalDate": "300",
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "Ava Recruiter <ava@talent.example>"},
                                {"name": "Subject", "value": "Recruiting for a staff engineer role"},
                                {"name": "Date", "value": "Sat, 29 Mar 2026 10:00:00 -0400"},
                            ]
                        },
                    }
                )
            if url.endswith("/messages/m-2?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date"):
                return _json_response(
                    {
                        "id": "m-2",
                        "threadId": "t-2",
                        "snippet": "Wanted to reach out personally about the platform direction.",
                        "internalDate": "200",
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "Mina CTO <mina@startup.example>"},
                                {"name": "Subject", "value": "Quick note from a fellow builder"},
                                {"name": "Date", "value": "Sat, 29 Mar 2026 09:00:00 -0400"},
                            ]
                        },
                    }
                )
            if url.endswith("/messages/m-3?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date"):
                return _json_response(
                    {
                        "id": "m-3",
                        "threadId": "t-3",
                        "snippet": "Here are the notes from lunch.",
                        "internalDate": "100",
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "Friend <friend@example.com>"},
                                {"name": "Subject", "value": "Lunch follow-up"},
                                {"name": "Date", "value": "Sat, 29 Mar 2026 08:00:00 -0400"},
                            ]
                        },
                    }
                )
            raise AssertionError(f"unexpected url {url!r}")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = GmailOperator().execute_query("starred_recruiter_mail", max_results=2)

        assert result["mode"] == "starred_recruiter_mail"
        assert result["matched_count"] == 2
        assert [msg["id"] for msg in result["messages"]] == ["m-1", "m-2"]
        assert result["messages"][0]["matched_signals"] == [
            "from:recruiter",
            "subject:recruiting",
            "snippet:role",
        ]
        assert result["messages"][1]["matched_signals"] == ["from:cto"]

    def test_query_rejects_unsupported_mode(self):
        from spoke.gmail_operator import GmailOperator, GmailOperatorError

        with pytest.raises(GmailOperatorError, match="unsupported mode"):
            GmailOperator().execute_query("all_mail")

    def test_query_requires_credentials(self, monkeypatch):
        from spoke.gmail_operator import GmailOperator, GmailOperatorError

        monkeypatch.delenv("SPOKE_GMAIL_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("SPOKE_GMAIL_CLIENT_ID", raising=False)
        monkeypatch.delenv("SPOKE_GMAIL_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("SPOKE_GMAIL_REFRESH_TOKEN", raising=False)

        with pytest.raises(GmailOperatorError, match="Gmail credentials are not configured"):
            GmailOperator().execute_query("starred_recruiter_mail")

