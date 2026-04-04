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
    def test_tool_schema_exposes_query_parameter(self):
        from spoke.gmail_operator import tool_schema

        schema = tool_schema()
        params = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "query_gmail"
        assert "query" in params
        assert params["query"]["type"] == "string"
        assert params["max_results"]["maximum"] == 10
        assert "query" in schema["function"]["parameters"]["required"]

    def test_query_returns_matching_messages(self, tmp_path, monkeypatch):
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
                return _json_response({"access_token": "access-token"})
            if url.startswith("https://gmail.googleapis.com/gmail/v1/users/me/messages?"):
                assert "q=is%3Astarred" in url
                assert "maxResults=3" in url
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
                        "labelIds": ["STARRED", "INBOX"],
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "Ava <ava@example.com>"},
                                {"name": "Subject", "value": "Staff engineer role"},
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
                        "snippet": "Platform direction chat.",
                        "internalDate": "200",
                        "labelIds": ["STARRED"],
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "Mina <mina@startup.example>"},
                                {"name": "Subject", "value": "Quick note"},
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
                        "snippet": "Lunch notes.",
                        "internalDate": "100",
                        "labelIds": ["STARRED"],
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
            result = GmailOperator().execute_query("is:starred", max_results=3)

        assert result["query"] == "is:starred"
        assert result["matched_count"] == 3
        assert [msg["id"] for msg in result["messages"]] == ["m-1", "m-2", "m-3"]
        assert result["messages"][0]["labels"] == ["STARRED", "INBOX"]
        assert "snippet" in result["messages"][0]

    def test_query_rejects_empty_query(self):
        from spoke.gmail_operator import GmailOperator, GmailOperatorError

        with pytest.raises(GmailOperatorError, match="query string must not be empty"):
            GmailOperator().execute_query("")

        with pytest.raises(GmailOperatorError, match="query string must not be empty"):
            GmailOperator().execute_query("   ")

    def test_query_requires_credentials(self, tmp_path, monkeypatch):
        from spoke.gmail_operator import GmailOperator, GmailOperatorError

        monkeypatch.delenv("SPOKE_GMAIL_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("SPOKE_GMAIL_CLIENT_ID", raising=False)
        monkeypatch.delenv("SPOKE_GMAIL_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("SPOKE_GMAIL_REFRESH_TOKEN", raising=False)

        # Point at a nonexistent file so the default path can't find real creds
        op = GmailOperator(credentials_path=tmp_path / "nonexistent.json")
        with pytest.raises(GmailOperatorError, match="Gmail credentials are not configured"):
            op.execute_query("is:starred")

    def test_env_credentials_override_malformed_credentials_file(self, tmp_path, monkeypatch):
        from spoke.gmail_operator import GmailOperator

        creds_path = tmp_path / "gmail_credentials.json"
        creds_path.write_text("{not-json", encoding="utf-8")
        monkeypatch.setenv("SPOKE_GMAIL_CREDENTIALS_PATH", str(creds_path))
        monkeypatch.setenv("SPOKE_GMAIL_CLIENT_ID", "client-id")
        monkeypatch.setenv("SPOKE_GMAIL_REFRESH_TOKEN", "refresh-token")

        fake_operator = GmailOperator()

        with patch.object(fake_operator, "_refresh_access_token", return_value="access-token") as refresh_mock:
            with patch.object(fake_operator, "_query_messages", return_value=[]):
                result = fake_operator.execute_query("is:unread")

        refresh_mock.assert_called_once_with(
            {
                "client_id": "client-id",
                "client_secret": "",
                "refresh_token": "refresh-token",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
        assert result["matched_count"] == 0
