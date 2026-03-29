"""Bounded read-only Gmail operator for the spoken command path.

This module intentionally keeps the first Gmail affordance narrow:
query recent starred messages and filter them down to recruiter- or CTO-like
mail so the command overlay can surface a compact answer without arbitrary
mailbox access or repo-embedded secrets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
import urllib.request


_DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
_DEFAULT_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
_DEFAULT_CREDENTIALS_PATH = (
    Path.home() / "Library/Application Support/Spoke/gmail_credentials.json"
)
_LIST_ENDPOINT = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


class GmailOperatorError(RuntimeError):
    """Raised when the bounded Gmail contract cannot be satisfied."""


def tool_schema() -> dict[str, Any]:
    """Return the OpenAI tool schema for the bounded Gmail query surface."""

    return {
        "type": "function",
        "function": {
            "name": "query_gmail",
            "description": (
                "Query Gmail in read-only mode for compact message summaries. "
                "Current scope is limited to recent starred recruiter- or CTO-"
                "style mail."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["starred_recruiter_mail"],
                        "description": (
                            "Bounded Gmail query mode. "
                            "'starred_recruiter_mail' looks for recent starred "
                            "messages that appear recruiter- or CTO-adjacent."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of matching message summaries to return.",
                    },
                },
                "required": ["mode"],
            },
        },
    }


class GmailOperator:
    """Execute the narrow Gmail query surface with local OAuth material."""

    def __init__(self, credentials_path: str | Path | None = None):
        if credentials_path is None:
            override = os.environ.get("SPOKE_GMAIL_CREDENTIALS_PATH")
            credentials_path = (
                Path(override).expanduser() if override else _DEFAULT_CREDENTIALS_PATH
            )
        self._credentials_path = Path(credentials_path).expanduser()

    def execute_query(self, mode: str, *, max_results: int = 5) -> dict[str, Any]:
        if mode != "starred_recruiter_mail":
            raise GmailOperatorError(f"unsupported mode: {mode!r}")
        if not 1 <= max_results <= 10:
            raise GmailOperatorError("max_results must be between 1 and 10")

        credentials = self._load_credentials()
        access_token = self._refresh_access_token(credentials)
        matched = self._query_starred_recruiter_mail(access_token, max_results=max_results)
        return {
            "mode": mode,
            "matched_count": len(matched),
            "messages": matched,
            "scope": _DEFAULT_GMAIL_SCOPE,
        }

    def _load_credentials(self) -> dict[str, str]:
        env_credentials = {
            "client_id": os.environ.get("SPOKE_GMAIL_CLIENT_ID"),
            "client_secret": os.environ.get("SPOKE_GMAIL_CLIENT_SECRET"),
            "refresh_token": os.environ.get("SPOKE_GMAIL_REFRESH_TOKEN"),
            "token_uri": os.environ.get("SPOKE_GMAIL_TOKEN_URI"),
        }
        if env_credentials["client_id"] and env_credentials["refresh_token"]:
            return {
                "client_id": env_credentials["client_id"],
                "client_secret": env_credentials["client_secret"] or "",
                "refresh_token": env_credentials["refresh_token"],
                "token_uri": env_credentials["token_uri"] or _DEFAULT_TOKEN_URI,
            }

        raw: dict[str, Any] = {}
        if self._credentials_path.exists():
            try:
                raw = json.loads(self._credentials_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise GmailOperatorError(
                    f"failed to read Gmail credentials from {self._credentials_path}"
                ) from exc

        nested = raw.get("installed") or raw.get("web") or {}
        credentials = {
            "client_id": env_credentials["client_id"]
            or raw.get("client_id")
            or nested.get("client_id"),
            "client_secret": env_credentials["client_secret"]
            or raw.get("client_secret")
            or nested.get("client_secret"),
            "refresh_token": env_credentials["refresh_token"] or raw.get("refresh_token"),
            "token_uri": env_credentials["token_uri"]
            or raw.get("token_uri")
            or nested.get("token_uri")
            or _DEFAULT_TOKEN_URI,
        }

        if not credentials["client_id"] or not credentials["refresh_token"]:
            raise GmailOperatorError(
                "Gmail credentials are not configured; set "
                "SPOKE_GMAIL_CREDENTIALS_PATH or provide "
                "SPOKE_GMAIL_CLIENT_ID / SPOKE_GMAIL_REFRESH_TOKEN."
            )
        return credentials

    def _refresh_access_token(self, credentials: dict[str, str]) -> str:
        fields: list[tuple[str, str]] = [
            ("client_id", credentials["client_id"]),
        ]
        if credentials.get("client_secret"):
            fields.append(("client_secret", credentials["client_secret"]))
        fields.extend(
            [
                ("refresh_token", credentials["refresh_token"]),
                ("grant_type", "refresh_token"),
            ]
        )
        payload = urlencode(fields).encode("utf-8")
        req = urllib.request.Request(
            credentials["token_uri"],
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                token_payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise GmailOperatorError("failed to refresh Gmail access token") from exc

        access_token = token_payload.get("access_token")
        if not access_token:
            raise GmailOperatorError("refresh token exchange returned no access_token")
        return str(access_token)

    def _query_starred_recruiter_mail(
        self, access_token: str, *, max_results: int
    ) -> list[dict[str, Any]]:
        # Fetch a bounded candidate set, then apply repo-side heuristics so the
        # first slice stays predictable and privacy-preserving.
        list_payload = self._api_get(
            _LIST_ENDPOINT,
            access_token,
            query=[
                ("labelIds", "STARRED"),
                ("maxResults", str(max(max_results * 5, 15))),
            ],
        )
        messages = list_payload.get("messages", [])
        if not isinstance(messages, list):
            return []

        matched: list[dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict) or "id" not in item:
                continue
            message = self._fetch_message_metadata(access_token, str(item["id"]))
            candidate = self._shape_candidate(message)
            if candidate is not None:
                matched.append(candidate)

        matched.sort(key=lambda item: item["internal_date"], reverse=True)
        return [self._public_message_shape(item) for item in matched[:max_results]]

    def _fetch_message_metadata(self, access_token: str, message_id: str) -> dict[str, Any]:
        return self._api_get(
            f"{_LIST_ENDPOINT}/{message_id}",
            access_token,
            query=[
                ("format", "metadata"),
                ("metadataHeaders", "From"),
                ("metadataHeaders", "Subject"),
                ("metadataHeaders", "Date"),
            ],
        )

    def _api_get(
        self,
        url: str,
        access_token: str,
        *,
        query: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        if query:
            url = f"{url}?{urlencode(query, doseq=True)}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise GmailOperatorError("failed to query Gmail API") from exc

    def _shape_candidate(self, message: dict[str, Any]) -> dict[str, Any] | None:
        headers = self._header_map(message)
        sender = headers.get("from", "")
        subject = headers.get("subject", "")
        date = headers.get("date", "")
        snippet = str(message.get("snippet", "")).strip()
        signals = self._matched_signals(sender, subject, snippet)
        if not signals:
            return None
        return {
            "id": str(message.get("id", "")),
            "thread_id": str(message.get("threadId", "")),
            "from": sender,
            "subject": subject,
            "date": date,
            "snippet": snippet,
            "matched_signals": signals,
            "internal_date": int(str(message.get("internalDate", "0")) or "0"),
        }

    def _matched_signals(
        self, sender: str, subject: str, snippet: str
    ) -> list[str]:
        sender_lower = sender.lower()
        subject_lower = subject.lower()
        snippet_lower = snippet.lower()
        signals: list[str] = []

        if any(term in sender_lower for term in ("recruiter", "recruiting", "talent", "sourcer", "headhunter")):
            signals.append("from:recruiter")
        if "cto" in sender_lower or "chief technology officer" in sender_lower:
            signals.append("from:cto")
        if "recruit" in subject_lower:
            signals.append("subject:recruiting")
        if any(term in snippet_lower for term in (" role", "role ", "opportunity", "staff role")):
            signals.append("snippet:role")

        return signals

    def _public_message_shape(self, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": message["id"],
            "thread_id": message["thread_id"],
            "from": message["from"],
            "subject": message["subject"],
            "date": message["date"],
            "snippet": message["snippet"],
            "matched_signals": message["matched_signals"],
        }

    def _header_map(self, message: dict[str, Any]) -> dict[str, str]:
        payload = message.get("payload", {})
        headers = payload.get("headers", []) if isinstance(payload, dict) else []
        mapped: dict[str, str] = {}
        for header in headers:
            if not isinstance(header, dict):
                continue
            name = str(header.get("name", "")).strip().lower()
            value = str(header.get("value", "")).strip()
            if name and value:
                mapped[name] = value
        return mapped
