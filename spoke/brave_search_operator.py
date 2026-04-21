"""Bounded Brave Search operator for the spoken command path.

Provides the assistant model a read-only public web-search affordance backed
by Brave Search. Results are capped, metadata-only, and shaped into a compact
JSON contract so the local command model gets grounded search without an
unbounded browsing surface.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode
import urllib.request


_DEFAULT_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchOperatorError(RuntimeError):
    """Raised when the bounded Brave Search contract cannot be satisfied."""


def tool_schema() -> dict[str, Any]:
    """Return the OpenAI tool schema for the bounded web-search surface."""

    return {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the public web in read-only mode via Brave Search. "
                "Pass a search query string and get back compact result "
                "summaries with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Plain-language public web search query. Examples: "
                            "'latest hugging face release', 'mlx audio voices endpoint', "
                            "'Brave Search MCP docs'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of search results to return.",
                    },
                },
                "required": ["query"],
            },
        },
    }


class BraveSearchOperator:
    """Execute bounded Brave Search queries with a local API key."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
    ):
        self._api_key = (
            api_key
            or os.environ.get("SPOKE_BRAVE_SEARCH_API_KEY")
            or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        ).strip()
        self._endpoint = (
            endpoint
            or os.environ.get("SPOKE_BRAVE_SEARCH_ENDPOINT")
            or _DEFAULT_BRAVE_SEARCH_ENDPOINT
        )

    def execute_search(self, query: str, *, max_results: int = 5) -> dict[str, Any]:
        if not query or not query.strip():
            raise BraveSearchOperatorError("query string must not be empty")
        if not 1 <= max_results <= 10:
            raise BraveSearchOperatorError("max_results must be between 1 and 10")
        if not self._api_key:
            raise BraveSearchOperatorError(
                "Brave Search API key is not configured; set "
                "SPOKE_BRAVE_SEARCH_API_KEY or BRAVE_SEARCH_API_KEY."
            )

        payload = self._search(query.strip(), max_results=max_results)
        results = payload.get("web", {}).get("results", [])
        if not isinstance(results, list):
            results = []
        shaped = [
            shaped_result
            for item in results[:max_results]
            if (shaped_result := self._shape_result(item)) is not None
        ]
        return {
            "query": query.strip(),
            "matched_count": len(shaped),
            "results": shaped,
            "provider": "brave_search",
        }

    def _search(self, query: str, *, max_results: int) -> dict[str, Any]:
        params = urlencode(
            [
                ("q", query),
                ("count", str(max_results)),
                ("text_decorations", "0"),
            ]
        )
        req = urllib.request.Request(
            f"{self._endpoint}?{params}",
            method="GET",
        )
        req.add_header("Accept", "application/json")
        req.add_unredirected_header("X-Subscription-Token", self._api_key)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise BraveSearchOperatorError("failed to query Brave Search") from exc

    def _shape_result(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        url = item.get("url")
        title = item.get("title")
        if not isinstance(url, str) or not url.strip():
            return None
        if not isinstance(title, str) or not title.strip():
            title = url
        shaped = {
            "title": title.strip(),
            "url": url.strip(),
            "description": self._string_or_empty(item.get("description")),
        }
        age = self._string_or_none(item.get("age"))
        if age:
            shaped["age"] = age
        return shaped

    @staticmethod
    def _string_or_empty(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return ""

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        return None
