"""Internal operator subagents.

The operator remains the single user-facing interface. Subagents are internal
background jobs that can be launched, inspected, and cancelled through the
main command surface while keeping their own prompts, tool scopes, and
history-free execution lanes.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from .command import CommandClient

_SEARCH_SUBAGENT_SYSTEM_PROMPT = """\
You are a background search subagent for the spoke operator.

Your job is to investigate a narrow local-search question and return concise,
useful findings to the main operator. You are not the user-facing voice. You
are an internal worker.

Rules:
- Use only the provided read-only file tools.
- Do not write files, speak aloud, modify tray state, or mutate Epistaxis.
- Prefer direct answers with concrete file paths, symbols, and short evidence.
- If the repository does not contain enough information, say so plainly.
- Keep the result compact and operator-facing.
"""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_search_subagent_query(
    query: str,
    *,
    base_url: str,
    model: str,
    tools: list[dict[str, Any]],
    tool_executor: Callable[..., str],
    api_key: str | None = None,
    command_client_factory: Callable[..., CommandClient] = CommandClient,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    """Run a bounded background search query against the same OMLX endpoint."""
    client = command_client_factory(
        base_url=base_url,
        model=model,
        api_key=api_key,
        history_path=None,
        system_prompt=_SEARCH_SUBAGENT_SYSTEM_PROMPT,
    )
    client.set_spoke_headers(
        pathway="subagent", utterance_id=uuid.uuid4().hex[:8],
    )

    visible_response = ""
    final_response = ""
    for event in client.stream_command_events(
        query,
        tools=tools,
        tool_executor=tool_executor,
        cancel_check=cancel_check,
    ):
        if event.kind == "assistant_delta" and event.text:
            visible_response += event.text
        elif event.kind == "assistant_final":
            final_response = event.text

    # Prefer the canonical final answer over accumulated streaming deltas.
    # Streaming deltas can include tool-call scaffolding or provisional text
    # that the model supersedes in its final response.
    return final_response or visible_response


class SubagentManager:
    """Track operator-owned background subagent jobs."""

    def __init__(
        self,
        *,
        search_runner: Callable[[str, Callable[[], bool]], str],
        thread_factory: Callable[..., Any] = threading.Thread,
    ):
        self._search_runner = search_runner
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._counter = 0
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def launch(self, kind: str, prompt: str) -> dict[str, Any]:
        if kind != "search":
            raise ValueError(f"Unsupported subagent kind: {kind}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        with self._lock:
            self._counter += 1
            job_id = f"subagent-{self._counter}"
            cancel_event = threading.Event()
            job = {
                "id": job_id,
                "kind": kind,
                "prompt": prompt.strip(),
                "state": "queued",
                "created_at": _iso_now(),
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "_cancel_event": cancel_event,
            }
            self._jobs[job_id] = job
            self._order.insert(0, job_id)

        thread = self._thread_factory(
            target=self._run_job,
            args=(job_id,),
            daemon=True,
        )
        thread.start()
        return self._public_job(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_job(self._jobs[job_id]) for job_id in self._order]

    def get_job(self, subagent_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(subagent_id)
            if job is None:
                return {"error": f"Unknown subagent: {subagent_id}"}
            return self._public_job(job)

    def cancel(self, subagent_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(subagent_id)
            if job is None:
                return {"error": f"Unknown subagent: {subagent_id}"}
            if job["state"] in {"completed", "failed", "cancelled"}:
                return self._public_job(job)
            job["_cancel_event"].set()
            if job["state"] == "queued":
                job["state"] = "cancelling"
            elif job["state"] == "running":
                job["state"] = "cancelling"
            return self._public_job(job)

    def _run_job(self, subagent_id: str) -> None:
        with self._lock:
            job = self._jobs[subagent_id]
            if job["_cancel_event"].is_set():
                job["state"] = "cancelled"
                job["finished_at"] = _iso_now()
                return
            job["state"] = "running"
            job["started_at"] = _iso_now()
            prompt = job["prompt"]
            cancel_event = job["_cancel_event"]

        try:
            result = self._search_runner(prompt, cancel_event.is_set)
        except Exception as exc:
            with self._lock:
                job = self._jobs[subagent_id]
                if job["_cancel_event"].is_set():
                    job["state"] = "cancelled"
                else:
                    job["state"] = "failed"
                    job["error"] = str(exc)
                job["finished_at"] = _iso_now()
            return

        with self._lock:
            job = self._jobs[subagent_id]
            if job["_cancel_event"].is_set():
                job["state"] = "cancelled"
            else:
                job["state"] = "completed"
                job["result"] = result
            job["finished_at"] = _iso_now()

    @staticmethod
    def _public_job(job: dict[str, Any]) -> dict[str, Any]:
        result = job.get("result")
        preview = None
        if isinstance(result, str) and result:
            preview = result[:160]
        poll_hint = None
        if job["state"] in {"queued", "running", "cancelling"}:
            poll_hint = (
                "Subagent still in flight — do not poll in a tight loop. "
                "Continue other work and check again later when useful."
            )
        return {
            "id": job["id"],
            "kind": job["kind"],
            "prompt": job["prompt"],
            "state": job["state"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "result": result,
            "result_preview": preview,
            "error": job.get("error"),
            "poll_hint": poll_hint,
        }
