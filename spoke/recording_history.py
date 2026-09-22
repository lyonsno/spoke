"""Re-transcription operations independent of the live dictation client."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def client_receipt(client) -> dict:
    """Snapshot effective client identity while its inference lock is held."""
    receipt = {"client": f"{type(client).__module__}.{type(client).__name__}"}
    for name in ("_model", "_model_id", "_model_path", "_binary", "_url",
                 "_decode_timeout", "_eager_eval", "_encoder_compute", "_decoder_compute"):
        value = getattr(client, name, None)
        if isinstance(value, (str, int, float, bool, Path)):
            receipt[name.removeprefix("_")] = str(value) if isinstance(value, Path) else value
    for name in ("_last_receipt", "_last_prompt_receipt", "last_route"):
        value = getattr(client, name, None)
        if isinstance(value, dict):
            receipt[name.removeprefix("_")] = copy.deepcopy(value)
    prompt = getattr(getattr(client, "_prompt_provider", None), "last_resolved", None)
    candidates = [receipt.get("last_prompt_receipt", {})]
    for name in ("last_receipt", "last_route"):
        candidates.append(receipt.get(name, {}).get("prompt", {}))
    if prompt is not None and any(
        isinstance(value, dict) and value.get("sha256") == prompt.sha256
        for value in candidates
    ):
        receipt["prompt_snapshot"] = {"text": prompt.text, "sha256": prompt.sha256,
                                      "sources": list(prompt.sources)}
    return receipt


def run_retranscription(spool, capture_id, attempt_id, route, client_factory, inference_context):
    """Run one exact-route replay. Failures never fall back or replace an attempt."""
    started = time.monotonic()
    phase = "audio_validation"
    text, error, client = None, None, None
    effective = {}
    try:
        wav = spool.read_recording_audio(capture_id)
        phase = "client_creation"
        client = client_factory(dict(route))
        phase = "inference"
        with inference_context(client):
            try:
                text = client.transcribe(wav)
                if not isinstance(text, str):
                    raise TypeError("Transcription backend returned non-text output")
            finally:
                effective = client_receipt(client)
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        phase = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Recording re-transcription failed at %s", phase)
    spool.finish_attempt(
        capture_id, attempt_id, text=text, error=error, effective=effective,
        wall_seconds=time.monotonic() - started, evidence={"phase": phase},
    )


def first_line(record: dict) -> str:
    """The original attempt remains the list preview after later replays."""
    attempts = record.get("attempts", [])
    original = next((a for a in attempts if a.get("kind") == "live"), None)
    if original is None:
        return "No original transcript" if not record.get("error") else "Unverified recording"
    text = original.get("text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.splitlines()[0].split())
    return {"pending": "Transcription pending", "failed": "Transcription failed",
            "blank": "Blank transcription"}.get(original.get("status"), "Unverified transcription")


def duration_label(seconds) -> str:
    if not isinstance(seconds, (int, float)):
        return "Unknown duration"
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def delivery_label(attempt: dict) -> str:
    deliveries = attempt.get("deliveries", [])
    if not deliveries:
        return "Delivery: unverified" if attempt.get("kind") == "live" else "Delivery: not requested"
    state = deliveries[-1]["state"]
    label = {
        "routed_to_switcher": "Teleporter filter; saved to tray",
        "saved_to_tray_focus_changed": "Saved to tray after focus changed",
        "saved_to_tray_grace_cancelled": "Saved to tray; insertion cancelled",
        "saved_to_tray": "Saved to tray",
        "insert_requested": "Paste requested; acceptance unverified",
        "clipboard_restored": "Clipboard restored; paste acceptance unverified",
        "paste_failed_saved_to_tray": "Paste failed; saved to tray",
        "delivery_skipped_stale": "Insertion skipped; retained in history",
        "command_requested": "Assistant request started; completion unverified",
        "command_response_started": "Assistant response started; completion unverified",
        "command_failed": "Assistant request failed; retained in history",
        "copied": "Copied to clipboard",
    }.get(state, f"Unverified state ({state})")
    return f"Delivery: {label}"
