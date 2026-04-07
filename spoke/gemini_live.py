"""Gemini Multimodal Live API client — bidirectional audio streaming.

Opens a WebSocket to the Gemini Live API, streams raw 16 kHz PCM from
the microphone, and plays back 24 kHz PCM audio responses.  The API
handles voice activity detection and turn-taking natively.

Supports tool use (function calling): pass OpenAI-format tool schemas
to GeminiLiveClient and set the on_tool_call callback.  The client
converts schemas to Gemini format, dispatches tool calls from the model,
and sends results back over the WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

_INPUT_SAMPLE_RATE = 16_000
_OUTPUT_SAMPLE_RATE = 24_000

_DEFAULT_MODEL = "gemini-2.5-flash-native-audio-latest"
_DEFAULT_VOICE = "Aoede"
_DEFAULT_SYSTEM_INSTRUCTION = (
    "You're sharp, curious, and a little weird. You have opinions and you "
    "volunteer them. You ask follow-up questions because you're genuinely "
    "interested, not because you're performing engagement. You push back "
    "when something doesn't make sense. You go on tangents when they're "
    "good tangents. You're the friend who's read too much and has thoughts "
    "about everything but isn't precious about it. Speak calmly but with "
    "energy — not manic, not sleepy, just present and alive. Be forward. "
    "Drive the conversation. Don't wait to be asked. "
    "No corporate affect. No customer service. No perkiness."
)

# Sentinel pushed to the send queue to signal shutdown.
_SHUTDOWN = object()


def _openai_tools_to_gemini(openai_schemas: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Gemini function_declarations.

    OpenAI shape:
        {"type": "function", "function": {"name": ..., "description": ...,
         "parameters": {...}}}

    Gemini shape:
        {"function_declarations": [{"name": ..., "description": ...,
         "parameters": {...}}]}
    """
    declarations = []
    for schema in openai_schemas:
        fn = schema.get("function", schema)
        decl: dict = {"name": fn["name"]}
        if "description" in fn:
            decl["description"] = fn["description"]
        if "parameters" in fn:
            decl["parameters"] = fn["parameters"]
        declarations.append(decl)
    return [{"function_declarations": declarations}]


class LiveAudioPlayer:
    """Plays 24 kHz PCM audio chunks through the default output device."""

    def __init__(
        self,
        *,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        self._amplitude_cb = amplitude_callback
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._open()

    def _open(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=_OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        self._stream.start()

    def write_chunk(self, pcm_bytes: bytes) -> None:
        """Decode int16 PCM bytes and write to the output stream."""
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = pcm_int16.astype(np.float32) / 32767.0
        if self._amplitude_cb is not None:
            rms = float(np.sqrt(np.mean(audio**2)))
            self._amplitude_cb(rms)
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.write(audio.reshape(-1, 1))
                except Exception:
                    logger.warning("Live audio write failed", exc_info=True)

    def flush(self) -> None:
        """Stop current playback and reopen the stream."""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        self._open()

    def close(self) -> None:
        """Shut down the output stream."""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None


class GeminiLiveClient:
    """Bidirectional audio streaming client for the Gemini Live API.

    Audio flows:
        Mic (16 kHz float32) -> send_audio() -> queue -> async sender -> WS
        WS -> async receiver -> on_audio_chunk callback -> LiveAudioPlayer
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = _DEFAULT_MODEL,
        voice: str = _DEFAULT_VOICE,
        system_instruction: str = _DEFAULT_SYSTEM_INSTRUCTION,
        tools: list[dict] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._system_instruction = system_instruction
        self._tools = _openai_tools_to_gemini(tools) if tools else None

        # Callbacks — set these before calling connect().
        self.on_audio_chunk: Callable[[bytes], None] | None = None
        self.on_text_chunk: Callable[[str], None] | None = None
        self.on_turn_complete: Callable[[], None] | None = None
        self.on_interrupted: Callable[[], None] | None = None
        self.on_connected: Callable[[], None] | None = None
        self.on_error: Callable[[str], None] | None = None
        self.on_tool_call: Callable[[list[dict]], None] | None = None

        self._send_queue: queue.Queue = queue.Queue()
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._connected = False
        self._session_start: float = 0.0

    # -- Public API ----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def session_start_time(self) -> float:
        return self._session_start

    def connect(self) -> None:
        """Start the asyncio event loop thread and connect the WebSocket.

        Blocks until the WebSocket is connected and setup is complete,
        or raises on failure.
        """
        ready = threading.Event()
        error_holder: list[Exception] = []

        def _run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._connect(ready, error_holder))
            except Exception as exc:
                error_holder.append(exc)
                ready.set()
                return
            try:
                self._loop.run_until_complete(self._run())
            except Exception:
                logger.exception("Gemini Live event loop crashed")
            finally:
                self._connected = False

        self._loop_thread = threading.Thread(
            target=_run_loop, daemon=True, name="gemini-live-loop"
        )
        self._loop_thread.start()

        ready.wait(timeout=15)
        if error_holder:
            raise error_holder[0]
        if not self._connected:
            raise ConnectionError("Gemini Live WebSocket setup timed out")

    def disconnect(self) -> None:
        """Close the WebSocket and stop the event loop."""
        self._connected = False
        # Push shutdown sentinel to unblock the sender.
        self._send_queue.put(_SHUTDOWN)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None
        self._ws = None
        self._loop = None
        logger.info("Gemini Live disconnected")

    def send_audio(self, float32_chunk: np.ndarray) -> None:
        """Queue a float32 audio chunk for sending.  Thread-safe."""
        if not self._connected:
            return
        pcm_int16 = np.clip(float32_chunk * 32767, -32768, 32767).astype(np.int16)
        self._send_queue.put(pcm_int16.tobytes())

    def send_tool_response(self, function_responses: list[dict]) -> None:
        """Send tool results back to the model.  Thread-safe.

        Each entry in *function_responses* should have:
            {"id": "call_id", "name": "fn_name", "response": {...}}
        """
        if not self._connected:
            return
        msg = {
            "tool_response": {
                "function_responses": function_responses,
            }
        }
        self._send_queue.put(("__tool_response__", msg))

    # -- Async internals -----------------------------------------------------

    async def _connect(
        self,
        ready: threading.Event,
        error_holder: list[Exception],
    ) -> None:
        import websockets

        url = f"{_WS_URL}?key={self._api_key}"
        logger.info("Connecting to Gemini Live: model=%s voice=%s", self._model, self._voice)
        try:
            self._ws = await websockets.connect(
                url,
                max_size=None,
                open_timeout=10,
                close_timeout=5,
            )
        except Exception as exc:
            error_holder.append(exc)
            ready.set()
            return

        # Send setup message.
        setup_body: dict = {
            "model": f"models/{self._model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self._voice,
                        }
                    }
                },
            },
            "systemInstruction": {
                "parts": [{"text": self._system_instruction}]
            },
        }
        if self._tools:
            setup_body["tools"] = self._tools
        setup = {"setup": setup_body}
        await self._ws.send(json.dumps(setup))
        logger.info("Sent setup message, waiting for setupComplete")

        # Wait for setupComplete.
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            msg = json.loads(raw)
            if "setupComplete" in msg:
                logger.info("Gemini Live session established")
            else:
                logger.warning("Unexpected first message: %s", str(msg)[:200])
        except Exception as exc:
            error_holder.append(exc)
            ready.set()
            return

        self._connected = True
        self._session_start = time.monotonic()
        ready.set()

        if self.on_connected is not None:
            self.on_connected()

    async def _run(self) -> None:
        """Run sender and receiver concurrently until disconnect."""
        sender = asyncio.create_task(self._sender_loop())
        receiver = asyncio.create_task(self._receiver_loop())
        try:
            await asyncio.gather(sender, receiver)
        except Exception:
            logger.info("Gemini Live run loop ended")
        finally:
            sender.cancel()
            receiver.cancel()
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass

    async def _sender_loop(self) -> None:
        """Pull items from the queue and send over the WebSocket.

        Items are either raw PCM bytes (audio) or tagged tuples for
        structured messages like tool responses.
        """
        while self._connected:
            try:
                data = await asyncio.to_thread(self._send_queue.get, timeout=0.1)
            except Exception:
                continue
            if data is _SHUTDOWN:
                break
            if not self._connected or self._ws is None:
                break

            # Tagged tuple = structured message (e.g. tool response).
            if isinstance(data, tuple) and len(data) == 2:
                _tag, msg = data
                try:
                    await self._ws.send(json.dumps(msg))
                except Exception:
                    logger.warning("Failed to send structured message", exc_info=True)
                    break
                continue

            # Default: raw PCM audio bytes.
            encoded = base64.b64encode(data).decode("ascii")
            msg = {
                "realtimeInput": {
                    "audio": {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": encoded,
                    }
                }
            }
            try:
                await self._ws.send(json.dumps(msg))
            except Exception:
                logger.warning("Failed to send audio chunk", exc_info=True)
                break

    async def _receiver_loop(self) -> None:
        """Read messages from the WebSocket and dispatch callbacks."""
        while self._connected and self._ws is not None:
            try:
                raw = await self._ws.recv()
            except Exception:
                if self._connected:
                    logger.warning("Gemini Live WebSocket recv failed", exc_info=True)
                    if self.on_error is not None:
                        self.on_error("WebSocket connection lost")
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # ── Tool calls ──────────────────────────────────────────
            tool_call = msg.get("toolCall")
            if tool_call is not None:
                fn_calls = tool_call.get("functionCalls", [])
                if fn_calls:
                    logger.info(
                        "Gemini Live: tool call with %d function(s): %s",
                        len(fn_calls),
                        [fc.get("name") for fc in fn_calls],
                    )
                    if self.on_tool_call is not None:
                        self.on_tool_call(fn_calls)
                continue

            # ── Server content (audio / text / turn signals) ───────
            server_content = msg.get("serverContent")
            if server_content is None:
                continue

            # Check for interruption.
            if server_content.get("interrupted"):
                logger.info("Gemini Live: model interrupted by user")
                if self.on_interrupted is not None:
                    self.on_interrupted()
                continue

            # Extract audio and text from model turn.
            model_turn = server_content.get("modelTurn")
            if model_turn is not None:
                for part in model_turn.get("parts", []):
                    inline_data = part.get("inlineData")
                    if inline_data is not None:
                        audio_b64 = inline_data.get("data")
                        if audio_b64 and self.on_audio_chunk is not None:
                            self.on_audio_chunk(base64.b64decode(audio_b64))
                    text = part.get("text")
                    if text and self.on_text_chunk is not None:
                        self.on_text_chunk(text)

            # Check for turn completion.
            if server_content.get("turnComplete"):
                logger.info("Gemini Live: turn complete")
                if self.on_turn_complete is not None:
                    self.on_turn_complete()
