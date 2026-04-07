"""Entry point for spoke — macOS global hold-to-dictate.

Run with:  uv run spoke
    or:    uv run python -m spoke

Configure via environment variables:
    SPOKE_WHISPER_URL          Sidecar server URL (optional — if unset, uses local MLX Whisper)
    SPOKE_WHISPER_MODEL        Model name (default: mlx-community/whisper-large-v3-turbo)
    SPOKE_HOLD_MS              Hold threshold in ms (default: 200, must be > 0)
    SPOKE_RESTORE_DELAY_MS     Pasteboard restore delay in ms (default: 1000)
    SPOKE_PARAKEET_MODEL_DIR   Path to FluidInference/parakeet-ctc-110m-coreml clone dir
                               (enables Parakeet CoreML/ANE preview model in the menu)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import faulthandler
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import uuid

import objc
from AppKit import (
    NSAlert,
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSTextField,
)
from Foundation import NSMakeRect, NSObject

_NS_COMMAND_KEY_MASK = 1 << 20
_NS_KEY_DOWN_MASK = 1 << 10

# Keep _PastableTextField as an alias so existing alloc() calls don't break.
_PastableTextField = NSTextField


def _extract_command_error_detail(raw: bytes) -> str | None:
    """Best-effort extraction of a useful provider error message."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    def _walk(value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        if isinstance(value, dict):
            for key in ("message", "detail", "error", "title"):
                found = _walk(value.get(key))
                if found:
                    return found
        if isinstance(value, list):
            for item in value:
                found = _walk(item)
                if found:
                    return found
        return None

    return _walk(payload)


def _format_command_http_error(exc) -> str:
    """Render HTTP provider failures into something the user can act on."""
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
    base = f"HTTP {exc.code}"
    if reason:
        base = f"{base} {reason}"

    detail = None
    try:
        if getattr(exc, "fp", None) is not None:
            detail = _extract_command_error_detail(exc.read())
    except Exception:
        logger.exception("Failed to read command HTTP error body")

    if detail:
        return f"{base} — {detail}"
    return base


def _run_modal_with_paste(alert) -> int:
    """Run an NSAlert modally with Cmd+V/C/X/A support.

    NSAlert modals lack an Edit menu, so standard editing key equivalents
    don't work.  This installs a local event monitor that intercepts
    Cmd+V/C/X/A keyDown events, routes them to the alert window's field
    editor, and swallows all other Cmd+key events so nothing accidentally
    dismisses the dialog.
    """
    from AppKit import NSEvent

    def _handle(event):
        try:
            if event.modifierFlags() & _NS_COMMAND_KEY_MASK:
                chars = event.charactersIgnoringModifiers()
                win = alert.window()
                if win is not None:
                    fr = win.firstResponder()
                    if chars in ("v", "c", "x", "a") and fr is not None:
                        if chars == "v":
                            fr.paste_(None)
                        elif chars == "c":
                            fr.copy_(None)
                        elif chars == "x":
                            fr.cut_(None)
                        elif chars == "a":
                            fr.selectAll_(None)
                # Swallow all Cmd+key so nothing dismisses the dialog.
                return None
        except Exception:
            logger.exception("Alert event monitor handler error")
        return event

    monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        _NS_KEY_DOWN_MASK, _handle
    )
    try:
        return alert.runModal()
    finally:
        NSEvent.removeMonitor_(monitor)

from .capture import AudioCapture
from .command import CommandClient, _DEFAULT_COMMAND_MODEL, _DEFAULT_COMMAND_URL
from .narrator import ThinkingNarrator
from .focus_check import has_focused_text_input, focused_text_contains
from .handsfree import HandsFreeController, HandsFreeState, match_voice_command
from .scene_capture import SceneCaptureCache
from .tool_dispatch import execute_tool, get_tool_schemas
from .glow import GlowOverlay
from .inject import inject_text, inject_text_raw, save_pasteboard, restore_pasteboard, set_pasteboard_only
from .input_tap import SpacebarHoldDetector
from .launch_targets import (
    current_launch_target,
    current_launch_target_id,
    iter_launch_targets,
    save_selected_launch_target,
)
from .menubar import MenuBarIcon
from .overlay import TranscriptionOverlay
from .transcribe import TranscriptionClient
from .transcribe_local import LocalTranscriptionClient, supports_eager_eval
from .transcribe_parakeet import ParakeetCoreMLClient, _PARAKEET_MODEL_ID
from .transcribe_qwen import LocalQwenClient
from .tts import TTSClient, RemoteTTSClient, CloudTTSClient, GEMINI_VOICES, _DEFAULT_VOICE
from .heartbeat import (
    HeartbeatManager,
    zombie_sweep,
    HEARTBEAT_INTERVAL_S,
    _is_process_alive,
)

logger = logging.getLogger(__name__)

_LOCK_PATH = os.environ.get(
    "SPOKE_LOCK_PATH",
    os.path.expanduser("~/Library/Logs/.spoke.lock"),
)

_DEFAULT_PREVIEW_MODEL = "mlx-community/whisper-base.en-mlx-8bit"
_DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-medium.en-mlx-8bit"
_DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT = 30.0
_DEFAULT_LOCAL_WHISPER_EAGER_EVAL = False
_DEFAULT_COMMAND_BACKEND = "local"
_DEFAULT_COMMAND_MODEL_DIR = Path.home() / ".lmstudio" / "models"
_DEFAULT_COMMAND_SIDECAR_URL = ""
_DEFAULT_CLOUD_PROVIDER = "google"
_DEFAULT_CLOUD_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_DEFAULT_CLOUD_MODEL = "gemini-2.5-flash"
_DEFAULT_OPENROUTER_CLOUD_URL = "https://openrouter.ai/api/v1"
_DEFAULT_OPENROUTER_CLOUD_MODEL = "stepfun/step-3.5-flash:free"
_DEFAULT_TTS_SIDECAR_URL = "http://MacBook-Pro-2.local:9001"
_DEFAULT_WHISPER_SIDECAR_URL = ""
_DEFAULT_WHISPER_CLOUD_URL = "https://api.openai.com"
_DEFAULT_WHISPER_CLOUD_MODEL = "whisper-1"


def _url_host(url: str) -> str:
    """Extract host:port from a URL for display."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc or url


def _cloud_provider_label(provider: str) -> str:
    if provider == "openrouter":
        return "OpenRouter"
    return "Google Cloud"


def _default_cloud_url_for_provider(provider: str) -> str:
    if provider == "openrouter":
        return _DEFAULT_OPENROUTER_CLOUD_URL
    return _DEFAULT_CLOUD_URL


def _default_cloud_model_for_provider(provider: str) -> str:
    if provider == "openrouter":
        return _DEFAULT_OPENROUTER_CLOUD_MODEL
    return _DEFAULT_CLOUD_MODEL


def _ensure_edit_menu() -> None:
    """Install a minimal Edit menu so Cmd+V/C/X/A work in NSAlert text fields.

    Agent-style apps (NSApplicationActivationPolicyAccessory) have no menu bar,
    so the standard key equivalents never reach NSTextField.  This installs an
    Edit menu once; subsequent calls are no-ops.
    """
    from AppKit import NSApp, NSMenu, NSMenuItem
    app = NSApp()
    if app is None:
        return
    main_menu = app.mainMenu()
    if main_menu is None:
        main_menu = NSMenu.new()
        app.setMainMenu_(main_menu)
    # Check if Edit menu already exists.
    for i in range(main_menu.numberOfItems()):
        if main_menu.itemAtIndex_(i).title() == "Edit":
            return
    edit_menu = NSMenu.alloc().initWithTitle_("Edit")
    for title, action, key in [
        ("Cut", "cut:", "x"),
        ("Copy", "copy:", "c"),
        ("Paste", "paste:", "v"),
        ("Select All", "selectAll:", "a"),
    ]:
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            title, action, key,
        )
        edit_menu.addItem_(item)
    edit_item = NSMenuItem.new()
    edit_item.setTitle_("Edit")
    edit_item.setSubmenu_(edit_menu)
    main_menu.addItem_(edit_item)
_CURATED_LOCAL_COMMAND_MODEL_IDS = [
    "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
    "mlx-community/Qwen3-4B-Thinking-2507-8bit",
    "lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit",
    "lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-8bit",
    "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx",
]

_TTS_MODELS = [
    ("mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "Voxtral 4B (4-bit)"),
    ("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "Qwen3-TTS 1.7B CustomVoice (8-bit)"),
    ("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "Qwen3-TTS 0.6B CustomVoice (8-bit)"),
    ("mlx-community/VibeVoice-Realtime-0.5B-fp16", "VibeVoice 0.5B Realtime (fp16)"),
    ("mlx-community/Kokoro-82M-bf16", "Kokoro 82M (bf16)"),
    ("k2-fsa/OmniVoice", "OmniVoice"),
]

# ── Per-model voice presets ──────────────────────────────────────
# Each entry maps a model ID (lowercased) to a list of (voice_value, label)
# tuples.  The menu shows these as a choice picker.  Models not listed here
# fall back to the manual "Set TTS Voice…" text field.

_VOXTRAL_VOICES: list[tuple[str, str]] = [
    ("casual_female", "Casual Female"),
    ("casual_male", "Casual Male"),
    ("cheerful_female", "Cheerful Female"),
    ("neutral_female", "Neutral Female"),
    ("neutral_male", "Neutral Male"),
    ("fr_female", "French Female"),
    ("fr_male", "French Male"),
    ("es_female", "Spanish Female"),
    ("es_male", "Spanish Male"),
    ("de_female", "German Female"),
    ("de_male", "German Male"),
    ("it_female", "Italian Female"),
    ("it_male", "Italian Male"),
    ("pt_female", "Portuguese Female"),
    ("pt_male", "Portuguese Male"),
    ("nl_female", "Dutch Female"),
    ("nl_male", "Dutch Male"),
    ("ar_male", "Arabic Male"),
    ("hi_female", "Hindi Female"),
    ("hi_male", "Hindi Male"),
]

_KOKORO_VOICES: list[tuple[str, str]] = [
    ("af_heart", "Heart (American F)"),
    ("af_alloy", "Alloy (American F)"),
    ("af_aoede", "Aoede (American F)"),
    ("af_bella", "Bella (American F)"),
    ("af_jessica", "Jessica (American F)"),
    ("af_kore", "Kore (American F)"),
    ("af_nicole", "Nicole (American F)"),
    ("af_nova", "Nova (American F)"),
    ("af_river", "River (American F)"),
    ("af_sarah", "Sarah (American F)"),
    ("af_sky", "Sky (American F)"),
    ("am_adam", "Adam (American M)"),
    ("am_echo", "Echo (American M)"),
    ("am_eric", "Eric (American M)"),
    ("am_fenrir", "Fenrir (American M)"),
    ("am_liam", "Liam (American M)"),
    ("am_michael", "Michael (American M)"),
    ("am_onyx", "Onyx (American M)"),
    ("am_puck", "Puck (American M)"),
    ("bf_alice", "Alice (British F)"),
    ("bf_emma", "Emma (British F)"),
    ("bf_isabella", "Isabella (British F)"),
    ("bf_lily", "Lily (British F)"),
    ("bm_daniel", "Daniel (British M)"),
    ("bm_fable", "Fable (British M)"),
    ("bm_george", "George (British M)"),
    ("bm_lewis", "Lewis (British M)"),
    ("ff_siwis", "Siwis (French F)"),
    ("ef_dora", "Dora (Spanish F)"),
    ("em_alex", "Alex (Spanish M)"),
    ("hf_alpha", "Alpha (Hindi F)"),
    ("hf_beta", "Beta (Hindi F)"),
    ("hm_omega", "Omega (Hindi M)"),
    ("hm_psi", "Psi (Hindi M)"),
    ("if_sara", "Sara (Italian F)"),
    ("im_nicola", "Nicola (Italian M)"),
    ("jf_alpha", "Alpha (Japanese F)"),
    ("jm_kumo", "Kumo (Japanese M)"),
    ("pf_dora", "Dora (Portuguese F)"),
    ("pm_alex", "Alex (Portuguese M)"),
    ("zf_xiaobei", "Xiaobei (Chinese F)"),
    ("zf_xiaoni", "Xiaoni (Chinese F)"),
    ("zm_yunjian", "Yunjian (Chinese M)"),
    ("zm_yunxi", "Yunxi (Chinese M)"),
]

_VIBEVOICE_VOICES: list[tuple[str, str]] = [
    ("en-Emma_woman", "Emma (English F)"),
    ("en-Grace_woman", "Grace (English F)"),
    ("en-Carter_man", "Carter (English M)"),
    ("en-Davis_man", "Davis (English M)"),
    ("en-Frank_man", "Frank (English M)"),
    ("en-Mike_man", "Mike (English M)"),
    ("fr-Spk1_woman", "French F"),
    ("fr-Spk0_man", "French M"),
    ("de-Spk1_woman", "German F"),
    ("de-Spk0_man", "German M"),
    ("it-Spk0_woman", "Italian F"),
    ("it-Spk1_man", "Italian M"),
    ("sp-Spk0_woman", "Spanish F"),
    ("sp-Spk1_man", "Spanish M"),
    ("pt-Spk0_woman", "Portuguese F"),
    ("pt-Spk1_man", "Portuguese M"),
    ("nl-Spk1_woman", "Dutch F"),
    ("nl-Spk0_man", "Dutch M"),
    ("jp-Spk1_woman", "Japanese F"),
    ("jp-Spk0_man", "Japanese M"),
    ("kr-Spk0_woman", "Korean F"),
    ("kr-Spk1_man", "Korean M"),
    ("pl-Spk1_woman", "Polish F"),
    ("pl-Spk0_man", "Polish M"),
    ("in-Samuel_man", "Samuel (Hindi M)"),
]

_QWEN3_TTS_VOICES: list[tuple[str, str]] = [
    ("serena", "Serena"),
    ("vivian", "Vivian"),
    ("ryan", "Ryan"),
    ("aiden", "Aiden"),
    ("eric", "Eric"),
    ("dylan", "Dylan"),
    ("ono_anna", "Ono Anna"),
    ("sohee", "Sohee"),
    ("uncle_fu", "Uncle Fu"),
]

_OMNIVOICE_PROMPT_PRESETS: list[tuple[str, str]] = [
    ("", "Auto voice"),
    ("female, child", "Female, child"),
    ("male, high pitch, indian accent", "Male, high pitch, Indian"),
    ("female, elderly, british accent", "Female, elderly, British"),
    ("female, young adult, whisper", "Female, young adult, whisper"),
    ("male, middle-aged, very low pitch", "Male, middle-aged, very low pitch"),
    ("female, low pitch, british accent", "Female, low pitch, British"),
    ("male, british accent", "Male, British"),
    ("female, whisper, british accent", "Female whisper, British"),
    ("female, high pitch, american accent", "Female, high pitch, American"),
    ("male, low pitch, american accent", "Male, low pitch, American"),
]

# Model ID (lowercased) → voice presets.  Checked with str.contains so
# partial model IDs work (e.g. "voxtral" matches all Voxtral quants).
_MODEL_VOICE_PRESETS: list[tuple[str, list[tuple[str, str]]]] = [
    ("voxtral", _VOXTRAL_VOICES),
    ("kokoro", _KOKORO_VOICES),
    ("vibevoice", _VIBEVOICE_VOICES),
    ("qwen3-tts", _QWEN3_TTS_VOICES),
    ("omnivoice", _OMNIVOICE_PROMPT_PRESETS),
    ("gemini", GEMINI_VOICES),
]

_OMNIVOICE_PROMPT_LEXICON = {
    "Gender": ("female", "male"),
    "Age": ("child", "young adult", "middle-aged", "elderly"),
    "Pitch": ("very low pitch", "low pitch", "high pitch", "very high pitch"),
    "Style": ("whisper",),
    "English accent": ("american accent", "british accent", "indian accent"),
    "Chinese dialect examples": ("sichuan dialect", "shaanxi dialect"),
}


def _is_omnivoice_tts_model(model_id: str | None) -> bool:
    return isinstance(model_id, str) and model_id.strip().lower() == "k2-fsa/omnivoice"


def _voice_presets_for_model(model_id: str | None) -> list[tuple[str, str]] | None:
    """Return the voice presets for a model, or None if no presets exist."""
    if not model_id:
        return None
    lowered = model_id.lower()
    for key, presets in _MODEL_VOICE_PRESETS:
        if key in lowered:
            return presets
    return None


def _voice_choices_for_model(
    model_id: str | None, current_voice: str
) -> list[tuple[str, str, bool]] | None:
    """Build a choice list for the voice menu, or None if no presets exist."""
    presets = _voice_presets_for_model(model_id)
    if presets is None:
        return None
    choices = [(voice, label, True) for voice, label in presets]
    # If current voice isn't in presets, insert it as a custom entry
    if current_voice and all(voice != current_voice for voice, _label in presets):
        choices.insert(0, (current_voice, f"Custom: {current_voice}", True))
    choices.append(("configure_voice", "Set Custom Voice…", True))
    return choices


def _voice_preset_label(model_id: str | None, voice: str) -> str:
    """Return a human label for the current voice, falling back to the raw value."""
    presets = _voice_presets_for_model(model_id)
    if presets:
        for preset_voice, label in presets:
            if voice == preset_voice:
                return label
    return voice or "(not set)"


def _omnivoice_prompt_lexicon_text() -> str:
    parts = []
    for heading, values in _OMNIVOICE_PROMPT_LEXICON.items():
        parts.append(f"{heading}: {', '.join(values)}")
    return (
        "Combine OmniVoice prompt keywords with commas.\n\n"
        + "\n".join(parts)
        + "\n\nExamples: female, low pitch, british accent; "
        "male, middle-aged, very low pitch; female, young adult, whisper."
    )

_NOT_CAPTURED = object()  # sentinel for _pre_paste_clipboard
_PROCESS_LAUNCH_ID = os.environ.get("SPOKE_LAUNCH_ID") or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
os.environ["SPOKE_LAUNCH_ID"] = _PROCESS_LAUNCH_ID


def _runtime_phase_path() -> Path:
    return Path(
        os.environ.get(
            "SPOKE_RUNTIME_PHASE_PATH",
            str(Path.home() / "Library" / "Logs" / "spoke-last-phase.json"),
        )
    ).expanduser()


def _flush_logging_handlers() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


# ── AVCaptureDevice mic permission helpers ─────────────────────
# These check/request microphone permission via the macOS AVFoundation API
# without allocating any PortAudio buffers.  Under memory pressure, PortAudio
# can fail to open a stream even though mic permission is already granted.
# Using AVCaptureDevice avoids that failure mode entirely.
#
# AVAuthorizationStatus values:
#   0 = notDetermined, 1 = restricted, 2 = denied, 3 = authorized

def _get_av_auth_status() -> int:
    """Return AVCaptureDevice.authorizationStatus(for: .audio) as an int."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        return int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio))
    except Exception:
        logger.debug("AVCaptureDevice import failed — falling back to unknown", exc_info=True)
        return -1  # unknown / unavailable


def _request_av_mic_access() -> bool:
    """Block until the user grants or denies mic access. Returns True if granted."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        import threading
        result = threading.Event()
        granted_box: list[bool] = [False]

        def handler(granted: bool) -> None:
            granted_box[0] = granted
            result.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, handler
        )
        result.wait(timeout=60)
        return granted_box[0]
    except Exception:
        logger.debug("AVCaptureDevice.requestAccess failed", exc_info=True)
        return False


def _record_runtime_phase(phase: str, **details) -> None:
    cwd = os.getcwd()
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase": phase,
        "launch_id": _PROCESS_LAUNCH_ID,
        "parent_launch_id": os.environ.get("SPOKE_PARENT_LAUNCH_ID"),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": cwd,
        "python": sys.executable,
        "thread": threading.current_thread().name,
    }
    phase_details = {key: value for key, value in details.items() if value is not None}
    launch_target = current_launch_target(Path(cwd))
    if launch_target is not None:
        phase_details.setdefault("launch_target_id", launch_target["id"])
        phase_details.setdefault("launch_target_label", launch_target["label"])
    payload.update(phase_details)

    try:
        path = _runtime_phase_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("Failed to write runtime phase snapshot")

    detail_text = ", ".join(f"{key}={value!r}" for key, value in phase_details.items())
    if detail_text:
        logger.info("Runtime phase: %s (%s)", phase, detail_text)
    else:
        logger.info("Runtime phase: %s", phase)
    _flush_logging_handlers()


def _install_crash_diagnostics() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        logger.warning("Failed to enable faulthandler", exc_info=True)

    previous_sys_excepthook = sys.excepthook

    def _sys_excepthook(exc_type, exc, tb):
        _record_runtime_phase(
            "exception.uncaught",
            exception_type=getattr(exc_type, "__name__", str(exc_type)),
            exception=str(exc),
            traceback="".join(traceback.format_exception(exc_type, exc, tb)),
        )
        previous_sys_excepthook(exc_type, exc, tb)

    sys.excepthook = _sys_excepthook

    previous_threading_excepthook = getattr(threading, "excepthook", None)

    def _threading_excepthook(args):
        _record_runtime_phase(
            "exception.thread",
            exception_type=getattr(args.exc_type, "__name__", str(args.exc_type)),
            exception=str(args.exc_value),
            thread_name=getattr(args.thread, "name", None),
            traceback="".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            ),
        )
        if previous_threading_excepthook is not None:
            previous_threading_excepthook(args)

    if previous_threading_excepthook is not None:
        threading.excepthook = _threading_excepthook


def _get_ram_gb() -> float:
    """Return system RAM in GB via sysctl."""
    import subprocess
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


_MIN_RAM_GB_FOR_V3_TURBO = 16.0
_MIN_RAM_GB_FOR_UNCAPPED_RECORDING = 16.0
_LOW_RAM_RECORDING_CAP_SECS = 20.0


def _max_record_secs_for_ram(ram_gb: float) -> float | None:
    """Return the recording cap for local inference on this RAM tier."""
    if ram_gb < _MIN_RAM_GB_FOR_UNCAPPED_RECORDING:
        return _LOW_RAM_RECORDING_CAP_SECS
    return None


def _iter_local_command_model_ids(model_dir: Path) -> list[str]:
    """Return a curated list of installed MLX model ids from a local model directory."""
    if not model_dir.is_dir():
        return []

    discovered: set[str] = set()
    for child in sorted(model_dir.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        child_entries = sorted(child.iterdir(), key=lambda path: path.name.lower())
        if _is_local_command_model_leaf(child):
            discovered.add(child.name)
            continue
        for grandchild in child_entries:
            if _is_local_command_model_leaf(grandchild):
                discovered.add(f"{child.name}/{grandchild.name}")
    return [
        model_id
        for model_id in _CURATED_LOCAL_COMMAND_MODEL_IDS
        if model_id in discovered
    ]


def _is_local_command_model_leaf(model_path: Path) -> bool:
    """Return True when a leaf directory looks like an installed MLX model."""
    if not model_path.is_dir():
        return False

    file_names = {
        entry.name
        for entry in model_path.iterdir()
        if entry.is_file()
    }
    if "config.json" not in file_names or "tokenizer.json" not in file_names:
        return False
    if "model.safetensors" in file_names or "model.safetensors.index.json" in file_names:
        return True
    return any(name.endswith(".safetensors") for name in file_names)
# Recording cap: let 16GB+ local boxes that can run v3 turbo record freely.
# Keep the 20s cap only for smaller local machines. No cap in sidecar mode.
_RAM_GB = _get_ram_gb()
_MAX_RECORD_SECS: float | None = _max_record_secs_for_ram(_RAM_GB)


@dataclass
class TrayEntry:
    """A tray entry with minimal ownership/provenance metadata."""

    text: str
    owner: str = "user"
    acknowledged: bool = True

    def __eq__(self, other):
        if isinstance(other, TrayEntry):
            return (
                self.text == other.text
                and self.owner == other.owner
                and self.acknowledged == other.acknowledged
            )
        if isinstance(other, str):
            return self.text == other
        return NotImplemented

    @property
    def display_owner(self) -> str:
        if self.owner == "assistant" and not self.acknowledged:
            return "assistant"
        return "user"


class SegmentAccumulator:
    """Thread-safe accumulator for opportunistic segment transcriptions.

    During recording, silence-bounded audio segments are dispatched to a remote
    transcription backend as they arrive.  Results are stored in order so that
    on release the final transcription only needs to cover the tail audio since
    the last segment boundary.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: list[str] = []  # transcription text per segment, in order
        self._pending: int = 0  # segments dispatched but not yet returned
        self._done = threading.Event()
        self._done.set()  # nothing pending initially

    def dispatch(self, wav_bytes: bytes, client: object) -> None:
        """Transcribe *wav_bytes* on a background thread and store the result."""
        with self._lock:
            idx = len(self._results)
            self._results.append("")  # placeholder
            self._pending += 1
            self._done.clear()

        def _work():
            try:
                text = client.transcribe(wav_bytes)
            except Exception:
                logger.exception("Segment %d transcription failed", idx)
                text = ""
            with self._lock:
                self._results[idx] = text
                self._pending -= 1
                if self._pending <= 0:
                    self._done.set()

        t = threading.Thread(target=_work, daemon=True)
        t.start()

    def wait(self, timeout: float = 10.0) -> bool:
        """Block until all pending segments are transcribed."""
        return self._done.wait(timeout=timeout)

    @property
    def text(self) -> str:
        """Concatenated transcription of all completed segments."""
        with self._lock:
            parts = [r for r in self._results if r]
        return " ".join(parts)

    @property
    def count(self) -> int:
        """Number of segments dispatched (completed + pending)."""
        with self._lock:
            return len(self._results)

    @property
    def has_results(self) -> bool:
        with self._lock:
            return any(self._results)

    def reset(self) -> None:
        with self._lock:
            self._results.clear()
            self._pending = 0
            self._done.set()


class SpokeAppDelegate(NSObject):
    """Main application delegate — wires input → capture → transcribe → inject."""

    def init(self):
        self = objc.super(SpokeAppDelegate, self).init()
        if self is None:
            return None

        whisper_url = os.environ.get("SPOKE_WHISPER_URL", "")
        hold_ms_raw = os.environ.get("SPOKE_HOLD_MS", "200")
        try:
            hold_ms = int(hold_ms_raw)
        except ValueError:
            logger.error("SPOKE_HOLD_MS must be an integer, got %r", hold_ms_raw)
            print(
                f"ERROR: SPOKE_HOLD_MS must be an integer, got {hold_ms_raw!r}.\n"
                "  Example: SPOKE_HOLD_MS=400 uv run spoke",
                file=sys.stderr,
            )
            sys.exit(1)

        if hold_ms <= 0:
            logger.error("SPOKE_HOLD_MS must be > 0, got %d", hold_ms)
            print(
                f"ERROR: SPOKE_HOLD_MS must be > 0, got {hold_ms}.\n"
                "  Example: SPOKE_HOLD_MS=400 uv run spoke",
                file=sys.stderr,
            )
            sys.exit(1)

        # Whisper backend configuration — shared URL/key pool.
        self._whisper_sidecar_url = (
            self._load_preference("whisper_sidecar_url")
            or os.environ.get("SPOKE_WHISPER_URL", "")
            or _DEFAULT_WHISPER_SIDECAR_URL
        )
        self._whisper_cloud_url = (
            self._load_preference("whisper_cloud_url")
            or _DEFAULT_WHISPER_CLOUD_URL
        )
        self._whisper_cloud_api_key = (
            self._load_preference("whisper_cloud_api_key") or ""
        )
        self._whisper_cloud_model = (
            self._load_preference("whisper_cloud_model")
            or _DEFAULT_WHISPER_CLOUD_MODEL
        )
        # If sidecar URL came from env but not yet saved, adopt it.
        if whisper_url and not self._load_preference("whisper_sidecar_url"):
            self._whisper_sidecar_url = whisper_url

        # Per-role backend selection: preview (partials) and transcription (finals).
        default_backend = "sidecar" if whisper_url else "local"
        self._whisper_backend = (
            self._load_preference("whisper_backend") or default_backend
        )
        self._preview_backend = (
            self._load_preference("preview_backend") or self._whisper_backend
        )

        # Resolve effective URL + API key for each role.
        transcription_url, transcription_api_key = self._resolve_whisper_endpoint(
            self._whisper_backend
        )
        preview_url, preview_api_key = self._resolve_whisper_endpoint(
            self._preview_backend
        )
        self._whisper_url = transcription_url
        self._whisper_api_key = transcription_api_key
        self._preview_url = preview_url
        self._preview_api_key = preview_api_key

        self._preview_model_id, self._transcription_model_id = self._resolve_model_ids()
        # Cloud backend uses its own model ID for each role.
        if self._whisper_backend == "cloud":
            self._transcription_model_id = self._whisper_cloud_model
        if self._preview_backend == "cloud":
            self._preview_model_id = self._whisper_cloud_model
        self._client_cache: dict[tuple[str, str], object] = {}
        self._capture = AudioCapture()
        self._capture.warmup()
        self._local_mode = not bool(transcription_url) and not bool(preview_url)
        (
            self._local_whisper_decode_timeout,
            self._local_whisper_eager_eval,
        ) = self._resolve_local_whisper_settings()
        self._client = self._get_client(
            transcription_url, self._transcription_model_id, transcription_api_key,
        )
        self._preview_client = self._get_client(
            preview_url, self._preview_model_id, preview_api_key,
        )
        self._detector = SpacebarHoldDetector.alloc().initWithHoldStart_holdEnd_holdMs_(
            self._on_hold_start,
            self._on_hold_end,
            hold_ms,
        )
        # Wire tray callbacks on the detector
        self._detector._on_shift_tap = self._on_tray_shift_tap
        self._detector._on_shift_tap_during_hold = self._on_tray_navigate_up
        self._detector._on_shift_tap_idle = self._on_audio_shift_tap
        # Bare Enter should belong to the foreground app; tray actions stay
        # on explicit space-rooted gestures instead of ambient key capture.
        self._detector._on_enter_pressed = None
        self._detector._on_tray_delete = self._on_tray_delete_gesture
        self._detector._on_cancel_spring_start = self._on_cancel_spring_start
        self._detector._on_cancel_spring_release = self._on_cancel_spring_release
        self._detector._on_enter_during_waiting = self._toggle_command_overlay
        self._detector._on_double_tap_shift = self._toggle_terraform_hud
        self._menubar: MenuBarIcon | None = None
        self._glow: GlowOverlay | None = None
        self._overlay: TranscriptionOverlay | None = None
        self._transcribing = False
        self._transcription_token = 0
        self._parallel_insert_token = 0
        self._cancel_spring_active = False
        self._cancel_spring_start = 0.0
        self._preview_active = False
        self._preview_thread: threading.Thread | None = None
        self._preview_done = threading.Event()
        self._preview_done.set()
        self._preview_session_token = 0
        self._local_inference_lock = threading.Lock()
        self._record_start_time: float = 0.0
        self._cap_fired = False
        self._last_preview_text = ""
        self._models_ready = False
        self._warm_error: Exception | None = None
        self._warmup_in_flight = False
        self._hold_rejected_during_warmup = False
        self._mic_probe_in_flight = False
        self._mic_ready = False  # True once mic probe succeeds at least once

        # Command pathway — always enabled, but can persist a sidecar URL.
        self._command_backend = None
        self._command_cloud_provider = self._load_cloud_provider_preference()
        self._command_sidecar_url = self._load_command_sidecar_url_preference()
        self._command_url = None
        command_backend, command_url = self._resolve_command_backend()
        if command_url:
            self._command_backend = command_backend
            self._command_url = command_url
            if command_backend == "sidecar":
                self._command_sidecar_url = command_url
            cloud_api_key = None
            if command_backend == "cloud":
                self._command_cloud_provider = self._load_cloud_provider_preference()
                self._command_model_id = (
                    self._load_cloud_model_preference(self._command_cloud_provider)
                    or self._load_command_model_preference()
                    or _default_cloud_model_for_provider(self._command_cloud_provider)
                )
                cloud_api_key = self._resolve_command_cloud_api_key(
                    command_url,
                    self._command_cloud_provider,
                )
            else:
                self._command_model_id = (
                    self._load_command_model_preference()
                    or os.environ.get("SPOKE_COMMAND_MODEL")
                    or _DEFAULT_COMMAND_MODEL
                )
            client_kwargs = {
                "base_url": command_url,
                "model": self._command_model_id,
            }
            if cloud_api_key:
                client_kwargs["api_key"] = cloud_api_key
            self._command_client = CommandClient(**client_kwargs)
            self._command_server_unreachable = False
            # Thinking narrator sidecar
            if ThinkingNarrator.is_enabled():
                self._narrator = ThinkingNarrator(
                    on_summary=lambda s: self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "narratorSummary:", {"summary": s}, False
                    ),
                    on_thinking_collapsed=lambda s: self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "narratorCollapsed:", {"text": s}, False
                    ),
                )
            else:
                self._narrator = None
            self._command_model_options = self._seed_command_model_options(
                self._command_model_id
            )
            self._command_models_refresh_in_flight = False
            self._command_overlay: TranscriptionOverlay | None = None
            self._scene_cache = SceneCaptureCache(max_captures=10)
            self._tool_schemas = get_tool_schemas()
            logger.info(
                "Command pathway enabled: backend=%s url=%s model=%s",
                command_backend,
                command_url,
                self._command_model_id,
            )
        else:
            self._command_url = None
            self._command_client = None
            self._narrator = None
            self._command_backend = None
            self._command_cloud_provider = self._load_cloud_provider_preference()
            self._command_url = None
            self._command_model_id = None
            self._command_model_options = []
            self._command_models_refresh_in_flight = False
            self._command_server_unreachable = False
            self._command_overlay = None
            self._scene_cache = None
            self._tool_schemas = None

        # Heartbeat — zombie sweep runs before us, this starts the writer.
        self._heartbeat = HeartbeatManager()
        self._heartbeat.set_context(
            launch_target=os.environ.get("SPOKE_LAUNCH_TARGET_ID"),
            worktree=os.getcwd(),
        )
        self._heartbeat.set_evict_callback(self._evict_model)
        self._heartbeat_timer = None

        # TTS autoplay — initialized if a voice is configured via preferences or env.
        # Preference-based model override takes priority over env var.
        tts_model_pref = self._load_preferences().get("tts_model")
        if tts_model_pref:
            os.environ["SPOKE_TTS_MODEL"] = tts_model_pref
        self._tts_backend = self._load_preference("tts_backend") or "local"
        self._tts_sidecar_url = (
            self._load_preference("tts_sidecar_url") or _DEFAULT_TTS_SIDECAR_URL
        )
        self._tts_client = self._build_tts_client()
        self._tts_server_unreachable = False
        self._command_tool_used_tts = False
        if self._tts_client is not None:
            if isinstance(self._tts_client, CloudTTSClient):
                backend_label = "cloud"
            elif isinstance(self._tts_client, RemoteTTSClient):
                backend_label = "sidecar"
            else:
                backend_label = "local"
            logger.info("TTS enabled: backend=%s voice=%s", backend_label, self._tts_client._voice)

        # Tray state — speech-native stacked clipboard
        self._tray_stack: list[TrayEntry | str] = []
        self._tray_index: int = 0
        self._tray_active: bool = False
        self._tray_tool_result: dict | None = None

        # Recovery mode state (implementation detail of tray)
        # _NOT_CAPTURED sentinel distinguishes "not captured yet" from
        # "captured but clipboard was empty (None)".
        self._pre_paste_clipboard: list[tuple[str, bytes]] | None | object = _NOT_CAPTURED
        self._verify_paste_text: str | None = None
        self._verify_paste_attempt: int = 0
        self._result_pending_inject = None
        self._recovery_saved_clipboard: list[tuple[str, bytes]] | None = None
        self._recovery_text: str | None = None
        self._recovery_clipboard_state: str = "idle"
        self._recovery_pending_insert = None
        self._recovery_hold_active: bool = False
        self._recovery_retry_pending: bool = False

        if self._local_mode and _MAX_RECORD_SECS is not None:
            logger.info(
                "RAM %.0fGB < %.0fGB — recording capped at %.0fs to avoid Metal crashes",
                _RAM_GB,
                _MIN_RAM_GB_FOR_UNCAPPED_RECORDING,
                _MAX_RECORD_SECS,
            )

        # Hands-free continuous dictation controller
        self._handsfree = HandsFreeController(self)
        self._handsfree.on_state_change = self._on_handsfree_state_change

        return self

    # ── NSApplication delegate ──────────────────────────────

    def applicationDidFinishLaunching_(self, notification) -> None:
        _record_runtime_phase(
            "app.did_finish_launching",
            local_mode=self._local_mode,
            preview_model=self._preview_model_id,
            transcription_model=self._transcription_model_id,
            command_model=self._command_model_id,
            tts_enabled=self._tts_client is not None,
        )
        _ensure_edit_menu()
        self._menubar = MenuBarIcon.alloc().initWithQuitCallback_selectModelCallback_(
            self._quit, self._handle_model_menu_action
        )
        self._menubar.setup()

        self._glow = GlowOverlay.alloc().initWithScreen_(None)
        self._glow.setup()

        self._overlay = TranscriptionOverlay.alloc().initWithScreen_(None)
        self._overlay.setup()

        # Command output overlay — separate surface for command responses
        if self._command_client is not None:
            from .command_overlay import CommandOverlay
            self._command_overlay = CommandOverlay.alloc().initWithScreen_(None)
            self._command_overlay.setup()
            self._command_overlay._on_cancel_spring_threshold = self._on_cancel_spring_threshold
            self._refresh_command_model_options_async()

        # Terraform topoi HUD — restores last visibility state (default: open)
        from .terraform_hud import TerraformHUD
        self._terraform_hud = TerraformHUD.alloc().init()
        self._terraform_hud.restore_visibility()
        self._menubar._on_toggle_terraform = self._terraform_hud.toggle

        # Hands-free mode — wire menubar toggle if Porcupine key is available
        if os.environ.get("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY"):
            self._menubar._on_toggle_handsfree = self._toggle_handsfree

        # Iron Giant: install event tap and probe mic in parallel.
        # The event tap (spacebar interception) only needs Accessibility permission,
        # not mic access.  Under memory pressure the mic probe can fail even though
        # the system has already granted mic permission, which previously blocked
        # the entire app.  Now the spacebar works immediately and mic readiness is
        # tracked independently.
        self._menubar.set_status_text("Starting up…")
        self._setup_event_tap()
        self._request_mic_permission()

    def _request_mic_permission(self) -> None:
        """Check mic permission via AVCaptureDevice (no PortAudio allocation).

        Under memory pressure, PortAudio can fail to open a stream even when
        mic permission is already granted.  This probe uses the macOS
        AVFoundation API which requires zero buffer allocation — it just asks
        the OS whether the app has permission.  PortAudio stream opening is
        deferred to first actual recording (capture.start()).
        """
        if self._mic_probe_in_flight:
            return
        self._mic_probe_in_flight = True
        _record_runtime_phase("mic_probe.start")
        threading.Thread(
            target=self._probe_mic_permission, daemon=True, name="mic-probe"
        ).start()

    def _probe_mic_permission(self) -> None:
        """Background-thread mic probe — uses AVCaptureDevice, no PortAudio."""
        status = _get_av_auth_status()

        if status == 3:  # authorized
            logger.info("Microphone access granted (AVCaptureDevice)")
            _record_runtime_phase("mic_probe.granted")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "micPermissionGranted:", None, False
            )
        elif status == 0:  # not determined — need to trigger the system prompt
            logger.info("Mic permission not yet determined — requesting")
            _record_runtime_phase("mic_probe.requesting")
            granted = _request_av_mic_access()
            if granted:
                logger.info("Microphone access granted after prompt")
                _record_runtime_phase("mic_probe.granted_after_prompt")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micPermissionGranted:", None, False
                )
            else:
                logger.warning("Mic permission denied by user")
                _record_runtime_phase("mic_probe.denied_by_user")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micPermissionDenied:", None, False
                )
        elif status in (1, 2):  # restricted or denied
            logger.warning("Mic permission denied (status=%d)", status)
            _record_runtime_phase("mic_probe.denied", status=status)
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "micPermissionDenied:", None, False
            )
        else:  # -1 or unknown — AVFoundation unavailable, fall back to sd.rec
            logger.warning("AVCaptureDevice unavailable — falling back to PortAudio probe")
            self._probe_mic_permission_portaudio_fallback()

    def _probe_mic_permission_portaudio_fallback(self) -> None:
        """Legacy PortAudio-based probe, used only if AVFoundation is unavailable."""
        import sounddevice as sd
        try:
            sd.rec(1600, samplerate=16000, channels=1, dtype='float32', blocking=True)
            logger.info("Microphone access granted (PortAudio fallback)")
            _record_runtime_phase("mic_probe.granted_portaudio_fallback")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "micPermissionGranted:", None, False
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(m in msg for m in ("permission", "not permitted", "access denied")):
                _record_runtime_phase("mic_probe.denied_portaudio_fallback")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micPermissionDenied:", None, False
                )
            else:
                logger.warning("PortAudio fallback probe failed", exc_info=True)
                _record_runtime_phase("mic_probe.failed_portaudio_fallback", error=str(exc))
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micProbeFailed:", None, False
                )

    def micPermissionGranted_(self, _sender) -> None:
        """Main-thread callback after mic probe succeeds."""
        self._mic_probe_in_flight = False
        self._mic_ready = True
        if self._menubar is not None and self._models_ready:
            self._menubar.set_status_text("Ready — hold spacebar")

    def micPermissionDenied_(self, _sender) -> None:
        """Main-thread callback after mic probe fails — schedule retry."""
        from Foundation import NSTimer
        self._mic_probe_in_flight = False
        if self._menubar is not None:
            self._menubar.set_status_text("Mic: grant access, then wait…")
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            2.0, self, "retryMicPermission:", None, False
        )

    def micProbeFailed_(self, _sender) -> None:
        """Main-thread callback after a non-permission mic probe failure."""
        from Foundation import NSTimer
        self._mic_probe_in_flight = False
        if self._menubar is not None:
            self._menubar.set_status_text("Mic unavailable — retrying…")
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            2.0, self, "retryMicPermission:", None, False
        )

    def retryMicPermission_(self, timer) -> None:
        """Retry mic access check — dispatches to background thread."""
        if self._mic_probe_in_flight:
            return
        self._mic_probe_in_flight = True
        threading.Thread(
            target=self._probe_mic_permission, daemon=True, name="mic-probe-retry"
        ).start()

    def _setup_event_tap(self) -> None:
        """Install the event tap after permissions are confirmed."""
        if not self._detector.install():
            self._show_accessibility_alert()
            self._menubar.set_status_text("Grant accessibility, then wait…")
            # Retry every 2 seconds
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryEventTap:", None, False
            )
            return

        self._complete_event_tap_startup()

    def _complete_event_tap_startup(self) -> None:
        self._hold_rejected_during_warmup = False
        self._models_ready = False
        self._warm_error = None
        self._refresh_startup_status()
        if self._warmup_in_flight:
            return
        self._warmup_in_flight = True
        threading.Thread(
            target=self._prepare_clients_in_background,
            daemon=True,
            name="client-warmup",
        ).start()

    def _prepare_clients_in_background(self) -> None:
        _record_runtime_phase("client_warmup.start")
        try:
            self._prepare_clients()
        except Exception as exc:
            self._warm_error = exc
            logger.exception("Model preparation failed")
            _record_runtime_phase("client_warmup.failed", error=str(exc))
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "clientWarmupFailed:", None, False
            )
            return

        _record_runtime_phase("client_warmup.succeeded")
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "clientWarmupSucceeded:", None, False
        )

    def clientWarmupSucceeded_(self, _sender) -> None:
        self._warmup_in_flight = False
        self._models_ready = True
        self._warm_error = None
        logger.info("spoke ready — hold spacebar to record")
        _record_runtime_phase("app.ready")
        if self._mic_ready:
            self._menubar.set_status_text("Ready — hold spacebar")
        else:
            self._menubar.set_status_text("Models ready — mic unavailable, retrying…")
        self._hide_startup_status()

        # Warn if cloud preview is active — each partial is a paid API call.
        if getattr(self, "_preview_backend", "local") == "cloud":
            overlay = getattr(self, "_overlay", None)
            if overlay is not None:
                overlay.flash_notice(
                    "Cloud preview active\n"
                    "Partials sent every 3 seconds — each is a charged API call.",
                    hold=4.0,
                    fade=2.0,
                )

        # Register loaded models with the heartbeat manager.
        self._register_loaded_models()
        self._start_heartbeat_timer()

        # Auto-enable hands-free wake word listening if Porcupine key is available.
        hf = getattr(self, "_handsfree", None)
        if hf is not None and os.environ.get("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY"):
            hf.enable()

        # Keep TTS lazy. Startup-time local TTS warmup can monopolize the same
        # MLX lock transcription uses, which starves preview/final text on
        # OmniVoice surfaces until the TTS model finishes loading.

    def clientWarmupFailed_(self, _sender) -> None:
        self._warmup_in_flight = False
        self._models_ready = False
        exc = self._warm_error or RuntimeError("Model warmup failed")
        self._refresh_startup_status()
        self._show_model_load_alert(exc)

    # ── Hands-free mode ────────────────────────────────────────

    def _toggle_handsfree(self) -> None:
        """Menu/manual toggle for hands-free mode."""
        hf = self._handsfree
        if hf.is_active:
            hf.disable()
        else:
            hf.enable()

    def handleWakeWord_(self, payload: dict) -> None:
        """Main-thread selector called by HandsFreeController on wake word detection."""
        role = payload.get("role", "")
        self._handsfree.handle_wake_word(role)

    def handsFreeInject_(self, payload: dict) -> None:
        """Main-thread selector: inject hands-free transcription at cursor."""
        text = payload.get("text", "")
        dest = payload.get("dest", "cursor")
        if not text:
            return

        # Check for voice commands (only for cursor-destined segments)
        if dest == "cursor":
            cmd = match_voice_command(text)
            if cmd is not None:
                action, value = cmd
                logger.info("Hands-free voice command: %r → %s(%r)", text, action, value)
                if action == "keystroke":
                    self._synthesize_keystroke(value)
                elif action == "chord":
                    self._synthesize_chord(value)
                elif action == "tray_enter":
                    self._handsfree_tray_enter()
                elif action == "tray_insert":
                    self._handsfree_tray_insert()
                else:
                    inject_text_raw(value)
                return

        # Route based on destination
        if dest == "tray":
            logger.info("Hands-free → tray: %r", text)
            self._add_tray_entry(text, owner="handsfree", activate=False)
            return

        # Normal text — append trailing space for continuous flow.
        # Use inject_text_raw to skip per-segment clipboard save/restore;
        # the HandsFreeController saves clipboard once at dictation start.
        logger.info("Hands-free inject: %r", text)
        inject_text_raw(text + " ")

    _KEYSTROKE_MAP = {
        "return": 36,
        "tab": 48,
        "escape": 53,
        "delete": 51,
        "space": 49,
    }

    def _synthesize_keystroke(self, key_name: str) -> None:
        """Synthesize a bare keystroke (no modifiers)."""
        from Quartz import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap
        keycode = self._KEYSTROKE_MAP.get(key_name)
        if keycode is None:
            logger.warning("Unknown keystroke: %s", key_name)
            return
        down = CGEventCreateKeyboardEvent(None, keycode, True)
        up = CGEventCreateKeyboardEvent(None, keycode, False)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
        logger.info("Synthesized keystroke: %s (keycode %d)", key_name, keycode)

    _CHORD_KEYCODE_MAP = {
        "[": 33,
        "]": 30,
    }

    def _synthesize_chord(self, chord: str) -> None:
        """Synthesize a modifier+key chord like 'cmd+shift+['."""
        from Quartz import (
            CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags,
            kCGEventFlagMaskCommand, kCGEventFlagMaskShift, kCGHIDEventTap,
        )
        parts = chord.lower().split("+")
        key = parts[-1]
        mods = set(parts[:-1])

        keycode = self._CHORD_KEYCODE_MAP.get(key) or self._KEYSTROKE_MAP.get(key)
        if keycode is None:
            logger.warning("Unknown chord key: %s", key)
            return

        flags = 0
        if "cmd" in mods:
            flags |= kCGEventFlagMaskCommand
        if "shift" in mods:
            flags |= kCGEventFlagMaskShift

        down = CGEventCreateKeyboardEvent(None, keycode, True)
        CGEventSetFlags(down, flags)
        up = CGEventCreateKeyboardEvent(None, keycode, False)
        CGEventSetFlags(up, 0)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
        logger.info("Synthesized chord: %s (keycode %d, flags %d)", chord, keycode, flags)

    def _handsfree_tray_enter(self) -> None:
        """Voice command: route the next spoken segment to the tray."""
        hf = getattr(self, "_handsfree", None)
        if hf is None:
            return
        logger.info("Hands-free: next segment → tray")
        hf.route_next_segment("tray")

    def _handsfree_tray_insert(self) -> None:
        """Voice command: insert the current tray entry at cursor."""
        if not getattr(self, "_tray_stack", None):
            logger.info("Hands-free: no tray entries to insert")
            return
        # Insert the most recent tray entry
        entry = self._tray_stack[-1]
        text = entry if isinstance(entry, str) else getattr(entry, "text", str(entry))
        if text:
            logger.info("Hands-free: inserting tray entry: %r", text)
            inject_text(text + " ")

    def _on_handsfree_state_change(self, state: HandsFreeState) -> None:
        """Update UI when hands-free state changes."""
        if self._menubar is None:
            return

        if state == HandsFreeState.DORMANT:
            self._menubar.set_status_text("Ready — hold spacebar")
            self._menubar.set_recording(False)
            if self._glow is not None:
                self._glow.hide()
        elif state == HandsFreeState.LISTENING:
            self._menubar.set_status_text("Listening for wake word…")
            self._menubar.set_recording(False)
            if self._glow is not None:
                self._glow.hide()
        elif state == HandsFreeState.DICTATING:
            self._menubar.set_status_text("Hands-free — dictating…")
            self._menubar.set_recording(True)
            if self._glow is not None:
                self._glow.show()

    def _warm_tts_in_background(self) -> None:
        tts = getattr(self, "_tts_client", None)
        if tts is None:
            return
        try:
            _record_runtime_phase("tts_warmup.start")
            tts.warm()
            _record_runtime_phase("tts_warmup.started")
        except Exception:
            _record_runtime_phase("tts_warmup.failed")
            logger.exception("TTS warmup failed")

    def _show_startup_status(self, text: str) -> None:
        overlay = getattr(self, "_overlay", None)
        if overlay is None:
            return
        overlay.show()
        overlay.set_text(text)

    def _hide_startup_status(self) -> None:
        overlay = getattr(self, "_overlay", None)
        if overlay is None:
            return
        overlay.hide()

    def _refresh_startup_status(self) -> None:
        if getattr(self, "_warm_error", None) is not None:
            self._show_startup_status(
                "Model load failed.\nChoose another model from the menu."
            )
            if self._menubar is not None:
                self._menubar.set_status_text(
                    "Model load failed — choose another model"
                )
            return

        self._show_startup_status(
            "Loading models...\nFirst launch may download selected models."
        )
        if self._menubar is not None:
            self._menubar.set_status_text("Loading models…")

    def retryEventTap_(self, timer) -> None:
        """Retry event tap installation."""
        if self._detector.install():
            self._complete_event_tap_startup()
        else:
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryEventTap:", None, False
            )

    # ── hold callbacks (called on main thread) ──────────────

    def _on_cancel_spring_start(self) -> None:
        """Called from the event tap when enter goes down while space is
        held and the command overlay is active.  Activates the cancel
        spring if generation is in progress."""
        if not self._transcribing:
            return
        if getattr(self, '_cancel_spring_active', False):
            return  # already winding — ignore key repeat
        logger.info("Cancel spring activated (enter added to hold during active generation)")
        self._cancel_spring_active = True
        self._cancel_spring_start = time.monotonic()
        self._detector.cancel_spring_active = True  # capture gesture in input tap
        overlay = self._command_overlay
        if overlay is not None:
            overlay.set_cancel_spring(1.0)
        else:
            logger.warning("Cancel spring: no command overlay available")

    def _on_cancel_spring_threshold(self) -> None:
        """Called from the overlay pulse tick when the spring crosses the
        cancel threshold.  Fires immediately — no waiting for key release."""
        if not getattr(self, '_cancel_spring_active', False):
            return
        logger.info(
            "Cancel spring threshold crossed — cancelling generation (token %d)",
            self._transcription_token,
        )
        self._cancel_spring_active = False
        self._detector.cancel_spring_active = False
        self._transcription_token += 1
        self._transcribing = False
        if self._menubar is not None:
            self._menubar.set_status_text("Cancelled")

    def _on_cancel_spring_release(self) -> None:
        """Called from the event tap when either key is released while
        the cancel spring is active.  If the threshold already fired,
        this is a no-op (just clean up).  If released early, snap back."""
        if not getattr(self, '_cancel_spring_active', False):
            return
        # Threshold already fired — just clean up state
        self._cancel_spring_active = False
        self._detector.cancel_spring_active = False
        elapsed = time.monotonic() - self._cancel_spring_start
        logger.info(
            "Cancel spring released at %.0fms — below threshold, snapping back",
            elapsed * 1000,
        )
        # Snap back visually (the overlay ease-out handles the animation)
        if self._command_overlay is not None:
            self._command_overlay.set_cancel_spring(0.0)

    def _on_hold_start(self) -> None:
        if not getattr(self, "_models_ready", True):
            logger.warning("Hold started before models were ready — ignoring")
            self._hold_rejected_during_warmup = True
            self._refresh_startup_status()
            return
        if not getattr(self, "_mic_ready", False):
            logger.warning("Hold started but mic not yet available — ignoring")
            if self._menubar is not None:
                self._menubar.set_status_text("Mic unavailable — retrying…")
            return
        # If TTS is playing from a tool call, cancel the playback but
        # don't invalidate the stream token — let the remaining tool batch
        # finish so subsequent read_aloud calls still execute.
        tts = getattr(self, "_tts_client", None)
        tts_playing = tts is not None and (
            getattr(tts, "_playback_active", False)
            or getattr(tts, "_stream", None) is not None
        )
        # ── Cancel spring: space+enter hold during active generation ──
        # If the spring was already activated by the enter-during-hold
        # callback, don't cancel the stream or start recording.
        if getattr(self, '_cancel_spring_active', False):
            logger.info("Hold timer fired while cancel spring active — suppressing recording")
            return

        # Don't cancel immediately — start the visual spring wind-up.
        # The cancel fires on release if the spring is past threshold.
        enter_held = getattr(self._detector, '_enter_held', False) is True
        if self._transcribing and enter_held and not tts_playing:
            logger.info("Space+enter hold during active generation — starting cancel spring")
            self._cancel_spring_active = True
            self._cancel_spring_start = time.monotonic()
            self._detector.cancel_spring_active = True
            if self._command_overlay is not None:
                self._command_overlay.set_cancel_spring(1.0)
            return  # don't start recording

        if self._transcribing and tts_playing:
            logger.info("Hold during TTS playback — cancelling audio, keeping stream alive")
            tts.cancel()
            # Fall through to start recording (generation continues)
        # Keep the detector flag aligned with the overlay's real visible state.
        overlay_visible = (
            self._command_overlay is not None
            and getattr(self._command_overlay, "_visible", False)
        )
        self._detector.command_overlay_active = overlay_visible
        logger.info("command_overlay_active -> %s (hold start)", overlay_visible)
        # Note: if command overlay is visible but finished, leave it up.
        # It will be dismissed if the user says nothing (empty recording)
        # or replaced if they send a new command.

        # Tray intercept: shift+space from tray = navigation, plain space = record.
        self._verify_paste_text = None
        if getattr(self, "_tray_active", False):
            shift_at_press = getattr(self._detector, '_shift_at_press', False)
            logger.info("Tray hold: shift_at_press=%s, shift_latched=%s",
                         shift_at_press, getattr(self._detector, '_shift_latched', 'N/A'))
            if shift_at_press or getattr(self._detector, '_shift_latched', False):
                # Shift+space hold during tray = navigation gesture, not recording.
                # Wait for release to route through _on_hold_end as navigate up.
                self._recovery_hold_active = True
                logger.info("Hold started during tray with shift — waiting for release (navigate)")
                return
            # Plain spacebar hold during tray = dismiss tray, start recording
            logger.info("Hold started during tray — dismissing tray, starting new recording")
            self._dismiss_tray()
            # Fall through to start recording
        elif getattr(self, "_recovery_text", None) is not None:
            self._recovery_hold_active = True
            logger.info("Hold started during recovery — waiting for release")
            return
        self._recovery_hold_active = False

        shift_at_press = getattr(self._detector, '_shift_at_press', False)
        logger.info(
            "Hold started — recording (shift_at_press=%s tray_active=%s recovery_active=%s recovery_hold_active=%s verify_pending=%s overlay_visible=%s)",
            shift_at_press,
            getattr(self, "_tray_active", False),
            getattr(self, "_recovery_text", None) is not None,
            getattr(self, "_recovery_hold_active", False),
            getattr(self, "_verify_paste_text", None) is not None,
            getattr(self._overlay, "_visible", False) if self._overlay is not None else False,
        )
        # Pause hands-free dictation if it's active — spacebar hold takes priority.
        self._handsfree_paused_for_hold = False
        hf = getattr(self, "_handsfree", None)
        if hf is not None and hf.is_dictating:
            logger.info("Pausing hands-free for spacebar hold")
            hf._stop_dictation_capture()
            hf._set_state(HandsFreeState.LISTENING)
            self._handsfree_paused_for_hold = True

        if self._menubar is not None:
            self._menubar.set_recording(True)
            self._menubar.set_status_text("Recording…")
        # Set up opportunistic segment transcription for remote backends.
        # Each silence-bounded segment is dispatched to the final client as it
        # arrives, so that on release we only need to transcribe the tail.
        self._segment_accumulator = SegmentAccumulator()
        use_segments = getattr(self, "_whisper_backend", "local") in ("sidecar", "cloud")
        segment_cb = None
        if use_segments:
            def segment_cb(wav_bytes: bytes):
                self._segment_accumulator.dispatch(wav_bytes, self._client)

        if self._glow is not None:
            self._glow.show()
        if self._overlay is not None:
            if self._glow is not None:
                self._overlay.set_brightness(
                    getattr(self._glow, "_brightness", 0.0),
                    immediate=True,
                )
            self._overlay.show()

        try:
            def on_vad_state(is_speech: bool):
                if self._menubar is not None:
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "updateVadState:", is_speech, False
                    )
            self._capture.start(
                amplitude_callback=self._on_amplitude,
                vad_state_callback=on_vad_state,
                segment_callback=segment_cb,
            )
        except Exception:
            logger.exception("Audio capture failed to start")
            if self._glow is not None:
                self._glow.hide()
            if self._overlay is not None:
                self._overlay.flash_notice(
                    "Audio unavailable — system under memory pressure.\n"
                    "Free memory or use sidecar for transcription.",
                    hold=4.0,
                    fade=2.0,
                )
            if self._menubar is not None:
                self._menubar.set_recording(False)
                self._menubar.set_status_text(
                    "Audio unavailable — memory pressure"
                )
            return
        self._record_start_time = time.monotonic()
        self._cap_fired = False
        self._last_preview_text = ""
        self._preview_cancelled_on_release = False
        self._preview_session_token = getattr(self, "_preview_session_token", 0) + 1
        self._is_speech = False
        self._force_preview_update = False
        if getattr(self, "_preview_done", None) is not None:
            self._preview_done.clear()

        # Start the adaptive preview loop
        self._preview_active = True
        token = self._preview_session_token
        self._preview_thread = threading.Thread(
            target=self._preview_loop, args=(token,), daemon=True
        )
        self._preview_thread.start()

    def _on_amplitude(self, rms: float) -> None:
        """Called from PortAudio thread — marshal to main thread."""
        from Foundation import NSNumber
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "amplitudeUpdate:", NSNumber.numberWithFloat_(rms), False
        )

    def updateVadState_(self, is_speech_number) -> None:
        is_speech = bool(is_speech_number)

        # If transitioning from speech to silence, force one last preview update
        if getattr(self, "_is_speech", False) and not is_speech:
            self._force_preview_update = True

        self._is_speech = is_speech

        if self._menubar is not None:
            self._menubar.set_vad_state(is_speech, self._transcribing or self._capture.is_recording)

    def amplitudeUpdate_(self, rms_number) -> None:
        """Main thread: forward amplitude to glow and overlay text."""
        rms = float(rms_number)
        
        # VAD gating: if we are in silence, clamp the raw RMS to 0.0.
        # This prevents background noise (keyboard clacks, rustling) from 
        # driving the visual glow and text breathing when speech isn't detected.
        # The glow system's own `_DECAY_FACTOR` will provide a smooth "ease out"
        # as it falls from its last speech value to zero.
        if not getattr(self, "_is_speech", True):
            rms = 0.0

        # Recording cap: check elapsed time on every amplitude tick
        if self._local_mode and _MAX_RECORD_SECS is not None and not self._cap_fired:
            elapsed = time.monotonic() - self._record_start_time
            _CAP_WARN_SECS = 3.0

            # Update countdown glow in the last 3 seconds
            warn_start = _MAX_RECORD_SECS - _CAP_WARN_SECS
            if elapsed >= warn_start and self._glow is not None:
                progress = min((elapsed - warn_start) / _CAP_WARN_SECS, 1.0)
                self._glow._cap_factor = 1.0 - progress  # linear

            # Fire the cap
            if elapsed >= _MAX_RECORD_SECS:
                self._cap_fired = True
                logger.warning(
                    "Recording capped at %.0fs (%.0fGB RAM) — transcribing what we have",
                    _MAX_RECORD_SECS, _RAM_GB,
                )
                self._detector.force_end()
                return

        if self._glow is not None:
            self._glow.update_amplitude(rms)
        if self._overlay is not None and self._glow is not None:
            glow_brightness = getattr(self._glow, "_brightness", 0.0)
            if glow_brightness != getattr(self._overlay, "_brightness", 0.0):
                self._overlay.set_brightness(glow_brightness)
            self._sync_command_overlay_brightness()
            # Text breathing: glow's smoothed amplitude, scaled independently
            self._overlay.update_text_amplitude(
                min(self._glow._smoothed_amplitude * 18.0, 1.0)
            )
            # Inner glow: track screen glow, saturate aggressively
            glow_opacity = self._glow._glow_layer.opacity()
            self._overlay.update_glow_amplitude(
                min(glow_opacity * 2.5, 1.0),
                cap_factor=self._glow._cap_factor,
            )

    def _preview_loop(self, token: int | None = None) -> None:
        """Background thread: preview transcription during recording.

        Uses streaming (KV-cache reuse) when the client supports it,
        falling back to full-buffer re-transcription otherwise.
        """
        use_streaming = getattr(self._preview_client, 'supports_streaming', False)

        if use_streaming:
            self._preview_loop_streaming(token)
        else:
            self._preview_loop_batch(token)

    def _preview_loop_streaming(self, token: int | None = None) -> None:
        """Streaming preview: feed incremental audio chunks, read state.text."""
        _FEED_INTERVAL = 0.15  # feed new audio every 150ms
        token = getattr(self, "_preview_session_token", 0) if token is None else token
        delegated_to_batch = False

        try:
            # Wait for some audio to accumulate before first feed
            time.sleep(0.4)

            try:
                with self._local_inference_context(self._preview_client):
                    self._preview_client.start_stream()
            except Exception:
                logger.exception("Failed to start streaming — falling back to batch")
                delegated_to_batch = True
                self._preview_loop_batch(token)
                return

            while self._preview_active:
                loop_start = time.monotonic()

                frames = self._capture.get_new_frames()

                is_speech = getattr(self, "_is_speech", True)
                force_update = getattr(self, "_force_preview_update", False)
                if not is_speech and not force_update:
                    elapsed = time.monotonic() - loop_start
                    remaining = _FEED_INTERVAL - elapsed
                    if remaining > 0 and self._preview_active:
                        time.sleep(remaining)
                    continue
                self._force_preview_update = False

                try:
                    with self._local_inference_context(self._preview_client):
                        text = self._preview_client.feed(frames)
                except Exception:
                    logger.debug("Streaming feed failed", exc_info=True)
                    time.sleep(0.5)
                    continue

                if text and self._preview_active:
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "previewTextUpdate:", {"token": token, "text": text}, False
                    )

                elapsed = time.monotonic() - loop_start
                remaining = _FEED_INTERVAL - elapsed
                if remaining > 0 and self._preview_active:
                    time.sleep(remaining)
        finally:
            if not delegated_to_batch:
                # Clean up streaming state if preview client differs from final client,
                # so the upstream library doesn't hold stale KV cache / session state.
                if (
                    self._preview_client is not self._client
                    and getattr(self._preview_client, "has_active_stream", False)
                ):
                    try:
                        if getattr(self, "_preview_cancelled_on_release", False):
                            cancel_stream = getattr(self._preview_client, "cancel_stream", None)
                            if callable(cancel_stream):
                                cancel_stream()
                        else:
                            self._preview_client.finish_stream()
                    except Exception:
                        logger.debug("Preview stream cleanup failed", exc_info=True)
                if getattr(self, "_preview_done", None) is not None:
                    self._preview_done.set()

    def _preview_batch_intervals(self) -> tuple[float, float]:
        """Return (min_interval, initial_delay) for batch preview by backend."""
        backend = getattr(self, "_preview_backend", "local")
        if backend == "cloud":
            return (3.0, 1.0)
        if backend == "sidecar":
            return (0.75, 0.3)
        return (0.2, 0.15)

    def _preview_loop_batch(self, token: int | None = None) -> None:
        """Batch preview: re-transcribe the full buffer each tick.

        When the segment accumulator is active, only the tail audio (since the
        last segment boundary) is sent to the preview client; cached segment
        text is prepended to the result.
        """
        _MIN_INTERVAL, _INITIAL_DELAY = self._preview_batch_intervals()
        token = getattr(self, "_preview_session_token", 0) if token is None else token

        try:
            time.sleep(_INITIAL_DELAY)

            while self._preview_active:
                loop_start = time.monotonic()

                acc = getattr(self, "_segment_accumulator", None)
                use_segments = acc is not None and acc.count > 0

                if use_segments:
                    wav_bytes = self._capture.get_tail_buffer()
                    cached_prefix = acc.text
                else:
                    wav_bytes = self._capture.get_buffer()
                    cached_prefix = ""

                if not wav_bytes and not cached_prefix:
                    time.sleep(min(_MIN_INTERVAL, 0.2))
                    continue

                is_speech = getattr(self, "_is_speech", True)
                force_update = getattr(self, "_force_preview_update", False)
                if not is_speech and not force_update:
                    elapsed = time.monotonic() - loop_start
                    remaining = _MIN_INTERVAL - elapsed
                    if remaining > 0 and self._preview_active:
                        time.sleep(remaining)
                    continue
                self._force_preview_update = False

                try:
                    if wav_bytes:
                        with self._local_inference_context(self._preview_client):
                            tail_text = self._preview_client.transcribe(wav_bytes)
                    else:
                        tail_text = ""
                except Exception:
                    logger.debug("Preview transcription failed", exc_info=True)
                    time.sleep(0.5)
                    continue

                if use_segments:
                    parts = [p for p in (cached_prefix, tail_text) if p]
                    text = " ".join(parts)
                else:
                    text = tail_text

                if text and self._preview_active:
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "previewTextUpdate:",
                        {"token": token, "text": text},
                        False,
                    )

                elapsed = time.monotonic() - loop_start
                remaining = _MIN_INTERVAL - elapsed
                if remaining > 0 and self._preview_active:
                    time.sleep(remaining)
        finally:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.set()

    def previewTextUpdate_(self, payload) -> None:
        """Main thread: update the overlay with preview transcription text."""
        if isinstance(payload, dict):
            token = payload.get("token")
            text = payload.get("text", "")
        else:
            token = None
            text = payload
        if token is not None and token != getattr(self, "_preview_session_token", token):
            logger.info(
                "Dropping stale preview update (token=%s current=%s len=%d)",
                token,
                getattr(self, "_preview_session_token", None),
                len(text),
            )
            return
        if not self._preview_active:
            logger.info(
                "Dropping preview update while preview inactive (token=%s current=%s len=%d)",
                token,
                getattr(self, "_preview_session_token", None),
                len(text),
            )
            return
        self._last_preview_text = text
        if self._overlay is not None:
            self._overlay.set_text(text)

    def _on_hold_end(
        self,
        shift_held: bool = False,
        enter_held: bool = False,
        **_kwargs,  # absorb legacy toggle_command_overlay from any remaining callers
    ) -> None:
        if getattr(self, "_hold_rejected_during_warmup", False):
            self._hold_rejected_during_warmup = False
            logger.info(
                "Hold ended before models were ready — keeping startup status"
            )
            if not getattr(self, "_models_ready", True):
                self._refresh_startup_status()
            return

        # ── Tray intercept ──
        # When the tray is active, gestures route through the tray handler.
        tray_active = getattr(self, "_tray_active", False)
        recovery_active = getattr(self, "_recovery_text", None) is not None
        if tray_active or recovery_active or getattr(self, "_recovery_hold_active", False):
            self._recovery_hold_active = False
            if shift_held:
                # Shift held + release spacebar from tray = navigate down (older)
                logger.info("Shift+space during tray — navigate down")
                self._tray_navigate_down()
            elif enter_held and tray_active:
                logger.info("Enter-first release during tray — sending current tray entry")
                self._tray_send_current()
            elif tray_active:
                # Spacebar from tray (tap or hold release) = insert
                logger.info("Spacebar during tray — inserting text")
                self._tray_insert_current()
            else:
                # Spacebar from recovery (non-tray) = retry insert
                logger.info("Spacebar during recovery — retrying Insert")
                self._recovery_retry_insert()
            return

        # ── Normal recording end ──
        logger.info("Hold ended — shift=%s enter=%s", shift_held, enter_held)
        self._preview_active = False
        self._preview_cancelled_on_release = True
        # Capture the tail buffer BEFORE stop() clears segment state.
        # After stop(), get_tail_buffer() falls through to get_buffer() (the
        # entire recording) because _segment_cb is None — causing the final
        # text to contain cached-segments + full-audio ≈ doubled output.
        # Also snapshot the segment count so the worker can detect whether
        # stop() flushed an extra final segment (which overlaps the tail).
        acc = getattr(self, "_segment_accumulator", None)
        if acc is not None and acc.count > 0:
            self._pre_stop_tail_wav = self._capture.get_tail_buffer()
            self._pre_stop_segment_count = acc.count
        else:
            self._pre_stop_tail_wav = None
            self._pre_stop_segment_count = 0
        wav_bytes = self._capture.stop()

        # Short shift-hold (under 800ms of recording) = recall into tray
        elapsed = time.monotonic() - self._record_start_time if self._record_start_time else 0
        if shift_held and elapsed < 0.8:
            logger.info("Short shift-hold (%.0fms) — recalling into tray", elapsed * 1000)
            wav_bytes = b""  # force the empty-audio path

        # Glow/dimmer: hide immediately for text insertion
        if not shift_held and not enter_held and self._glow is not None:
            self._glow.hide()
        if self._menubar is not None:
            self._menubar.set_vad_state(False, False)
            self._menubar.set_recording(False)

        if not wav_bytes:
            logger.info(
                "No audio — instant path (shift=%s, enter=%s)",
                shift_held,
                enter_held,
            )
            if self._overlay is not None:
                self._overlay.hide()
            if self._glow is not None:
                self._glow.hide()
            if shift_held:
                if self._tray_stack:
                    logger.info("Shift+empty — recalling tray (stack has %d entries)", len(self._tray_stack))
                    self._tray_active = True
                    self._detector.tray_active = True
                    self._tray_index = len(self._tray_stack) - 1
                    self._show_tray_current(acknowledge=True)
                    return
                else:
                    logger.info("Shift+empty — no tray entries to recall")
            elif enter_held and self._command_client is not None:
                logger.info("Enter held on empty — assistant path with no utterance")
            else:
                logger.info("Empty tap — no action")

            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")
            return

        if self._transcribing and not shift_held and not enter_held:
            self._parallel_insert_token += 1
            parallel_token = self._parallel_insert_token
            logger.info(
                "Plain hold during active turn — transcribing on parallel insert lane (token %d)",
                parallel_token,
            )
            if self._overlay is not None:
                self._overlay.hide()
            if self._glow is not None:
                self._glow.hide()
            thread = threading.Thread(
                target=self._parallel_insert_worker,
                args=(wav_bytes, parallel_token),
                daemon=True,
            )
            thread.start()
            return
        # Invalidate any in-flight transcription so its result is discarded
        self._transcription_token += 1
        token = self._transcription_token

        self._transcribing = True
        self._transcribe_start = time.monotonic()

        if enter_held and self._command_client is not None:
            # Command pathway (enter held): transcribe then send to OMLX
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing command…")
            thread = threading.Thread(
                target=self._command_transcribe_worker,
                args=(wav_bytes, token),
                daemon=True,
            )
        elif shift_held:
            # Tray pathway: transcribe, then enter tray with the result.
            # Don't set _tray_active yet — defer until _enter_tray is called
            # from trayTranscriptionComplete_. Setting it prematurely would
            # let gestures fire on stale/empty stack state during transcription.
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing…")
            thread = threading.Thread(
                target=self._tray_transcribe_worker,
                args=(wav_bytes, token),
                daemon=True,
            )
        else:
            # Text pathway: transcribe and paste
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing…")
            thread = threading.Thread(
                target=self._transcribe_worker, args=(wav_bytes, token), daemon=True
            )
        thread.start()

    def _transcribe_segments_and_tail(self, wav_bytes: bytes) -> str | None:
        """Try segment-accelerated transcription.  Returns final text, or None
        to signal the caller should fall back to full-buffer transcription."""
        acc = getattr(self, "_segment_accumulator", None)
        if acc is None or acc.count == 0:
            return None

        # Use the tail buffer captured before stop() cleared segment state.
        # After stop(), capture.get_tail_buffer() falls through to get_buffer()
        # (the entire recording) because _segment_cb is already None — which
        # would produce cached-segments + full-audio ≈ doubled output.
        pre_stop_tail = getattr(self, "_pre_stop_tail_wav", None)
        self._pre_stop_tail_wav = None

        # Check whether stop() flushed the final in-progress speech as an
        # extra segment.  If the accumulator count grew beyond what was
        # dispatched during recording, the tail audio is already covered.
        pre_stop_count = getattr(self, "_pre_stop_segment_count", 0)
        self._pre_stop_segment_count = 0

        # Wait for any in-flight segment transcriptions to land.
        # Cloud endpoints can take >10s on congested connections.
        if not acc.wait(timeout=30.0):
            logger.warning(
                "Segment accumulator timed out with %d pending — falling back to full buffer",
                acc._pending,
            )
            return None
        cached = acc.text

        # If stop() flushed the final segment (count grew after we captured
        # the tail), that audio is already in the accumulator — skip tail.
        final_flushed = acc.count > pre_stop_count

        # Transcribe the tail — audio recorded after the last segment boundary.
        tail_wav = pre_stop_tail if pre_stop_tail is not None else b""
        tail_text = ""
        if tail_wav and not final_flushed:
            try:
                tail_text = self._client.transcribe(tail_wav)
            except Exception:
                logger.exception("Tail transcription failed — falling back to full buffer")
                return None

        parts = [p for p in (cached, tail_text) if p]
        text = " ".join(parts)
        logger.info(
            "Segment-accelerated transcription: %d segments cached, tail=%d bytes, "
            "final_flushed=%s, result=%r",
            acc.count, len(tail_wav) if tail_wav else 0, final_flushed, text[:80],
        )
        return text

    def _transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: finalize transcription and marshal result to main thread."""
        release_cutover = getattr(self, "_preview_cancelled_on_release", False)

        # Normally we wait for preview shutdown before the final pass. On release
        # cutover, stop blocking on preview wind-down and let the local inference
        # lock serialize any in-flight work that still needs to exit.
        if self._preview_thread is not None and not release_cutover:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            # Fast path: use cached segment transcriptions + tail only.
            text = self._transcribe_segments_and_tail(wav_bytes)
            if text is None:
                # Slow path: full-buffer transcription.
                with self._local_inference_context(self._client):
                    # If the preview client was streaming, finalize it for the final text
                    # (this runs the tail refinement pass with the existing KV cache).
                    if (
                        release_cutover
                        and getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        cancel_stream = getattr(self._client, "cancel_stream", None)
                        if callable(cancel_stream):
                            cancel_stream()
                        text = self._client.transcribe(wav_bytes)
                    elif (
                        getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        text = self._client.finish_stream()
                    else:
                        text = self._client.transcribe(wav_bytes)
        except Exception:
            logger.exception("Transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "transcriptionFailed:", {"token": token}, False
            )
            return

        elapsed_ms = (time.monotonic() - self._transcribe_start) * 1000
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "transcriptionComplete:",
            {"token": token, "text": text, "elapsed_ms": elapsed_ms},
            False,
        )

    def _parallel_insert_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: transcribe a plain-space recording without disturbing
        an active assistant turn."""
        release_cutover = getattr(self, "_preview_cancelled_on_release", False)

        if self._preview_thread is not None and not release_cutover:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            text = self._transcribe_segments_and_tail(wav_bytes)
            if text is None:
                with self._local_inference_context(self._client):
                    if (
                        release_cutover
                        and getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        cancel_stream = getattr(self._client, "cancel_stream", None)
                        if callable(cancel_stream):
                            cancel_stream()
                        text = self._client.transcribe(wav_bytes)
                    elif (
                        getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        text = self._client.finish_stream()
                    else:
                        text = self._client.transcribe(wav_bytes)
        except Exception:
            logger.exception("Parallel insert transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "parallelTranscriptionFailed:", {"token": token}, False
            )
            return

        elapsed_ms = (time.monotonic() - self._transcribe_start) * 1000
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "parallelTranscriptionComplete:",
            {"token": token, "text": text, "elapsed_ms": elapsed_ms},
            False,
        )

    _INSERT_GRACE_S = 0.35  # grace window before auto-insert after transcription

    def transcriptionComplete_(self, payload: dict) -> None:
        """Main thread: inject transcribed text at cursor (with grace window)."""
        if payload["token"] != self._transcription_token:
            logger.info("Discarding stale transcription (token %d)", payload["token"])
            return
        self._transcribing = False
        text = payload["text"]
        if text:
            elapsed_ms = payload.get("elapsed_ms", 0)
            logger.info("Transcribed: %r (%.0fms) — starting insert grace window", text, elapsed_ms)
            self._grace_pending_text = text
            # Arm the Enter-cancels-insert callback on the detector
            self._detector._on_enter_cancel_grace = self._cancel_grace_insert
            # Start shrink wind-up animation
            if self._overlay is not None:
                self._overlay.start_insert_windup()
            # Start grace timer — Enter during this window cancels the insert
            from Foundation import NSTimer
            self._grace_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                self._INSERT_GRACE_S, self, "graceTimerFired:", None, False
            )
            return
        if self._overlay is not None:
            self._overlay.hide()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def parallelTranscriptionComplete_(self, payload: dict) -> None:
        """Main thread: inject a parallel plain-space transcription at cursor."""
        if payload["token"] != self._parallel_insert_token:
            logger.info("Discarding stale parallel transcription (token %d)", payload["token"])
            return
        text = payload["text"]
        if text:
            elapsed_ms = payload.get("elapsed_ms", 0)
            logger.info(
                "Parallel transcription: %r (%.0fms) — starting insert grace window",
                text,
                elapsed_ms,
            )
            self._grace_pending_text = text
            self._detector._on_enter_cancel_grace = self._cancel_grace_insert
            if self._overlay is not None:
                self._overlay.start_insert_windup()
            from Foundation import NSTimer
            self._grace_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                self._INSERT_GRACE_S, self, "graceTimerFired:", None, False
            )

    def graceTimerFired_(self, timer) -> None:
        """Grace window expired — proceed with insert."""
        self._grace_timer = None
        self._detector._on_enter_cancel_grace = None
        text = getattr(self, "_grace_pending_text", None)
        self._grace_pending_text = None
        if text:
            logger.info("Grace window expired — injecting: %r", text)
            self._inject_result_text(text, "Pasted!")

    def _cancel_grace_insert(self) -> None:
        """Cancel a pending grace-window insert (Enter arrived during window)."""
        if getattr(self, "_grace_timer", None) is not None:
            self._grace_timer.invalidate()
            self._grace_timer = None
        self._detector._on_enter_cancel_grace = None
        text = getattr(self, "_grace_pending_text", None)
        self._grace_pending_text = None
        if text:
            logger.info("Grace insert cancelled — redirecting to overlay toggle")
            # Add to tray so the transcription isn't lost
            self._add_tray_entry(text, owner="user", activate=False)
        if self._overlay is not None:
            self._overlay.cancel_insert_windup()
        self._toggle_command_overlay()

    def transcriptionFailed_(self, payload: dict) -> None:
        """Main thread: handle transcription error."""
        if payload["token"] != self._transcription_token:
            return  # stale failure, ignore
        self._transcribing = False
        if self._last_preview_text:
            logger.warning(
                "Final transcription failed — falling back to latest preview text"
            )
            self._inject_result_text(self._last_preview_text, "Pasted preview")
            return
        logger.error("Transcription failed — no text injected")
        if self._overlay is not None:
            self._overlay.hide()
        if self._menubar is not None:
            self._menubar.set_status_text("Error — try again")

    def parallelTranscriptionFailed_(self, payload: dict) -> None:
        """Main thread: handle failure on the parallel insert lane."""
        if payload["token"] != self._parallel_insert_token:
            return
        if self._last_preview_text:
            logger.warning("Parallel transcription failed — falling back to latest preview text")
            self._inject_result_text(self._last_preview_text, "Pasted preview")

    def hideOverlayAfterInject_(self, timer) -> None:
        """Hide the overlay after briefly showing the final transcription."""
        if self._overlay is not None:
            self._overlay.hide()

    def _last_command_overlay_snapshot(self) -> tuple[str, str] | None:
        """Return the most recent assistant overlay content, including failures."""
        utterance = getattr(self, "_last_command_utterance", "")
        response = getattr(self, "_last_command_response", "")
        if self._command_client is not None:
            history = self._command_client.history
            if history:
                hist_utterance, hist_response = history[-1]
                if (
                    utterance
                    and response
                    and hist_utterance == utterance
                    and len(response) >= len(hist_response)
                ):
                    return utterance, response
                return hist_utterance, hist_response
        if utterance and response:
            return utterance, response
        return None

    def _recallLastResponse_(self, payload) -> None:
        """Main thread: recall the last command/response from history."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        if self._overlay is not None:
            self._overlay.hide()
        if self._glow is not None:
            self._glow.hide()

        snapshot = self._last_command_overlay_snapshot()
        if snapshot is not None:
            last_utterance, last_response = snapshot
            logger.info("Recalling last response: %r", last_utterance[:50])
            if self._command_overlay is not None:
                self._sync_command_overlay_brightness(immediate=True)
                self._command_overlay.show()
                self._command_overlay.set_utterance(last_utterance)
                for ch in last_response:
                    self._command_overlay.append_token(ch)
                self._command_overlay.finish()
                self._detector.command_overlay_active = True
                logger.info("command_overlay_active -> True (shift recall)")
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")
            return

        logger.info("No history to recall")
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def _sync_command_overlay_brightness(self, immediate: bool = False) -> None:
        if self._command_overlay is None or self._glow is None:
            return
        self._command_overlay.set_brightness(
            getattr(self._glow, "_brightness", 0.0),
            immediate=immediate,
        )

    def _resetStatusAfterCancel_(self, timer) -> None:
        """Reset menubar status after a cancel."""
        if self._menubar is not None and not self._transcribing:
            self._menubar.set_status_text("Ready — hold spacebar")

    # ── tray ───────────────────────────────────────────────

    def _tray_transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: transcribe audio, then enter tray on main thread."""
        release_cutover = getattr(self, "_preview_cancelled_on_release", False)

        if self._preview_thread is not None and not release_cutover:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            # Fast path: use cached segment transcriptions + tail only.
            text = self._transcribe_segments_and_tail(wav_bytes)
            if text is None:
                with self._local_inference_context(self._client):
                    if (
                        release_cutover
                        and getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        cancel_stream = getattr(self._client, "cancel_stream", None)
                        if callable(cancel_stream):
                            cancel_stream()
                        text = self._client.transcribe(wav_bytes)
                    elif (
                        getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        text = self._client.finish_stream()
                    else:
                        text = self._client.transcribe(wav_bytes)
        except Exception:
            logger.exception("Tray transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "trayTranscriptionFailed:", {"token": token}, False
            )
            return

        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "trayTranscriptionComplete:",
            {"token": token, "text": text},
            False,
        )

    def trayTranscriptionComplete_(self, payload: dict) -> None:
        """Main thread: transcription done — enter tray with the text."""
        if payload["token"] != self._transcription_token:
            logger.info("Discarding stale tray transcription (token %d)", payload["token"])
            return
        self._transcribing = False
        text = payload["text"]
        if not text:
            # Empty transcription — use last preview text if available
            if self._last_preview_text:
                logger.info("Tray transcription empty — using last preview text")
                text = self._last_preview_text
            else:
                logger.info("Tray transcription returned empty — dismissing")
                if self._overlay is not None:
                    self._overlay.hide()
                if self._glow is not None:
                    self._glow.hide()
                self._tray_active = False
                self._detector.tray_active = False
                if self._menubar is not None:
                    self._menubar.set_status_text("Ready — hold spacebar")
                return
        self._enter_tray(text)

    def trayTranscriptionFailed_(self, payload: dict) -> None:
        """Main thread: tray transcription failed — fall back to preview text."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        if self._last_preview_text:
            logger.warning("Tray transcription failed — using preview text")
            self._enter_tray(self._last_preview_text)
            return
        logger.error("Tray transcription failed — no text")
        if self._overlay is not None:
            self._overlay.hide()
        if self._glow is not None:
            self._glow.hide()
        self._tray_active = False
        self._detector.tray_active = False
        if self._menubar is not None:
            self._menubar.set_status_text("Error — try again")

    def _enter_tray(self, text: str) -> None:
        """Enter the tray with new text, pushing it onto the stack."""
        self._add_tray_entry(text, owner="user", activate=True)

    def _coerce_tray_entry(self, entry: TrayEntry | str) -> TrayEntry:
        if isinstance(entry, TrayEntry):
            return entry
        return TrayEntry(text=str(entry))

    def _get_tray_entry(self, index: int) -> TrayEntry:
        entry = self._coerce_tray_entry(self._tray_stack[index])
        self._tray_stack[index] = entry
        return entry

    def _acknowledge_tray_entry(self, index: int) -> None:
        if index < 0 or index >= len(self._tray_stack):
            return
        entry = self._get_tray_entry(index)
        if entry.owner == "assistant" and not entry.acknowledged:
            entry.acknowledged = True

    def _add_tray_entry(
        self,
        text: str,
        *,
        owner: str = "user",
        activate: bool = True,
        position: str = "top",
    ) -> TrayEntry:
        entry = TrayEntry(
            text=text,
            owner=owner,
            acknowledged=(owner != "assistant"),
        )
        if position == "bottom":
            had_entries = bool(self._tray_stack)
            self._tray_stack.insert(0, entry)
            if activate or not had_entries:
                self._tray_index = 0
            elif self._tray_active:
                self._tray_index += 1
        else:
            self._tray_stack.append(entry)
            self._tray_index = len(self._tray_stack) - 1

        if activate:
            self._tray_active = True
            self._detector.tray_active = True
            logger.info(
                "Entering tray (entries=%d index=%d text_len=%d)",
                len(self._tray_stack),
                self._tray_index,
                len(text),
            )

            if self._glow is not None:
                if hasattr(self._glow, "show_tray_dim"):
                    self._glow.show_tray_dim()
                else:
                    self._glow.hide()

            self._show_tray_current()

        return entry

    def _stash_failed_paste_to_tray(self, text: str, *, status: str) -> None:
        """Preserve an unverified paste without surfacing the tray UI."""
        entry = self._add_tray_entry(
            text,
            owner="user",
            activate=False,
            position="bottom",
        )
        if self._overlay is not None:
            flash = getattr(self._overlay, "flash_tray_capture", None)
            if callable(flash):
                flash(text, owner=entry.display_owner)
        if self._menubar is not None:
            self._menubar.set_status_text("Saved to tray")
        logger.info(
            "Stashed unverified paste at bottom of tray (status=%s entries=%d)",
            status,
            len(self._tray_stack),
        )

    def _show_tray_current(self, *, acknowledge: bool = False) -> None:
        """Update the tray overlay to display the current stack entry."""
        if not self._tray_stack:
            self._dismiss_tray()
            return
        # Defensive bounds clamp
        if self._tray_index >= len(self._tray_stack):
            self._tray_index = len(self._tray_stack) - 1
        if acknowledge:
            self._acknowledge_tray_entry(self._tray_index)
        entry = self._get_tray_entry(self._tray_index)
        text = entry.text
        # Set recovery_text for compatibility with existing dismiss/cleanup
        self._recovery_text = text
        self._recovery_clipboard_state = "idle"
        if self._overlay is not None:
            self._overlay.show_tray(text, owner=entry.display_owner)
        if self._menubar is not None:
            pos = f"{self._tray_index + 1}/{len(self._tray_stack)}"
            self._menubar.set_status_text(f"Tray [{pos}]")

    def _dismiss_tray(self) -> None:
        """Dismiss the tray overlay. Stack is preserved for re-entry."""
        logger.info(
            "Dismissing tray (entries=%d index=%d recovery_active=%s)",
            len(self._tray_stack),
            self._tray_index,
            getattr(self, "_recovery_text", None) is not None,
        )
        self._tray_active = False
        self._detector.tray_active = False
        if self._glow is not None:
            self._glow.hide()
        self._cancel_recovery()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def _on_tray_shift_tap(self) -> None:
        """Shift tap (no spacebar) during tray = dismiss."""
        if self._tray_active:
            logger.info("Shift tap during tray — dismiss")
            self._acknowledge_tray_entry(self._tray_index)
            self._dismiss_tray()

    def _toggle_command_overlay(self) -> None:
        """Toggle command overlay visibility."""
        if self._command_client is None:
            return
        overlay_visible = (
            self._command_overlay is not None
            and getattr(self._command_overlay, '_visible', False)
        )
        if overlay_visible:
            logger.info("Double-tap Enter — dismissing command overlay")
            self._command_overlay.cancel_dismiss()
            self._detector.command_overlay_active = False
        elif self._transcribing and self._command_overlay is not None:
            # Generation still in progress — re-show with accumulated text.
            utterance = getattr(self, "_last_command_utterance", "")
            streaming = getattr(self, "_command_streaming_text", "")
            logger.info(
                "Double-tap Enter — resuming in-progress overlay (%d chars so far)",
                len(streaming),
            )
            try:
                self._sync_command_overlay_brightness(immediate=True)
                self._command_overlay.show(preserve_thinking_timer=True)
                self._command_overlay.set_utterance(utterance)
                if streaming:
                    self._command_overlay.set_response_text(streaming)
                    self._command_overlay.invert_thinking_timer()
                self._detector.command_overlay_active = True
            except Exception:
                logger.exception("Resume overlay failed")
        else:
            snapshot = self._last_command_overlay_snapshot()
            if snapshot is not None:
                last_utterance, last_response = snapshot
                logger.info("Double-tap Enter — recalling last response")
                if self._command_overlay is not None:
                    try:
                        self._sync_command_overlay_brightness(immediate=True)
                        self._command_overlay.show()
                        self._command_overlay.set_utterance(last_utterance)
                        self._command_overlay.append_token(last_response)
                        self._command_overlay.finish()
                        self._detector.command_overlay_active = True
                    except Exception:
                        logger.exception("Recall overlay failed")
            else:
                logger.info("Double-tap Enter — no assistant overlay snapshot to recall")

    def _toggle_terraform_hud(self) -> None:
        """Toggle Terror Form HUD visibility — called from double-tap Shift."""
        hud = getattr(self, '_terraform_hud', None)
        if hud is not None:
            hud.toggle()
            logger.info("Double-tap Shift — toggled Terror Form HUD")

    def _on_audio_shift_tap(self) -> None:
        """Shift tap while idle toggles current TTS audibility."""
        if self._tray_active:
            return
        tts = getattr(self, "_tts_client", None)
        if tts is None:
            return
        audible = tts.toggle_audio()
        logger.info("Idle shift tap — audio target now %s", "on" if audible else "off")
        if self._menubar is not None:
            self._menubar.set_status_text("Audio on" if audible else "Audio muted")

    def _on_tray_navigate_up(self) -> None:
        """Spacebar held + shift tapped during tray = navigate up (more recent)."""
        if self._tray_active:
            logger.info("Shift tap during hold — navigate up")
            self._tray_navigate_up()

    def _on_tray_enter_pressed(self) -> None:
        """Legacy tray Enter hook retained as a no-op safety shim."""
        if self._tray_active:
            logger.info("Enter during tray — ignored (space-rooted contract)")

    def _on_tray_delete_gesture(self) -> None:
        """Shift held + double-tap spacebar = delete current tray entry."""
        if self._tray_active:
            logger.info("Double-tap delete during tray")
            self._tray_delete_current()

    def _tray_cycle(self) -> None:
        """Cycle to the next tray entry, wrapping at the edges."""
        if not self._tray_active or len(self._tray_stack) < 2:
            return
        self._tray_index = (self._tray_index - 1) % len(self._tray_stack)
        self._show_tray_current()

    def _tray_navigate_up(self) -> None:
        """Navigate up toward more recent entries. Stop at top."""
        if not self._tray_active:
            return
        if self._tray_index >= len(self._tray_stack) - 1:
            # Already at the top — stay put (don't dismiss mid-hold)
            return
        else:
            self._tray_index += 1
            self._show_tray_current(acknowledge=True)

    def _tray_navigate_down(self) -> None:
        """Navigate down toward older entries. Stop at bottom."""
        if not self._tray_active:
            return
        if self._tray_index > 0:
            self._tray_index -= 1
            self._show_tray_current(acknowledge=True)

    def _tray_delete_current(self) -> None:
        """Delete the currently displayed tray entry."""
        if not self._tray_active or not self._tray_stack:
            return
        del self._tray_stack[self._tray_index]
        if not self._tray_stack:
            self._dismiss_tray()
            return
        # Adjust index: stay at same position or move up if we were at the end
        if self._tray_index >= len(self._tray_stack):
            self._tray_index = len(self._tray_stack) - 1
        self._show_tray_current(acknowledge=True)

    def _tray_insert_current(self) -> None:
        """Insert the current tray entry at cursor and consume it."""
        if not self._tray_active or not self._tray_stack:
            return
        entry = self._get_tray_entry(self._tray_index)
        text = entry.text

        # Dismiss tray first, then inject. Dismissing before inject ensures
        # the tray overlay doesn't interfere with focus on the target app.
        self._tray_active = False
        self._detector.tray_active = False

        # Remove consumed entry from stack
        del self._tray_stack[self._tray_index]
        if self._tray_index >= len(self._tray_stack) and self._tray_stack:
            self._tray_index = len(self._tray_stack) - 1

        if self._glow is not None:
            self._glow.hide()
        self._cancel_recovery()
        if self._overlay is not None:
            self._overlay.hide()

        # Small delay to let focus settle after overlay dismissal,
        # then inject the text
        self._tray_pending_inject = text
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, "trayInjectDelayed:", None, False
        )

    def trayInjectDelayed_(self, timer) -> None:
        """Inject tray text after a short delay for focus to settle."""
        text = getattr(self, "_tray_pending_inject", None)
        self._tray_pending_inject = None
        if not text:
            return

        inject_text(text)
        logger.info("Tray injected: %r", text[:50])
        if self._menubar is not None:
            self._menubar.set_status_text("Pasted!")

    def _tray_send_current(self) -> None:
        """Send the current tray entry to the assistant and consume it."""
        if not self._tray_stack:
            # Nothing to send — make sure tray state is clean
            if self._tray_active:
                self._dismiss_tray()
            return
        if not self._tray_active:
            return
        entry = self._get_tray_entry(self._tray_index)
        text = entry.text

        # Remove consumed entry from stack
        del self._tray_stack[self._tray_index]
        if self._tray_index >= len(self._tray_stack) and self._tray_stack:
            self._tray_index = len(self._tray_stack) - 1

        self._tray_active = False
        self._detector.tray_active = False
        if self._glow is not None:
            self._glow.hide()
        self._cancel_recovery()

        if self._command_client is not None:
            self._send_text_as_command(text)
        else:
            logger.warning("Tray send — no command client configured")
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")

    def _add_assistant_content_to_tray(self, text: str) -> dict:
        """Place assistant-created content into the tray on the main thread.

        Must be called from a background thread — uses waitUntilDone=True to
        synchronously dispatch tray mutation onto the main thread. Relies on
        tool calls being executed sequentially (the command.py tool loop) so
        that _tray_tool_result is never written concurrently.
        """
        self._tray_tool_result = None
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "toolAddToTray:",
            {"text": text},
            True,
        )
        return self._tray_tool_result or {"error": "Tray update failed"}

    def toolAddToTray_(self, payload: dict) -> None:
        """Main-thread tray mutation entrypoint for command tool calls."""
        text = payload.get("text", "")
        reveal_now = bool(self._tray_active)
        self._add_tray_entry(
            text,
            owner="assistant",
            activate=reveal_now,
        )
        self._tray_tool_result = {
            "status": "added",
            "tray_visible": reveal_now,
            "stack_size": len(self._tray_stack),
            "owner": "assistant",
        }

    # ── command pathway ────────────────────────────────────

    def _send_text_as_command(self, text: str) -> None:
        """Send pre-transcribed text to the command pathway.

        Used when shift+space is pressed during recovery — the text is
        already transcribed, so we skip the audio transcription step and
        go straight to OMLX streaming.
        """
        self._transcription_token += 1
        token = self._transcription_token
        self._transcribing = True
        self._command_tool_used_tts = False
        self._transcribe_start = time.monotonic()

        if self._menubar is not None:
            self._menubar.set_status_text("Sending to assistant…")

        # Dispatch utterance to the main thread for overlay setup
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "commandUtteranceReady:",
            {"token": token, "utterance": text},
            False,
        )

        # Stream the command in a background thread
        def _stream():
            full_response = ""
            stale_break = False
            try:
                for event in self._command_client.stream_command_events(
                    text,
                    tools=self._tool_schemas,
                    tool_executor=self._make_tool_executor(),
                    cancel_check=lambda: token != self._transcription_token,
                ):
                    if token != self._transcription_token:
                        stale_break = True
                        break  # stale
                    if event.kind == "assistant_delta":
                        full_response += event.text
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToken:", {"token": token, "text": event.text}, False
                        )
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToolEnd:", {"token": token}, False
                        )
                    elif event.kind == "tool_call":
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToken:", {"token": token, "text": event.text}, False
                        )
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToolStart:", {"token": token}, False
                        )
                    elif event.kind == "assistant_final":
                        if not full_response:
                            full_response = event.text
            except urllib.error.HTTPError as exc:
                logger.exception("Command stream failed with HTTP error")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "commandFailed:",
                    {"token": token, "error": _format_command_http_error(exc)},
                    False,
                )
                return
            except Exception:
                logger.exception("Command stream failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "commandFailed:", {"token": token, "error": "Command failed"}, False
                )
                return
            finally:
                # Repair history only when a stale token break interrupts the
                # streaming loop before the command client can finalize the turn.
                if stale_break and full_response:
                    self._command_client._history.append((text, full_response))
                    max_h = self._command_client._max_history
                    if len(self._command_client._history) > max_h:
                        self._command_client._history.pop(0)
                    logger.info("Command history saved: %d turns", len(self._command_client._history))

            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandComplete:", {"token": token, "response": full_response}, False
            )

        threading.Thread(target=_stream, daemon=True).start()

    def _make_tool_executor(self):
        """Build a tool executor closure with current app state."""
        scene_cache = self._scene_cache
        raw_tts_client = self._ensure_tts_client(allow_default_voice=True)
        tts_client = raw_tts_client
        # Get last assistant response for last_response refs
        last_response = None
        try:
            history = self._command_client.history if self._command_client else []
            if history:
                _, last_response = history[-1]
        except (TypeError, ValueError):
            pass

        if raw_tts_client is not None:
            delegate = self

            class _ToolTTSProxy:
                def __init__(self, client):
                    self._client = client

                def speak_async(self, text, *args, **kwargs):
                    result = self._client.speak_async(text, *args, **kwargs)
                    delegate._command_tool_used_tts = True
                    return result

                def speak(self, text, *args, **kwargs):
                    result = self._client.speak(text, *args, **kwargs)
                    delegate._command_tool_used_tts = True
                    return result

                def __getattr__(self, name):
                    return getattr(self._client, name)

            tts_client = _ToolTTSProxy(raw_tts_client)

        def _executor(name, arguments, **kwargs):
            return execute_tool(
                name=name,
                arguments=arguments,
                scene_cache=scene_cache,
                last_response=last_response,
                tts_client=tts_client,
                tray_writer=self._add_assistant_content_to_tray,
            )
        return _executor

    def _command_transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: transcribe then send command to OMLX."""
        self._command_tool_used_tts = False
        # Wait for preview loop to finish
        if self._preview_thread is not None:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        # Step 1: Transcribe the audio
        try:
            # Fast path: use cached segment transcriptions + tail only.
            utterance = self._transcribe_segments_and_tail(wav_bytes)
            if utterance is None:
                with self._local_inference_context(self._client):
                    if (
                        getattr(self._client, 'supports_streaming', False)
                        and self._client is self._preview_client
                        and getattr(self._client, "has_active_stream", False)
                    ):
                        utterance = self._client.finish_stream()
                    else:
                        utterance = self._client.transcribe(wav_bytes)
        except Exception:
            logger.exception("Command transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandFailed:", {"token": token, "error": "Transcription failed"}, False
            )
            return

        if not utterance:
            # No speech with shift held = recall last response
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "_recallLastResponse:", {"token": token}, False
            )
            return

        transcribe_ms = (time.monotonic() - self._transcribe_start) * 1000
        logger.info("Command utterance: %r (%.0fms)", utterance, transcribe_ms)

        # Show the utterance in the input overlay before hiding it
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "commandUtteranceReady:",
            {"token": token, "utterance": utterance},
            False,
        )

        # Step 2: Stream the command response
        full_response = ""
        stale_break = False
        narrator_started = False
        vamp_started = False
        first_event_received = False
        try:
            # Start the narrator if available
            if self._narrator is not None:
                self._narrator.start()
                narrator_started = True
                # Start loading vamp — runs in background while the HTTP
                # request blocks during model loading.  Stops when the
                # first event arrives (model is loaded and generating).
                model_id = getattr(self, "_command_model_id", "") or ""
                self._narrator.start_loading_vamp(
                    utterance=utterance, model_id=model_id
                )
                vamp_started = True
                # Enable color shimmer on the narrator label during loading
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "narratorShimmer:", {"active": True}, False
                )

            for event in self._command_client.stream_command_events(
                utterance,
                tools=self._tool_schemas,
                tool_executor=self._make_tool_executor(),
                cancel_check=lambda: token != self._transcription_token,
            ):
                if token != self._transcription_token:
                    stale_break = True
                    break  # stale

                # Stop loading vamp on first event from the model
                if not first_event_received:
                    first_event_received = True
                    if vamp_started and self._narrator is not None:
                        self._narrator.stop_loading_vamp()
                        vamp_started = False
                    # Hide the narrator label — vamp text shouldn't linger
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "narratorHide:", {"token": token}, False
                    )

                if event.kind == "thinking_delta":
                    # Feed thinking tokens to the narrator sidecar
                    if self._narrator is not None:
                        # Restart narrator if it was stopped (e.g. thinking between tool rounds)
                        if not narrator_started:
                            self._narrator.start()
                            narrator_started = True
                            # Unsuppress narrator display and show "Thinking" placeholder
                            if self._command_overlay is not None:
                                self._command_overlay._narrator_suppressed = False
                            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                                "narratorSummary:", {"summary": "Thinking"}, False
                            )
                        self._narrator.feed(event.text)
                elif event.kind == "assistant_delta" or event.kind == "tool_call":
                    # Stop narrator when visible content starts — produce collapsed summary
                    if narrator_started and self._narrator is not None:
                        self._narrator.stop_and_summarize()
                        narrator_started = False
                        # Hide narrator label immediately so it doesn't overlap response
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "narratorHide:", {"token": token}, False
                        )
                    if event.text:
                        full_response += event.text
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "commandToken:", {"token": token, "text": event.text}, False
                    )
                    if event.kind == "tool_call":
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToolStart:", {"token": token}, False
                        )
                    else:
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToolEnd:", {"token": token}, False
                        )
                elif event.kind == "assistant_final":
                    if not full_response:
                        full_response = event.text
        except urllib.error.HTTPError as exc:
            logger.exception("Command stream failed with HTTP error")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandFailed:",
                {"token": token, "error": _format_command_http_error(exc)},
                False,
            )
            return
        except Exception:
            logger.exception("Command stream failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandFailed:", {"token": token, "error": "Command failed"}, False
            )
            return
        finally:
            # Always stop the narrator and vamp on stream end
            if vamp_started and self._narrator is not None:
                self._narrator.stop_loading_vamp()
            if narrator_started and self._narrator is not None:
                self._narrator.stop()
            # Repair history only when a stale token break interrupts the
            # streaming loop before the command client can finalize the turn.
            if stale_break and full_response:
                # Stale break — no full message chain available, store minimal pair
                self._command_client._history.append([
                    {"role": "user", "content": utterance},
                    {"role": "assistant", "content": full_response},
                ])
                max_h = self._command_client._max_history
                if len(self._command_client._history) > max_h:
                    self._command_client._history.pop(0)
                logger.info(
                    "Command history saved after stale token break: %d turns",
                    len(self._command_client._history),
                )

        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "commandComplete:", {"token": token, "response": full_response}, False
        )

    def commandUtteranceReady_(self, payload: dict) -> None:
        """Main thread: show transcribed command, prepare output overlay."""
        if payload["token"] != self._transcription_token:
            return
        utterance = payload["utterance"]
        self._last_command_utterance = utterance
        self._last_command_response = ""
        self._command_streaming_text = ""
        # Hide the input overlay
        if self._overlay is not None:
            self._overlay.hide()
        # Show the command overlay with the utterance as context
        if self._command_overlay is not None:
            self._sync_command_overlay_brightness(immediate=True)
            self._command_overlay.show()
            self._command_overlay.set_utterance(utterance)
            self._detector.command_overlay_active = True
            logger.info("command_overlay_active -> True (command started)")
        self._command_first_token = True
        if self._menubar is not None:
            self._menubar.set_status_text("Thinking…")

    def commandToolStart_(self, payload: dict) -> None:
        """Main thread: tool execution started."""
        if payload["token"] != self._transcription_token:
            return
        if self._command_overlay is not None:
            self._command_overlay.set_tool_active(True)

    def commandToolEnd_(self, payload: dict) -> None:
        """Main thread: tool execution finished."""
        if payload["token"] != self._transcription_token:
            return
        if self._command_overlay is not None:
            self._command_overlay.set_tool_active(False)

    def narratorHide_(self, payload: dict) -> None:
        """Main thread: hide the narrator label immediately."""
        if payload.get("token") != self._transcription_token:
            return
        overlay = self._command_overlay
        if overlay is not None:
            try:
                overlay._hide_narrator()
            except Exception:
                logger.exception("Command overlay failed to hide narrator")

    def narratorShimmer_(self, payload: dict) -> None:
        """Main thread: enable/disable color shimmer on narrator label."""
        overlay = self._command_overlay
        if overlay is not None:
            try:
                overlay.set_narrator_shimmer(payload.get("active", False))
            except Exception:
                logger.exception("Command overlay failed to set narrator shimmer")

    def narratorCollapsed_(self, payload: dict) -> None:
        """Main thread: inject collapsed thinking summary into the overlay."""
        text = payload.get("text", "")
        if not text:
            return
        overlay = self._command_overlay
        if overlay is not None:
            try:
                overlay.set_thinking_collapsed(text)
            except Exception:
                logger.exception("Command overlay failed to set collapsed thinking")

    def narratorSummary_(self, payload: dict) -> None:
        """Main thread: update the overlay with a narrator thinking summary."""
        summary = payload.get("summary", "")
        if not summary:
            return
        overlay = self._command_overlay
        if overlay is not None:
            try:
                overlay.set_narrator_summary(summary)
            except Exception:
                logger.exception("Command overlay failed to set narrator summary")

    def commandToken_(self, payload: dict) -> None:
        """Main thread: append a streamed token to the command overlay."""
        if payload["token"] != self._transcription_token:
            return
        # Always accumulate streaming text so we can restore the overlay
        # if the user dismisses and re-opens mid-generation.
        text = payload["text"]
        if not hasattr(self, "_command_streaming_text"):
            self._command_streaming_text = ""
        self._command_streaming_text += text
        overlay = self._command_overlay
        # First content token: invert the thinking timer and update status
        if getattr(self, "_command_first_token", False):
            self._command_first_token = False
            if overlay is not None:
                try:
                    overlay.invert_thinking_timer()
                except Exception:
                    logger.exception("Command overlay failed to invert thinking timer")
            if self._menubar is not None:
                self._menubar.set_status_text("Responding…")
        if overlay is not None:
            try:
                overlay.append_token(text)
            except Exception:
                logger.exception("Command overlay failed to append streamed token")

    def commandComplete_(self, payload: dict) -> None:
        """Main thread: command response finished streaming."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        # Reset cancel spring if generation finishes while spring is winding
        self._cancel_spring_active = False
        self._detector.cancel_spring_active = False
        if self._command_overlay is not None:
            self._command_overlay.set_cancel_spring(0.0)
        overlay = self._command_overlay
        if overlay is not None:
            overlay.set_tool_active(False)
        response = payload.get("response", "")
        if response:
            self._last_command_response = response
        if overlay is not None and response:
            try:
                overlay.set_response_text(response)
            except Exception:
                logger.exception("Command overlay failed to apply final response text")
        if overlay is not None:
            try:
                overlay.finish()
            except Exception:
                logger.exception("Command overlay finish failed")
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")
        # TTS autoplay removed — the model has read_aloud as a tool; if it
        # chose not to call it, the response is text-only.  Clean up the
        # tool-used flag so it doesn't leak across commands.
        self._command_tool_used_tts = False
        if self._glow is not None:
            self._glow.hide()

    def _on_tts_amplitude(self, rms: float) -> None:
        """Called from TTS thread — marshal to main thread."""
        from Foundation import NSNumber
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "ttsAmplitudeUpdate:", NSNumber.numberWithFloat_(rms), False
        )

    def ttsAmplitudeUpdate_(self, rms_number) -> None:
        """Main thread: forward TTS amplitude to command overlay."""
        rms = float(rms_number)
        if self._command_overlay is not None:
            try:
                self._command_overlay.update_tts_amplitude(rms)
            except Exception:
                logger.exception("Command overlay amplitude update failed")

    def ttsFinished_(self, _) -> None:
        """Main thread: TTS playback ended — restore overlay opacity."""
        if self._command_overlay is not None:
            try:
                self._command_overlay.tts_stop()
            except Exception:
                logger.exception("Command overlay TTS stop failed")

    def ttsError_(self, message) -> None:
        """Main thread: TTS playback failed — show error in menubar."""
        msg = str(message) if message else "TTS playback failed"
        logger.error("TTS autoplay error surfaced: %s", msg)
        if self._menubar is not None:
            # Truncate for menubar but keep it useful
            short = msg if len(msg) <= 80 else msg[:77] + "..."
            self._menubar.set_status_text(f"TTS error: {short}")

    def commandFailed_(self, payload: dict) -> None:
        """Main thread: show error in the command overlay, then fade."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        error = payload.get("error", "Unknown error")
        error_text = (
            "couldn't reach the model — try again in a moment"
            if error in {"Unknown error", "Command failed"}
            else error
        )
        logger.error("Command pathway error: %s", error)
        if self._glow is not None:
            self._glow.hide()
        if self._overlay is not None:
            self._overlay.hide()
        if getattr(self, "_last_command_utterance", ""):
            self._last_command_response = error_text
        # Show the error in the command overlay like a response
        if self._command_overlay is not None:
            if not self._command_overlay._visible:
                self._sync_command_overlay_brightness(immediate=True)
                try:
                    self._command_overlay.show()
                except Exception:
                    logger.exception("Command overlay show failed during error presentation")
            try:
                self._command_overlay.append_token(error_text)
            except Exception:
                logger.exception("Command overlay append failed during error presentation")
            try:
                self._command_overlay.finish()
                self._detector.command_overlay_active = True
                logger.info("command_overlay_active -> True (command failed)")
            except Exception:
                logger.exception("Command overlay finish failed during error presentation")
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def verifyPaste_(self, timer) -> None:
        """Background OCR verification — confirm the pasted text appeared on screen."""
        text = getattr(self, "_verify_paste_text", None)
        if text is None:
            return

        attempt = getattr(self, "_verify_paste_attempt", 0)

        # Run OCR in background thread to avoid blocking the main thread
        import threading
        def _verify():
            focused_match = focused_text_contains(text)
            if focused_match is True:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "verifyPasteResult:",
                    {
                        "found": True,
                        "status": "confirmed_ax",
                        "text": text,
                        "attempt": attempt,
                    },
                    False,
                )
                return
            from .paste_verify import capture_screen_text, classify_paste_result
            screen_text = capture_screen_text()
            status = classify_paste_result(text, screen_text)
            found = status == "confirmed"
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "verifyPasteResult:",
                {
                    "found": found,
                    "status": status,
                    "text": text,
                    "attempt": attempt,
                },
                False,
            )
        threading.Thread(target=_verify, daemon=True).start()

    def verifyPasteResult_(self, payload) -> None:
        """Main thread: handle OCR verification result."""
        found = payload["found"]
        status = payload.get("status", "confirmed" if found else "missing")
        text = payload["text"]
        attempt = payload["attempt"]

        # If we've moved on (new recording started, or recovery already active),
        # discard this result.
        if getattr(self, "_verify_paste_text", None) != text:
            return

        is_retry = getattr(self, "_recovery_retry_pending", False)

        if found:
            logger.info("Paste verified by OCR (attempt %d)", attempt + 1)
            self._verify_paste_text = None
            if is_retry:
                # Retry succeeded — clear recovery state
                self._recovery_retry_pending = False
                self._clear_recovery_state()
                if self._menubar is not None:
                    self._menubar.set_status_text("Pasted!")
            return

        if attempt == 0:
            # First check failed — retry once at 300ms to give slow apps time
            logger.debug("Paste not verified on first check, retrying in 200ms")
            self._verify_paste_attempt = 1
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.2, self, "verifyPaste:", None, False
            )
            return

        # Second check also failed
        self._verify_paste_text = None
        if is_retry:
            # Retry from recovery failed — bounce the overlay back
            logger.warning("Recovery retry not verified by OCR — bouncing back")
            self._recovery_retry_pending = False
            self._enter_recovery_mode(text)
        else:
            # Normal paste failed verification — preserve it without surfacing tray UI.
            logger.warning(
                "Paste not verified by OCR after %d attempts (%s) — stashing silently to tray",
                attempt + 1,
                status,
            )
            self._stash_failed_paste_to_tray(text, status=status)

    # ── helpers ─────────────────────────────────────────────

    _MODEL_OPTIONS = [
        ("mlx-community/whisper-tiny.en-mlx", "Tiny.en (float16)"),
        ("mlx-community/whisper-base.en-mlx", "Base.en (float16)"),
        ("mlx-community/whisper-base.en-mlx-8bit", "Base.en (8bit)"),
        ("mlx-community/whisper-small.en-mlx", "Small.en (float16)"),
        ("mlx-community/whisper-small.en-mlx-8bit", "Small.en (8bit)"),
        ("mlx-community/whisper-medium.en-mlx-4bit", "Medium.en (4bit)"),
        ("mlx-community/whisper-medium.en-mlx-8bit", "Medium.en (8bit)"),
        ("mlx-community/whisper-medium.en-mlx", "Medium.en (float16)"),
        ("mlx-community/whisper-large-v3-turbo-4bit", "v3 Large Turbo (4bit)"),
        ("mlx-community/whisper-large-v3-turbo-8bit", "v3 Large Turbo (8bit)"),
        ("mlx-community/whisper-large-v3-turbo", "v3 Large Turbo (float16)"),
        ("Qwen/Qwen3-ASR-0.6B", "Qwen3 ASR 0.6B (streaming)"),
        (_PARAKEET_MODEL_ID, "Parakeet CTC-110M (CoreML/ANE, preview only)"),
    ]

    # Parakeet is preview-only: too rough for final transcription
    _PREVIEW_ONLY_MODELS = frozenset({_PARAKEET_MODEL_ID})

    def _select_model(self, model_id):
        """Model picker. Pass None to get the menu list, or a model ID to switch."""
        if model_id is None:
            # Return (model_id, label, enabled) tuples — hide disallowed models
            return [
                (mid, label, True)
                for mid, label in self._MODEL_OPTIONS
                if self._model_allowed(mid)
            ]
        current_preview = getattr(
            self,
            "_preview_model_id",
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or _DEFAULT_PREVIEW_MODEL,
        )
        current_transcription = getattr(
            self,
            "_transcription_model_id",
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or self._default_transcription_model(),
        )
        current_preview, current_transcription = self._sanitize_model_ids(
            current_preview,
            current_transcription,
        )
        self._apply_model_selection(
            preview_model=current_preview if model_id is None else model_id,
            transcription_model=current_transcription if model_id is None else model_id,
        )

    def _handle_model_menu_action(self, selection):
        """Menu callback for preview/transcription role-specific model choices."""
        if selection is None:
            current_preview = getattr(
                self, "_preview_model_id", _DEFAULT_PREVIEW_MODEL
            )
            current_transcription = getattr(
                self, "_transcription_model_id", self._default_transcription_model()
            )
            current_preview, current_transcription = self._sanitize_model_ids(
                current_preview,
                current_transcription,
            )
            state = {
                "transcription": {
                    "selected": current_transcription,
                    "models": self._select_model(None),
                },
                "preview": {
                    "selected": current_preview,
                    "models": self._select_model(None),
                },
            }
            launch_target = self._launch_target_menu_state()
            if launch_target is not None:
                state["launch_target"] = launch_target
            if self._command_client is not None:
                assistant_title = "Assistant Model"
                server_unreachable = getattr(self, "_command_server_unreachable", False)
                command_backend = getattr(self, "_command_backend", _DEFAULT_COMMAND_BACKEND)
                cloud_provider = getattr(
                    self,
                    "_command_cloud_provider",
                    self._load_cloud_provider_preference(),
                )
                if server_unreachable:
                    assistant_title = "Assistant Model (server unreachable)"
                state["assistant"] = {
                    "title": assistant_title,
                    "selected": self._command_model_id,
                    "models": [] if server_unreachable else self._command_model_options,
                }
                state["assistant_backend"] = {
                    "title": "Assistant Backend",
                    "items": [
                        (
                            "local",
                            "Local OMLX",
                            getattr(self, "_command_backend", _DEFAULT_COMMAND_BACKEND)
                            == "local",
                            True,
                        ),
                        (
                            "sidecar",
                            "Sidecar OMLX",
                            command_backend == "sidecar",
                            True,
                        ),
                        (
                            "cloud_google",
                            "Google Cloud",
                            command_backend == "cloud" and cloud_provider == "google",
                            True,
                        ),
                        (
                            "cloud_openrouter",
                            "OpenRouter",
                            command_backend == "cloud" and cloud_provider == "openrouter",
                            True,
                        ),
                        ("configure", "Set Sidecar URL…", False, True),
                        ("configure_cloud_google", "Set Google Cloud Endpoint…", False, True),
                        ("configure_cloud_openrouter", "Set OpenRouter Endpoint…", False, True),
                    ],
                }
            tts_client = getattr(self, "_tts_client", None)
            tts_backend = getattr(self, "_tts_backend", "local")
            tts_sidecar_url = getattr(self, "_tts_sidecar_url", "")
            tts_voice_pref = (
                self._load_preference("tts_cloud_voice")
                if tts_backend == "cloud"
                else self._load_preference("tts_voice")
            ) or os.environ.get("SPOKE_TTS_VOICE", "")
            saved_tts_model = (
                self._load_preference("tts_cloud_model")
                if tts_backend == "cloud"
                else self._load_preference("tts_sidecar_model")
                if tts_backend == "sidecar"
                else self._load_preference("tts_model")
            ) or os.environ.get("SPOKE_TTS_MODEL", "")
            if tts_client is not None or tts_voice_pref or saved_tts_model or tts_backend in ("sidecar", "cloud"):
                has_tts_sidecar_url = bool(tts_sidecar_url)
                has_tts_cloud = bool(
                    os.environ.get("GEMINI_API_KEY")
                    or self._load_preference("tts_cloud_api_key")
                    or self._load_preference("command_cloud_api_key")
                )
                tts_backend_labels = {"local": "Local", "sidecar": "Sidecar", "cloud": "Cloud (Gemini)"}
                tts_target = "not active"
                if tts_client is not None:
                    if isinstance(tts_client, CloudTTSClient):
                        tts_target = "Gemini"
                    elif isinstance(tts_client, RemoteTTSClient):
                        tts_target = _url_host(getattr(tts_client, "_base_url", ""))
                    else:
                        tts_target = "local runtime"
                state["tts_backend"] = {
                    "title": f"TTS Backend: {tts_backend_labels.get(tts_backend, tts_backend)}",
                    "items": [
                        ("local", "Local runtime", tts_backend == "local"),
                        ("sidecar", (
                            f"Sidecar ({_url_host(tts_sidecar_url)})"
                            if has_tts_sidecar_url
                            else "Sidecar (not configured)"
                        ), tts_backend == "sidecar", has_tts_sidecar_url),
                        ("cloud", (
                            "Cloud (Gemini)"
                            if has_tts_cloud
                            else "Cloud (no API key)"
                        ), tts_backend == "cloud", has_tts_cloud),
                        ("configure_tts", "Set TTS Sidecar URL\u2026", False, True),
                    ],
                }
                state["tts_endpoint"] = {
                    "title": f"TTS Endpoint: {tts_target}",
                    "note": (
                        "Routing source: Gemini cloud API"
                        if tts_backend == "cloud"
                        else "Routing source: saved sidecar URL"
                        if tts_backend == "sidecar" and has_tts_sidecar_url
                        else "Routing source: local runtime"
                    ),
                }
            whisper_backend = getattr(self, "_whisper_backend", "local")
            preview_backend = getattr(self, "_preview_backend", "local")
            whisper_sidecar_url = getattr(self, "_whisper_sidecar_url", "")
            has_whisper_sidecar_url = bool(whisper_sidecar_url)
            whisper_cloud_url = getattr(self, "_whisper_cloud_url", "")
            has_whisper_cloud = bool(
                whisper_cloud_url and getattr(self, "_whisper_cloud_api_key", "")
            )
            backend_labels = {
                "local": "Local",
                "sidecar": "Sidecar",
                "cloud": "Cloud (OpenAI)",
            }

            def _backend_items(selected: str) -> list:
                return [
                    ("local", "Local Whisper", selected == "local"),
                    ("sidecar", (
                        f"Sidecar ({_url_host(whisper_sidecar_url)})"
                        if has_whisper_sidecar_url
                        else "Sidecar (not configured)"
                    ), selected == "sidecar", has_whisper_sidecar_url),
                    ("cloud", (
                        "Cloud (OpenAI)"
                        if has_whisper_cloud
                        else "Cloud (not configured)"
                    ), selected == "cloud", has_whisper_cloud),
                    ("configure_whisper", "Set Whisper Sidecar URL\u2026", False, True),
                    ("configure_whisper_cloud", "Set Cloud API Key\u2026", False, True),
                ]

            state["transcription_backend"] = {
                "title": f"Final: {backend_labels.get(whisper_backend, whisper_backend)}",
                "items": _backend_items(whisper_backend),
            }
            state["preview_backend"] = {
                "title": f"Preview: {backend_labels.get(preview_backend, preview_backend)}",
                "items": _backend_items(preview_backend),
            }
            if self._local_whisper_controls_available():
                eager_eval_available = self._local_whisper_eager_eval_available()
                state["local_whisper"] = {
                    "title": "Local Whisper",
                    "items": [
                        (
                            "decode_timeout",
                            "Decode timeout guard (30s)",
                            getattr(
                                self,
                                "_local_whisper_decode_timeout",
                                _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
                            )
                            is not None,
                            True,
                        ),
                        (
                            "eager_eval",
                            (
                                "Stability mode (eager eval)"
                                if eager_eval_available
                                else "Stability mode (eager eval) [mlx-whisper update needed]"
                            ),
                            (
                                getattr(
                                    self,
                                    "_local_whisper_eager_eval",
                                    _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
                                )
                                if eager_eval_available
                                else False
                            ),
                            eager_eval_available,
                        ),
                    ],
                }
            tts = getattr(self, "_tts_client", None)
            show_tts_menus = (
                tts is not None
                or tts_backend in ("sidecar", "cloud")
                or bool(tts_voice_pref)
                or bool(saved_tts_model)
            )
            if show_tts_menus:
                current_tts_model = getattr(tts, "_model_id", "") if tts else saved_tts_model
                if tts_backend == "cloud":
                    cloud_model = self._load_preference("tts_cloud_model") or "gemini-2.5-flash-preview-tts"
                    current_tts_model = current_tts_model or cloud_model
                    tts_models = [(cloud_model, f"Gemini ({cloud_model})", True)]
                elif tts_backend == "sidecar":
                    sidecar_models = self._discover_tts_sidecar_models()
                    if sidecar_models:
                        tts_models = sidecar_models
                    elif current_tts_model:
                        tts_models = [(current_tts_model, f"{current_tts_model} (fallback)", True)]
                    else:
                        tts_models = []
                else:
                    tts_models = [
                        (model_id, label, model_id == current_tts_model)
                        for model_id, label in _TTS_MODELS
                    ]
                tts_unreachable = getattr(self, "_tts_server_unreachable", False)
                if tts_models or (tts_backend == "sidecar" and tts_unreachable):
                    tts_title = "TTS Model"
                    if tts_backend == "sidecar" and tts_unreachable:
                        tts_title = "TTS Model (server unreachable)"
                    state["tts"] = {
                        "title": tts_title,
                        "selected": current_tts_model,
                        "models": tts_models,
                    }
                current_voice = getattr(tts, "_voice", "") if tts else tts_voice_pref
                # Cloud voices come from the preset list
                if tts_backend == "cloud":
                    sidecar_voices = []
                    local_choices = _voice_choices_for_model(current_tts_model, current_voice)
                    # Fall back: if model key didn't match, use Gemini voices directly
                    if local_choices is None:
                        local_choices = [
                            (v, label, v == current_voice) for v, label in GEMINI_VOICES
                        ]
                # Try sidecar discovery first, then local presets
                elif tts_backend == "sidecar":
                    sidecar_voices = self._discover_tts_sidecar_voices(current_tts_model)
                    local_choices = (
                        _voice_choices_for_model(current_tts_model, current_voice)
                        if not sidecar_voices
                        else None
                    )
                else:
                    sidecar_voices = []
                    local_choices = (
                        _voice_choices_for_model(current_tts_model, current_voice)
                        if not sidecar_voices
                        else None
                    )
                if sidecar_voices:
                    voice_models = [
                        (v, v, v == current_voice) for v in sidecar_voices
                    ]
                    state["tts_voice"] = {
                        "type": "choice",
                        "selected": current_voice,
                        "models": voice_models,
                    }
                elif local_choices is not None:
                    voice_label = _voice_preset_label(current_tts_model, current_voice)
                    state["tts_voice"] = {
                        "type": "choice",
                        "title": f"TTS Voice: {voice_label}",
                        "selected": current_voice,
                        "models": local_choices,
                    }
                else:
                    title = f"TTS Voice: {current_voice or '(not set)'}"
                    items = [
                        ("configure_voice", "Set TTS Voice\u2026", False, True),
                    ]
                    if tts_backend == "sidecar":
                        title += " [sidecar /v1/voices needed]"
                        items = [
                            (
                                "voice_discovery_unavailable",
                                "Voice discovery unavailable on this sidecar",
                                False,
                                False,
                            ),
                            *items,
                        ]
                    state["tts_voice"] = {
                        "type": "toggle",
                        "title": title,
                        "items": (
                            items
                        ),
                    }
            return state
        if not isinstance(selection, tuple) or len(selection) != 2:
            self._select_model(selection)
            return
        role, model_id = selection
        if role == "assistant":
            self._apply_command_model_selection(model_id)
            return
        if role == "assistant_backend":
            self._apply_command_backend_selection(model_id)
            return
        if role == "launch_target":
            self._apply_launch_target_selection(model_id)
            return
        if role == "tts_backend":
            self._apply_tts_backend_selection(model_id)
            return
        if role == "tts_voice":
            self._apply_tts_voice_selection(model_id)
            return
        if role == "tts":
            self._apply_tts_model_selection(model_id)
            return
        if role == "transcription_backend":
            self._apply_transcription_backend_selection(model_id)
            return
        if role == "preview_backend":
            self._apply_preview_backend_selection(model_id)
            return
        if role == "local_whisper":
            self._toggle_local_whisper_setting(model_id)
            return
        if role not in {"preview", "transcription"}:
            return
        current_preview = getattr(
            self,
            "_preview_model_id",
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or _DEFAULT_PREVIEW_MODEL,
        )
        current_transcription = getattr(
            self,
            "_transcription_model_id",
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or self._default_transcription_model(),
        )
        current_preview, current_transcription = self._sanitize_model_ids(
            current_preview,
            current_transcription,
        )
        preview_model = current_preview
        transcription_model = current_transcription
        if role == "preview":
            preview_model = model_id
        else:
            transcription_model = model_id
        self._apply_model_selection(preview_model, transcription_model)

    def _current_checkout_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _launch_target_menu_state(self) -> dict | None:
        targets = iter_launch_targets()
        if not targets:
            return None
        current_target = current_launch_target_id(self._current_checkout_root())
        return {
            "title": "Launch Target",
            "selected": current_target,
            "items": [
                (target["id"], target["label"], target["enabled"]) for target in targets
            ],
        }

    def _persist_launch_target_selection(self, target_id: str) -> bool:
        return save_selected_launch_target(target_id)

    def _invoke_launch_target_helper(self, target_id: str) -> bool:
        helper_path = Path(__file__).resolve().parents[1] / "scripts" / "launch-target.sh"
        if not helper_path.is_file():
            logger.warning("Launch target helper is missing: %s", helper_path)
            return False
        import subprocess

        subprocess.Popen(
            ["/bin/bash", str(helper_path), target_id],
            cwd=helper_path.parent.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True

    def _apply_launch_target_selection(self, target_id: str) -> None:
        current_target = current_launch_target_id(self._current_checkout_root())
        if not self._persist_launch_target_selection(target_id):
            logger.warning(
                "Skipping launch-target switch because %s could not be persisted",
                target_id,
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save launch target")
            return
        if target_id == current_target:
            return
        logger.info("Switching launch target (handoff): %s -> %s", current_target, target_id)
        if not self._invoke_launch_target_helper(target_id):
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't switch launch target")
            return
        if self._menubar is not None:
            self._menubar.set_status_text(f"Switching to {target_id}…")

    def _apply_model_selection(self, preview_model: str, transcription_model: str) -> None:
        if not self._model_allowed(preview_model):
            logger.warning(
                "Preview model %s not available on this machine (%.0fGB RAM)",
                preview_model,
                _RAM_GB,
            )
            return
        if not self._model_allowed(transcription_model):
            logger.warning(
                "Transcription model %s not available on this machine (%.0fGB RAM)",
                transcription_model,
                _RAM_GB,
            )
            return
        current_preview = getattr(
            self,
            "_preview_model_id",
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or _DEFAULT_PREVIEW_MODEL,
        )
        current_transcription = getattr(
            self,
            "_transcription_model_id",
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or self._default_transcription_model(),
        )
        if (
            preview_model == current_preview
            and transcription_model == current_transcription
        ):
            prefs = self._load_model_preferences()
            if (
                prefs.get("preview_model") != preview_model
                or prefs.get("transcription_model") != transcription_model
            ):
                logger.info(
                    "Repairing stale model preferences without relaunch: preview=%s, transcription=%s",
                    preview_model,
                    transcription_model,
                )
                self._save_model_preferences(preview_model, transcription_model)
            return
        logger.info(
            "Switching models (relaunching): preview %s → %s, transcription %s → %s",
            current_preview,
            preview_model,
            current_transcription,
            transcription_model,
        )
        if not self._save_model_preferences(preview_model, transcription_model):
            logger.warning(
                "Skipping relaunch because the new model selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save model selection")
            return
        os.environ["SPOKE_PREVIEW_MODEL"] = preview_model
        os.environ["SPOKE_TRANSCRIPTION_MODEL"] = transcription_model
        if preview_model == transcription_model:
            os.environ["SPOKE_WHISPER_MODEL"] = preview_model
        else:
            os.environ.pop("SPOKE_WHISPER_MODEL", None)
        self._relaunch()

    def _default_transcription_model(self) -> str:
        if self._model_allowed(_DEFAULT_TRANSCRIPTION_MODEL):
            return _DEFAULT_TRANSCRIPTION_MODEL
        return _DEFAULT_PREVIEW_MODEL

    def _fallback_model_for_role(self, role: str) -> str:
        if role == "preview":
            return _DEFAULT_PREVIEW_MODEL
        return self._default_transcription_model()

    def _sanitize_model_id(self, model_id: str, *, role: str) -> str:
        # Preview-only models (e.g. Parakeet) must not be used for transcription
        if role == "transcription" and model_id in self._PREVIEW_ONLY_MODELS:
            fallback = self._default_transcription_model()
            logger.warning(
                "Model %s is preview-only and cannot be used for transcription — falling back to %s",
                model_id,
                fallback,
            )
            return fallback
        if self._model_allowed(model_id):
            return model_id
        fallback = self._fallback_model_for_role(role)
        logger.warning(
            "%s model %s not available on this machine (%.0fGB RAM) — falling back to %s",
            role.title(),
            model_id,
            _RAM_GB,
            fallback,
        )
        return fallback

    def _sanitize_model_ids(
        self, preview_model: str, transcription_model: str
    ) -> tuple[str, str]:
        return (
            self._sanitize_model_id(preview_model, role="preview"),
            self._sanitize_model_id(transcription_model, role="transcription"),
        )

    def _resolve_whisper_endpoint(self, backend: str) -> tuple[str, str]:
        """Return (url, api_key) for the given backend choice."""
        if backend == "sidecar" and self._whisper_sidecar_url:
            return (self._whisper_sidecar_url, "")
        if backend == "cloud" and self._whisper_cloud_url:
            return (self._whisper_cloud_url, self._whisper_cloud_api_key)
        return ("", "")

    def _resolve_model_ids(self) -> tuple[str, str]:
        prefs = self._load_model_preferences()
        legacy_model = os.environ.get("SPOKE_WHISPER_MODEL")
        raw_preview_model = (
            prefs.get("preview_model")
            or os.environ.get("SPOKE_PREVIEW_MODEL")
            or legacy_model
            or _DEFAULT_PREVIEW_MODEL
        )
        raw_transcription_model = (
            prefs.get("transcription_model")
            or os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or legacy_model
            or self._default_transcription_model()
        )
        preview_model, transcription_model = self._sanitize_model_ids(
            raw_preview_model,
            raw_transcription_model,
        )
        if (
            not os.environ.get("SPOKE_PREVIEW_MODEL")
            and not os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            and not legacy_model
            and (
                prefs.get("preview_model") is not None
                or prefs.get("transcription_model") is not None
            )
            and (
                preview_model != raw_preview_model
                or transcription_model != raw_transcription_model
            )
        ):
            logger.info(
                "Repairing unsupported saved model preferences: preview=%s, transcription=%s",
                preview_model,
                transcription_model,
            )
            self._save_model_preferences(preview_model, transcription_model)
        return preview_model, transcription_model

    def _resolve_local_whisper_settings(self) -> tuple[float | None, bool]:
        prefs = self._load_local_whisper_preferences()
        decode_timeout_raw = os.environ.get("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT")
        if decode_timeout_raw is None and "decode_timeout" in prefs:
            decode_timeout = prefs["decode_timeout"]
        else:
            if decode_timeout_raw is None:
                decode_timeout_raw = _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
            decode_timeout = self._coerce_decode_timeout_setting(decode_timeout_raw)

        eager_eval_raw = os.environ.get("SPOKE_LOCAL_WHISPER_EAGER_EVAL")
        if eager_eval_raw is None:
            if "eager_eval" in prefs:
                eager_eval_raw = prefs["eager_eval"]
            else:
                eager_eval_raw = _DEFAULT_LOCAL_WHISPER_EAGER_EVAL
        eager_eval = self._coerce_eager_eval_setting(eager_eval_raw)
        return decode_timeout, eager_eval

    def _load_local_whisper_preferences(self) -> dict:
        raw_prefs = self._load_preferences()
        prefs = {}
        if "local_whisper_decode_timeout" in raw_prefs:
            prefs["decode_timeout"] = raw_prefs["local_whisper_decode_timeout"]
        if "local_whisper_eager_eval" in raw_prefs:
            prefs["eager_eval"] = raw_prefs["local_whisper_eager_eval"]
        return prefs

    def _preferences_path(self) -> Path:
        override = os.environ.get("SPOKE_MODEL_PREFERENCES_PATH")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Library/Application Support/Spoke/model_preferences.json"

    def _load_preferences(self) -> dict:
        path = self._preferences_path()
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except Exception:
            logger.warning("Failed to read model preferences from %s", path, exc_info=True)
            return {}

    def _load_model_preferences(self) -> dict:
        prefs = self._load_preferences()
        return {
            "preview_model": prefs.get("preview_model"),
            "transcription_model": prefs.get("transcription_model"),
        }

    def _load_command_model_preference(self) -> str | None:
        return self._load_preferences().get("command_model")

    def _load_command_backend_preference(self) -> str | None:
        return self._coerce_command_backend(
            self._load_preferences().get("command_backend")
        )

    def _load_command_sidecar_url_preference(self) -> str | None:
        return self._normalize_command_url(
            self._load_preferences().get("command_sidecar_url")
        )

    @staticmethod
    def _coerce_cloud_provider(value: str | None) -> str | None:
        if value is None:
            return None
        provider = str(value).strip().lower()
        if provider in {"google", "openrouter"}:
            return provider
        return None

    def _load_cloud_provider_preference(self) -> str:
        prefs = self._load_preferences()
        provider = self._coerce_cloud_provider(prefs.get("command_cloud_provider"))
        if provider:
            return provider
        legacy_url = self._normalize_command_url(prefs.get("command_cloud_url")) or ""
        if "openrouter.ai" in _url_host(legacy_url).lower():
            return "openrouter"
        return _DEFAULT_CLOUD_PROVIDER

    def _load_cloud_url_preference(self, provider: str | None = None) -> str | None:
        provider = self._coerce_cloud_provider(provider) or self._load_cloud_provider_preference()
        prefs = self._load_preferences()
        if provider == "openrouter":
            legacy_url = self._normalize_command_url(prefs.get("command_cloud_url"))
            if legacy_url and "openrouter.ai" in _url_host(legacy_url).lower():
                return self._normalize_command_url(
                    prefs.get("command_cloud_openrouter_url") or legacy_url
                )
            return self._normalize_command_url(
                prefs.get("command_cloud_openrouter_url")
            )
        return self._normalize_command_url(
            prefs.get("command_cloud_google_url") or prefs.get("command_cloud_url")
        )

    def _load_cloud_api_key_preference(self, provider: str | None = None) -> str | None:
        provider = self._coerce_cloud_provider(provider) or self._load_cloud_provider_preference()
        prefs = self._load_preferences()
        if provider == "openrouter":
            legacy_url = self._normalize_command_url(prefs.get("command_cloud_url")) or ""
            val = (
                prefs.get("command_cloud_openrouter_api_key")
                or (
                    prefs.get("command_cloud_api_key")
                    if "openrouter.ai" in _url_host(legacy_url).lower()
                    else None
                )
            )
        else:
            val = prefs.get("command_cloud_google_api_key") or prefs.get("command_cloud_api_key")
        return str(val).strip() if val else None

    def _load_cloud_model_preference(self, provider: str | None = None) -> str | None:
        provider = self._coerce_cloud_provider(provider) or self._load_cloud_provider_preference()
        prefs = self._load_preferences()
        if provider == "openrouter":
            legacy_url = self._normalize_command_url(prefs.get("command_cloud_url")) or ""
            val = (
                prefs.get("command_cloud_openrouter_model")
                or (
                    prefs.get("command_cloud_model")
                    if "openrouter.ai" in _url_host(legacy_url).lower()
                    else None
                )
            )
        else:
            val = prefs.get("command_cloud_google_model") or prefs.get("command_cloud_model")
        return str(val).strip() if val else None

    def _save_command_cloud_provider_preference(self, provider: str) -> bool:
        provider = self._coerce_cloud_provider(provider)
        if provider is None:
            return False
        payload = self._load_preferences()
        payload["command_cloud_provider"] = provider
        return self._save_preferences(payload)

    def _save_cloud_preferences(
        self, provider: str, cloud_url: str, cloud_api_key: str, cloud_model: str
    ) -> bool:
        provider = self._coerce_cloud_provider(provider)
        if provider is None:
            return False
        payload = self._load_preferences()
        normalized_url = self._normalize_command_url(cloud_url)
        normalized_key = cloud_api_key.strip() if cloud_api_key else ""
        normalized_model = cloud_model.strip() if cloud_model else ""
        payload["command_cloud_provider"] = provider
        if provider == "openrouter":
            payload["command_cloud_openrouter_url"] = normalized_url
            payload["command_cloud_openrouter_api_key"] = normalized_key
            payload["command_cloud_openrouter_model"] = normalized_model
        else:
            payload["command_cloud_google_url"] = normalized_url
            payload["command_cloud_google_api_key"] = normalized_key
            payload["command_cloud_google_model"] = normalized_model
            # Keep legacy generic keys aligned with Google Cloud because
            # adjacent Gemini/TTS plumbing still reads them.
            payload["command_cloud_url"] = normalized_url
            payload["command_cloud_api_key"] = normalized_key
            payload["command_cloud_model"] = normalized_model
        return self._save_preferences(payload)

    def _save_cloud_model_preference(self, provider: str, model_id: str) -> bool:
        provider = self._coerce_cloud_provider(provider)
        if provider is None:
            return False
        payload = self._load_preferences()
        model_id = model_id.strip()
        if provider == "openrouter":
            payload["command_cloud_openrouter_model"] = model_id
        else:
            payload["command_cloud_google_model"] = model_id
            payload["command_cloud_model"] = model_id
        return self._save_preferences(payload)

    def _save_model_preferences(
        self, preview_model: str, transcription_model: str
    ) -> bool:
        payload = self._load_preferences()
        payload["preview_model"] = preview_model
        payload["transcription_model"] = transcription_model
        return self._save_preferences(payload)

    def _save_command_model_preference(self, model_id: str) -> bool:
        payload = self._load_preferences()
        payload["command_model"] = model_id
        return self._save_preferences(payload)

    def _save_command_backend_preferences(
        self, backend: str, sidecar_url: str | None
    ) -> bool:
        payload = self._load_preferences()
        payload["command_backend"] = self._coerce_command_backend(backend)
        normalized_sidecar_url = self._normalize_command_url(sidecar_url)
        if normalized_sidecar_url:
            payload["command_sidecar_url"] = normalized_sidecar_url
        else:
            payload.pop("command_sidecar_url", None)
        return self._save_preferences(payload)

    def _save_local_whisper_preferences(
        self, decode_timeout: float | None, eager_eval: bool
    ) -> bool:
        payload = self._load_preferences()
        payload["local_whisper_decode_timeout"] = decode_timeout
        payload["local_whisper_eager_eval"] = eager_eval
        return self._save_preferences(payload)

    def _save_preferences(self, payload: dict) -> bool:
        path = self._preferences_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.rename(path)
            return True
        except Exception:
            logger.warning("Failed to save model preferences to %s", path, exc_info=True)
            return False

    def _load_preference(self, key: str):
        return self._load_preferences().get(key)

    def _save_preference(self, key: str, value) -> bool:
        payload = self._load_preferences()
        payload[key] = value
        return self._save_preferences(payload)

    @staticmethod
    def _coerce_command_backend(value: str | None) -> str | None:
        if value is None:
            return None
        backend = str(value).strip().lower()
        if backend in {"local", "sidecar", "cloud"}:
            return backend
        return None

    @staticmethod
    def _normalize_command_url(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().rstrip("/")
        return normalized or None

    def _resolve_command_backend(self) -> tuple[str | None, str | None]:
        pref_backend = self._load_command_backend_preference()
        pref_sidecar_url = self._load_command_sidecar_url_preference()

        if pref_backend == "sidecar" and pref_sidecar_url:
            return "sidecar", pref_sidecar_url
        if pref_backend == "sidecar" and not pref_sidecar_url:
            logger.warning(
                "Saved assistant backend is sidecar but no sidecar URL is configured; falling back to local OMLX"
            )
        if pref_backend == "cloud":
            provider = self._load_cloud_provider_preference()
            cloud_url = self._load_cloud_url_preference(provider) or _default_cloud_url_for_provider(provider)
            return "cloud", cloud_url
        return "local", _DEFAULT_COMMAND_URL

    def _resolve_command_cloud_api_key(
        self, cloud_url: str | None, provider: str | None = None
    ) -> str:
        provider = self._coerce_cloud_provider(provider) or self._load_cloud_provider_preference()
        persisted = self._load_cloud_api_key_preference(provider)
        if persisted:
            return persisted
        if provider == "openrouter":
            provider_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if provider_key:
                return provider_key
        if provider == "google":
            provider_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if provider_key:
                return provider_key
        host = _url_host(cloud_url or "").lower()
        if "openrouter.ai" in host:
            provider_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if provider_key:
                return provider_key
        if "googleapis.com" in host or "generativelanguage.googleapis.com" in host:
            provider_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if provider_key:
                return provider_key
        return os.environ.get("SPOKE_COMMAND_CLOUD_API_KEY", "").strip()

    def _ensure_tts_client(self, *, allow_default_voice: bool = False):
        """Return the active TTS client, building it lazily when appropriate."""
        tts = getattr(self, "_tts_client", None)
        if tts is not None:
            return tts
        tts = self._build_tts_client(allow_default_voice=allow_default_voice)
        if tts is not None:
            self._tts_client = tts
            if isinstance(tts, CloudTTSClient):
                backend_label = "cloud"
            elif isinstance(tts, RemoteTTSClient):
                backend_label = "sidecar"
            else:
                backend_label = "local"
            logger.info("TTS enabled: backend=%s voice=%s", backend_label, getattr(tts, "_voice", ""))
        return tts

    def _build_tts_client(self, *, allow_default_voice: bool = False):
        """Build a TTS client based on backend preference and env vars."""
        voice = self._load_preference("tts_voice") or os.environ.get("SPOKE_TTS_VOICE")
        if self._tts_backend == "cloud":
            api_key = (
                self._load_preference("tts_cloud_api_key")
                or os.environ.get("GEMINI_API_KEY", "")
                or self._load_preference("command_cloud_api_key")
            )
            if not api_key:
                logger.warning("Cannot build cloud TTS client: no API key")
                return None
            cloud_voice = self._load_preference("tts_cloud_voice") or voice or "Aoede"
            cloud_model = (
                self._load_preference("tts_cloud_model")
                or "gemini-2.5-flash-preview-tts"
            )
            return CloudTTSClient(
                api_key=api_key,
                model=cloud_model,
                voice=cloud_voice,
            )
        if self._tts_backend == "sidecar" and self._tts_sidecar_url:
            if not voice:
                return None
            model_id = (
                self._load_preference("tts_sidecar_model")
                or os.environ.get("SPOKE_TTS_MODEL", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
            )
            return RemoteTTSClient(
                base_url=self._tts_sidecar_url,
                model_id=model_id,
                voice=voice,
            )
        model_id = (
            self._load_preference("tts_model")
            or os.environ.get("SPOKE_TTS_MODEL", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
        )
        if not voice and not allow_default_voice:
            return None
        return TTSClient(
            model_id=model_id,
            voice=voice or None,
            gpu_lock=self._local_inference_lock,
        )

    def _discover_tts_sidecar_models(self) -> list[tuple[str, str, bool]]:
        """Fetch available models from the TTS sidecar's /v1/models endpoint."""
        url = getattr(self, "_tts_sidecar_url", "")
        if not url:
            self._tts_server_unreachable = True
            return []
        import urllib.request
        import urllib.error
        models_url = f"{url.rstrip('/')}/v1/models"
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            logger.warning("Failed to fetch TTS sidecar models from %s", models_url, exc_info=True)
            self._tts_server_unreachable = True
            return []
        self._tts_server_unreachable = False
        current_model = ""
        tts = getattr(self, "_tts_client", None)
        if tts is not None:
            current_model = getattr(tts, "_model_id", "")
        entries = data.get("data", []) if isinstance(data, dict) else []
        options = []
        for entry in entries:
            model_id = entry.get("id", "") if isinstance(entry, dict) else str(entry)
            if model_id:
                options.append((model_id, model_id, model_id == current_model))
        return options

    def _discover_tts_sidecar_voices(self, model_name: str = "") -> list[str]:
        """Fetch available voices from the TTS sidecar's /v1/voices endpoint."""
        url = getattr(self, "_tts_sidecar_url", "")
        if not url:
            return []
        import urllib.request
        import urllib.error
        import urllib.parse
        voices_url = f"{url.rstrip('/')}/v1/voices"
        if model_name:
            voices_url += f"?model_name={urllib.parse.quote(model_name)}"
        try:
            req = urllib.request.Request(voices_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            logger.warning("Failed to fetch TTS sidecar voices from %s", voices_url, exc_info=True)
            return []
        entries = data.get("data", []) if isinstance(data, dict) else []
        voices: list[str] = []
        for entry in entries:
            if isinstance(entry, dict):
                voices.extend(entry.get("voices", []))
        return voices

    def _apply_tts_backend_selection(self, backend: str) -> None:
        """Switch TTS backend between 'local', 'sidecar', and 'cloud', then relaunch."""
        if backend == "configure_tts":
            self._configure_tts_sidecar_url()
            return
        if backend == self._tts_backend:
            return
        if backend == "sidecar" and not self._tts_sidecar_url:
            logger.warning("Cannot switch to sidecar: no TTS sidecar URL configured")
            if self._menubar is not None:
                self._menubar.set_status_text("No TTS sidecar URL configured")
            return
        if backend == "cloud":
            api_key = (
                self._load_preference("tts_cloud_api_key")
                or os.environ.get("GEMINI_API_KEY", "")
                or self._load_preference("command_cloud_api_key")
            )
            if not api_key:
                logger.warning("Cannot switch to cloud TTS: no GEMINI_API_KEY")
                if self._menubar is not None:
                    self._menubar.set_status_text("No Gemini API key configured")
                return
            # Default to Aoede voice when switching to cloud for the first time.
            # Use a separate preference key so we don't clobber the local/sidecar voice.
            cloud_voice = self._load_preference("tts_cloud_voice")
            if not cloud_voice or not any(v == cloud_voice for v, _ in GEMINI_VOICES):
                self._save_preference("tts_cloud_voice", "Aoede")
        logger.info("Switching TTS backend: %s -> %s", self._tts_backend, backend)
        self._save_preference("tts_backend", backend)
        self._tts_backend = backend
        self._relaunch()

    def _configure_tts_sidecar_url(self) -> None:
        """Show a dialog to set or change the TTS sidecar URL."""
        current_url = getattr(self, "_tts_sidecar_url", "") or _DEFAULT_TTS_SIDECAR_URL
        alert = NSAlert.new()
        alert.setMessageText_("TTS Sidecar URL")
        alert.setInformativeText_(
            "Enter the base URL for the OpenAI-compatible /v1/audio/speech endpoint."
        )
        field = _PastableTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
        field.setStringValue_(current_url)
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        alert.layout()
        alert.window().makeFirstResponder_(field)
        response = _run_modal_with_paste(alert)
        if response != 1000:
            return
        value = field.stringValue()
        if not isinstance(value, str):
            return
        value = value.strip().rstrip("/")
        if not value:
            return
        self._save_preference("tts_sidecar_url", value)
        self._tts_sidecar_url = value
        logger.info("TTS sidecar URL saved: %s", value)
        if self._tts_backend == "sidecar":
            self._relaunch()
            return
        if self._menubar is not None:
            self._menubar.set_status_text("TTS sidecar URL saved")

    def _apply_transcription_backend_selection(self, backend: str) -> None:
        """Switch Whisper transcription backend between local/sidecar/cloud, then relaunch."""
        if backend == "configure_whisper":
            self._configure_whisper_sidecar_url()
            return
        if backend == "configure_whisper_cloud":
            self._configure_whisper_cloud()
            return
        if backend == getattr(self, "_whisper_backend", "local"):
            return
        if backend == "sidecar" and not getattr(self, "_whisper_sidecar_url", ""):
            logger.warning("Cannot switch to sidecar: no Whisper sidecar URL configured")
            if self._menubar is not None:
                self._menubar.set_status_text("No Whisper sidecar URL configured")
            return
        if backend == "cloud" and not getattr(self, "_whisper_cloud_api_key", ""):
            logger.warning("Cannot switch to cloud: no API key configured")
            if self._menubar is not None:
                self._menubar.set_status_text("No cloud API key configured")
            return
        logger.info("Switching Whisper backend: %s -> %s", self._whisper_backend, backend)
        self._save_preference("whisper_backend", backend)
        self._whisper_backend = backend
        self._relaunch()

    def _apply_preview_backend_selection(self, backend: str) -> None:
        """Switch preview (partials) backend between local/sidecar/cloud, then relaunch."""
        if backend == "configure_whisper":
            self._configure_whisper_sidecar_url()
            return
        if backend == "configure_whisper_cloud":
            self._configure_whisper_cloud()
            return
        if backend == getattr(self, "_preview_backend", "local"):
            return
        if backend == "sidecar" and not getattr(self, "_whisper_sidecar_url", ""):
            logger.warning("Cannot switch preview to sidecar: no Whisper sidecar URL configured")
            if self._menubar is not None:
                self._menubar.set_status_text("No Whisper sidecar URL configured")
            return
        if backend == "cloud" and not getattr(self, "_whisper_cloud_api_key", ""):
            logger.warning("Cannot switch preview to cloud: no API key configured")
            if self._menubar is not None:
                self._menubar.set_status_text("No cloud API key configured")
            return
        logger.info("Switching preview backend: %s -> %s", self._preview_backend, backend)
        self._save_preference("preview_backend", backend)
        self._preview_backend = backend
        self._relaunch()

    def _configure_whisper_sidecar_url(self) -> None:
        """Show a dialog to set or change the Whisper sidecar URL."""
        current_url = getattr(self, "_whisper_sidecar_url", "") or ""
        alert = NSAlert.new()
        alert.setMessageText_("Whisper Sidecar URL")
        alert.setInformativeText_(
            "Enter the base URL for the OpenAI-compatible "
            "/v1/audio/transcriptions endpoint."
        )
        field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
        field.setStringValue_(current_url)
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        response = alert.runModal()
        if response != 1000:
            return
        value = field.stringValue()
        if not isinstance(value, str):
            return
        value = value.strip().rstrip("/")
        if not value:
            return
        self._save_preference("whisper_sidecar_url", value)
        self._whisper_sidecar_url = value
        logger.info("Whisper sidecar URL saved: %s", value)
        if self._whisper_backend == "sidecar":
            self._relaunch()
            return
        if self._menubar is not None:
            self._menubar.set_status_text("Whisper sidecar URL saved")

    def _configure_whisper_cloud(self) -> None:
        """Show a dialog to set or change the OpenAI Whisper cloud API key."""
        current_key = getattr(self, "_whisper_cloud_api_key", "") or ""

        alert = NSAlert.new()
        alert.setMessageText_("Cloud Whisper (OpenAI)")
        alert.setInformativeText_(
            "Enter your OpenAI API key for cloud transcription.\n"
            "Uses the whisper-1 model at api.openai.com."
        )
        field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
        field.setStringValue_(current_key)
        field.setPlaceholderString_("sk-...")
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        response = alert.runModal()
        if response != 1000:
            return
        value = field.stringValue()
        if not isinstance(value, str):
            return
        value = value.strip()
        if not value:
            return
        self._save_preference("whisper_cloud_api_key", value)
        self._save_preference("whisper_cloud_url", _DEFAULT_WHISPER_CLOUD_URL)
        self._save_preference("whisper_cloud_model", _DEFAULT_WHISPER_CLOUD_MODEL)
        self._whisper_cloud_api_key = value
        self._whisper_cloud_url = _DEFAULT_WHISPER_CLOUD_URL
        self._whisper_cloud_model = _DEFAULT_WHISPER_CLOUD_MODEL
        logger.info("Whisper cloud API key saved")
        # Auto-switch to cloud if not already on it.
        if self._whisper_backend != "cloud":
            self._save_preference("whisper_backend", "cloud")
            self._whisper_backend = "cloud"
        self._relaunch()

    def _get_client(self, whisper_url: str, model_id: str, api_key: str = ""):
        cache_key = (whisper_url, model_id, api_key)
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]
        client = self._build_client(whisper_url, model_id, api_key=api_key)
        self._client_cache[cache_key] = client
        return client

    def _build_client(self, whisper_url: str, model_id: str, api_key: str = ""):
        if whisper_url:
            logger.info("Using remote transcription: %s (%s)", whisper_url, model_id)
            return TranscriptionClient(
                base_url=whisper_url, model=model_id, api_key=api_key,
            )
        if model_id == _PARAKEET_MODEL_ID:
            model_dir = self._resolve_parakeet_model_dir()
            logger.info("Using Parakeet CoreML: %s", model_dir)
            return ParakeetCoreMLClient(model_dir=model_dir)
        if model_id.startswith("Qwen/"):
            logger.info("Using local Qwen3 ASR: %s", model_id)
            return LocalQwenClient(model=model_id)
        logger.info("Using local transcription: %s", model_id)
        return LocalTranscriptionClient(
            model=model_id,
            decode_timeout=getattr(
                self,
                "_local_whisper_decode_timeout",
                _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
            ),
            eager_eval=getattr(
                self,
                "_local_whisper_eager_eval",
                _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
            ),
        )

    def _resolve_parakeet_model_dir(self) -> Path:
        """Return the Parakeet CoreML model directory.

        Checks SPOKE_PARAKEET_MODEL_DIR env var first, then the default HF
        cache location (~/.cache/huggingface/hub/models--FluidInference--parakeet-ctc-110m-coreml/snapshots/).
        Falls back to an empty path so the client can log a clear warning.
        """
        env_dir = os.environ.get("SPOKE_PARAKEET_MODEL_DIR", "")
        if env_dir:
            return Path(env_dir).expanduser()

        # Try HF hub cache — look for the most recent snapshot
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        snapshots = hf_cache / "models--FluidInference--parakeet-ctc-110m-coreml" / "snapshots"
        if snapshots.is_dir():
            candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for candidate in candidates:
                if (candidate / "AudioEncoder.mlmodelc").exists():
                    return candidate

        # No model found — return a path that will produce a clear warning
        return Path("/nonexistent/parakeet-ctc-110m-coreml")

    def _discover_command_models(
        self, selected_model: str
    ) -> list[tuple[str, str, bool]]:
        command_backend = getattr(self, "_command_backend", "local")
        command_cloud_provider = getattr(
            self,
            "_command_cloud_provider",
            self._load_cloud_provider_preference(),
        )
        server_model_ids: list[str] = []
        server_reachable = True
        if self._command_client is not None:
            try:
                server_model_ids = self._command_client.list_models()
                server_reachable = True
            except Exception:
                server_reachable = False
                logger.warning("Failed to fetch assistant models from OMLX", exc_info=True)
        if command_backend in ("sidecar", "cloud"):
            self._command_server_unreachable = not server_reachable and not server_model_ids
            model_ids = server_model_ids or ([selected_model] if selected_model else [])
        else:
            if not server_reachable:
                # Don't list local-disk models when the server is down —
                # they'll appear selectable but fail on every request.
                logger.info("Local model server unreachable — suppressing disk-only model list")
                model_ids = []
            else:
                local_model_dir = Path(
                    os.environ.get("SPOKE_COMMAND_MODEL_DIR", str(_DEFAULT_COMMAND_MODEL_DIR))
                ).expanduser()
                local_model_ids = _iter_local_command_model_ids(local_model_dir)
                if local_model_ids:
                    local_model_set = set(local_model_ids)
                    model_ids = [
                        model_id
                        for model_id in server_model_ids
                        if model_id in local_model_set
                    ]
                    model_ids.extend(
                        model_id for model_id in local_model_ids if model_id not in model_ids
                    )
                else:
                    model_ids = server_model_ids
        if command_backend == "cloud" and command_cloud_provider == "openrouter":
            model_ids = sorted(model_ids, key=str.casefold)
        seen: set[str] = set()
        options = []
        for model_id in model_ids:
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            options.append((model_id, model_id, model_id == selected_model))
        return options

    def _seed_command_model_options(
        self, selected_model: str
    ) -> list[tuple[str, str, bool]]:
        """Seed the Assistant menu — sidecar/cloud queries /v1/models, local uses disk."""
        command_backend = getattr(self, "_command_backend", "local")
        command_cloud_provider = getattr(
            self,
            "_command_cloud_provider",
            self._load_cloud_provider_preference(),
        )
        if command_backend in ("sidecar", "cloud"):
            if self._command_client is not None:
                try:
                    server_model_ids = self._command_client.list_models()
                    if server_model_ids:
                        self._command_server_unreachable = False
                        if command_backend == "cloud" and command_cloud_provider == "openrouter":
                            server_model_ids = sorted(server_model_ids, key=str.casefold)
                        return [
                            (mid, mid, mid == selected_model)
                            for mid in server_model_ids
                        ]
                except Exception:
                    self._command_server_unreachable = True
                    logger.warning(
                        "Model seed failed — falling back to persisted model",
                        exc_info=True,
                    )
            else:
                self._command_server_unreachable = True
            return [(selected_model, selected_model, True)] if selected_model else []
        # Local backend: check server reachability before listing disk models
        server_reachable = False
        if self._command_client is not None:
            try:
                self._command_client.list_models()
                server_reachable = True
            except Exception:
                logger.info("Local model server unreachable at seed — suppressing model list")
        if not server_reachable:
            return []
        local_model_dir = Path(
            os.environ.get("SPOKE_COMMAND_MODEL_DIR", str(_DEFAULT_COMMAND_MODEL_DIR))
        ).expanduser()
        local_model_ids = _iter_local_command_model_ids(local_model_dir)
        if local_model_ids:
            return [
                (model_id, model_id, model_id == selected_model)
                for model_id in local_model_ids
            ]
        return [(selected_model, selected_model, True)] if selected_model else []

    def _refresh_command_model_options_async(self) -> None:
        if self._command_client is None or self._command_models_refresh_in_flight:
            return
        self._command_models_refresh_in_flight = True

        def _load():
            _record_runtime_phase("command_models.refresh.start")
            options = self._discover_command_models(self._command_model_id)
            _record_runtime_phase(
                "command_models.refresh.succeeded",
                option_count=len(options),
            )
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandModelsDiscovered:",
                {"options": options},
                False,
            )

        threading.Thread(
            target=_load,
            daemon=True,
            name="command-model-refresh",
        ).start()

    def commandModelsDiscovered_(self, payload: dict) -> None:
        command_backend = getattr(self, "_command_backend", "local")
        self._command_models_refresh_in_flight = False
        options = payload.get("options") or []
        self._command_model_options = options
        if (
            command_backend in ("sidecar", "cloud")
            and options
            and self._command_model_id not in {model_id for model_id, _, _ in options}
        ):
            healed_model_id = options[0][0]
            logger.info(
                "Healing stale sidecar assistant model without relaunch: %s -> %s",
                self._command_model_id,
                healed_model_id,
            )
            self._command_model_id = healed_model_id
            if self._command_client is not None:
                self._command_client._model = healed_model_id
            if not self._save_command_model_preference(healed_model_id):
                logger.warning(
                    "Failed to persist healed sidecar assistant model: %s",
                    healed_model_id,
                )
            if command_backend == "cloud":
                self._save_cloud_model_preference(
                    getattr(self, "_command_cloud_provider", self._load_cloud_provider_preference()),
                    healed_model_id,
                )
            self._command_model_options = [
                (model_id, label, model_id == healed_model_id)
                for model_id, label, _selected in options
            ]
        if self._menubar is not None:
            self._menubar.refresh_menu()

    def _apply_command_model_selection(self, model_id: str) -> None:
        current_model = self._command_model_id
        if model_id == current_model:
            if self._load_command_model_preference() != model_id:
                logger.info(
                    "Repairing stale assistant model preference without relaunch: %s",
                    model_id,
                )
                self._save_command_model_preference(model_id)
            if getattr(self, "_command_backend", None) == "cloud":
                self._save_cloud_model_preference(
                    getattr(self, "_command_cloud_provider", self._load_cloud_provider_preference()),
                    model_id,
                )
            return
        logger.info(
            "Switching assistant model (relaunching): %s -> %s",
            current_model,
            model_id,
        )
        if not self._save_command_model_preference(model_id):
            logger.warning(
                "Skipping relaunch because the assistant model selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save model selection")
            return
        if getattr(self, "_command_backend", None) == "cloud":
            self._save_cloud_model_preference(
                getattr(self, "_command_cloud_provider", self._load_cloud_provider_preference()),
                model_id,
            )
        self._command_model_id = model_id
        self._relaunch()

    def _apply_command_backend_selection(self, selection: str) -> None:
        if selection == "configure":
            self._configure_command_sidecar_url()
            return
        if selection == "configure_cloud_google":
            self._configure_cloud_endpoint("google")
            return
        if selection == "configure_cloud_openrouter":
            self._configure_cloud_endpoint("openrouter")
            return
        target_cloud_provider = None
        if selection == "cloud_google":
            target_cloud_provider = "google"
            selection = "cloud"
        elif selection == "cloud_openrouter":
            target_cloud_provider = "openrouter"
            selection = "cloud"
        if selection not in {"local", "sidecar", "cloud"}:
            return

        current_backend = getattr(self, "_command_backend", _DEFAULT_COMMAND_BACKEND)
        current_cloud_provider = getattr(
            self, "_command_cloud_provider", self._load_cloud_provider_preference()
        )
        current_url = self._normalize_command_url(getattr(self, "_command_url", None))
        current_sidecar_url = self._normalize_command_url(
            getattr(self, "_command_sidecar_url", None)
        )

        target_sidecar_url = current_sidecar_url
        if selection == "sidecar" and target_sidecar_url is None:
            target_sidecar_url = self._prompt_for_command_sidecar_url("")
            if target_sidecar_url is None:
                if self._menubar is not None:
                    self._menubar.set_status_text("Assistant sidecar URL required")
                return

        if selection == "cloud":
            target_cloud_provider = (
                self._coerce_cloud_provider(target_cloud_provider)
                or current_cloud_provider
            )
            cloud_url = (
                self._load_cloud_url_preference(target_cloud_provider)
                or _default_cloud_url_for_provider(target_cloud_provider)
            )
            target_url = cloud_url
        elif selection == "sidecar":
            target_url = target_sidecar_url
        else:
            target_url = _DEFAULT_COMMAND_URL
        persisted_sidecar_url = (
            target_sidecar_url if selection == "sidecar" else current_sidecar_url
        )

        if (
            selection == current_backend
            and target_url == current_url
            and (
                selection != "cloud"
                or target_cloud_provider == current_cloud_provider
            )
        ):
            if (
                self._load_command_backend_preference() != selection
                or self._load_command_sidecar_url_preference() != persisted_sidecar_url
            ):
                self._save_command_backend_preferences(
                    selection, persisted_sidecar_url
                )
            if selection == "cloud":
                self._save_command_cloud_provider_preference(target_cloud_provider)
            return

        logger.info(
            "Switching assistant backend (relaunching): %s -> %s (%s)",
            current_backend,
            selection,
            target_url,
        )
        if not self._save_command_backend_preferences(
            selection, persisted_sidecar_url
        ):
            logger.warning(
                "Skipping relaunch because the assistant backend selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save assistant backend")
            return

        if selection == "cloud":
            self._save_command_cloud_provider_preference(target_cloud_provider)

        self._command_backend = selection
        if selection == "cloud":
            self._command_cloud_provider = target_cloud_provider
        self._command_url = target_url
        self._command_sidecar_url = persisted_sidecar_url
        self._relaunch()

    def _configure_command_sidecar_url(self) -> None:
        current_backend = getattr(self, "_command_backend", _DEFAULT_COMMAND_BACKEND)
        current_sidecar_url = self._normalize_command_url(
            getattr(self, "_command_sidecar_url", None)
        )
        sidecar_url = self._prompt_for_command_sidecar_url(current_sidecar_url or "")
        if sidecar_url is None:
            return
        if not self._save_command_backend_preferences(current_backend, sidecar_url):
            logger.warning("Couldn't persist assistant sidecar URL")
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save assistant backend")
            return

        self._command_sidecar_url = sidecar_url
        if current_backend == "sidecar":
            self._command_url = sidecar_url
            self._relaunch()
            return
        if self._menubar is not None:
            self._menubar.set_status_text("Assistant sidecar URL saved")

    def _prompt_for_command_sidecar_url(self, current_url: str) -> str | None:
        alert = NSAlert.new()
        alert.setMessageText_("Assistant Sidecar URL")
        alert.setInformativeText_(
            "Enter the OpenAI-compatible base URL for the assistant sidecar."
        )
        field = _PastableTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
        field.setStringValue_(current_url)
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        alert.layout()
        alert.window().makeFirstResponder_(field)
        response = _run_modal_with_paste(alert)
        if response != 1000:
            return None
        value = field.stringValue()
        if not isinstance(value, str):
            return None
        return self._normalize_command_url(value)

    def _configure_cloud_endpoint(self, provider: str | None = None) -> None:
        provider = self._coerce_cloud_provider(provider) or getattr(
            self, "_command_cloud_provider", self._load_cloud_provider_preference()
        )
        current_url = self._load_cloud_url_preference(provider) or _default_cloud_url_for_provider(provider)
        current_key = self._load_cloud_api_key_preference(provider) or ""
        current_model = self._load_cloud_model_preference(provider) or _default_cloud_model_for_provider(provider)

        alert = NSAlert.new()
        provider_label = _cloud_provider_label(provider)
        alert.setMessageText_(f"{provider_label} Assistant Endpoint")
        if provider == "openrouter":
            alert.setInformativeText_(
                "OpenAI-compatible endpoint for OpenRouter assistant.\n"
                "Example: https://openrouter.ai/api/v1"
            )
        else:
            alert.setInformativeText_(
                "OpenAI-compatible endpoint for Google Cloud assistant.\n"
                "Example: https://generativelanguage.googleapis.com/v1beta/openai"
            )

        from AppKit import NSView
        container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 90))

        url_field = _PastableTextField.alloc().initWithFrame_(NSMakeRect(0, 60, 320, 24))
        url_field.setStringValue_(current_url)
        url_field.setPlaceholderString_("Endpoint URL")
        container.addSubview_(url_field)

        key_field = _PastableTextField.alloc().initWithFrame_(NSMakeRect(0, 30, 320, 24))
        key_field.setStringValue_(current_key)
        key_field.setPlaceholderString_("API Key")
        container.addSubview_(key_field)

        model_field = _PastableTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
        model_field.setStringValue_(current_model)
        model_field.setPlaceholderString_(
            "Model (e.g. stepfun/step-3.5-flash:free)"
            if provider == "openrouter"
            else "Model (e.g. gemini-2.5-flash)"
        )
        container.addSubview_(model_field)

        alert.setAccessoryView_(container)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        alert.layout()
        alert.window().makeFirstResponder_(url_field)
        response = _run_modal_with_paste(alert)
        if response != 1000:
            return

        cloud_url = self._normalize_command_url(
            url_field.stringValue() if isinstance(url_field.stringValue(), str) else ""
        )
        cloud_key = str(key_field.stringValue()).strip() if key_field.stringValue() else ""
        cloud_model = str(model_field.stringValue()).strip() if model_field.stringValue() else ""

        if not cloud_url:
            if self._menubar is not None:
                self._menubar.set_status_text("Cloud endpoint URL required")
            return

        if not self._save_cloud_preferences(provider, cloud_url, cloud_key, cloud_model):
            logger.warning("Couldn't persist cloud endpoint config")
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save cloud config")
            return

        self._save_command_cloud_provider_preference(provider)
        if not self._save_command_backend_preferences(
            "cloud", self._normalize_command_url(getattr(self, "_command_sidecar_url", None))
        ):
            logger.warning("Couldn't persist cloud backend selection")
            return

        self._command_backend = "cloud"
        self._command_cloud_provider = provider
        self._command_url = cloud_url
        self._relaunch()

    def _apply_tts_model_selection(self, model_id: str) -> None:
        tts = getattr(self, "_tts_client", None)
        current_model = tts._model_id if tts else None
        if model_id == current_model:
            return
        logger.info(
            "Switching TTS model (relaunching): %s -> %s",
            current_model,
            model_id,
        )
        payload = self._load_preferences()
        tts_backend = getattr(self, "_tts_backend", "local")
        if tts_backend == "cloud":
            payload["tts_cloud_model"] = model_id
        elif tts_backend == "sidecar":
            payload["tts_sidecar_model"] = model_id
        else:
            payload["tts_model"] = model_id
            # Reset voice when switching between incompatible model families
            # (e.g. OmniVoice prompts are not valid Voxtral voice names and
            # vice versa).
            # Reset voice when switching between models with different
            # voice presets (e.g. Voxtral voices ≠ Kokoro voices ≠ OmniVoice
            # prompts).  Set to the first preset of the new model.
            old_presets = _voice_presets_for_model(current_model)
            new_presets = _voice_presets_for_model(model_id)
            if old_presets is not new_presets:
                if new_presets:
                    payload["tts_voice"] = new_presets[0][0]
                else:
                    payload.pop("tts_voice", None)
                logger.info(
                    "Reset tts_voice to %r on model switch (%s -> %s)",
                    payload.get("tts_voice"),
                    current_model,
                    model_id,
                )
        if not self._save_preferences(payload):
            logger.warning(
                "Skipping relaunch because the TTS model selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save TTS model selection")
            return
        self._relaunch()

    def _apply_tts_voice_selection(self, selection: str) -> None:
        """Handle TTS voice menu selections."""
        if selection == "configure_voice":
            self._configure_tts_voice()
            return
        # Direct voice name selection from discovered voices
        tts = getattr(self, "_tts_client", None)
        current_voice = getattr(tts, "_voice", "") if tts else ""
        if selection == current_voice:
            return
        # Write to backend-scoped key so cloud voice doesn't clobber local/sidecar
        voice_key = "tts_cloud_voice" if getattr(self, "_tts_backend", "local") == "cloud" else "tts_voice"
        self._save_preference(voice_key, selection)
        logger.info("TTS voice changed: %s -> %s (key=%s)", current_voice, selection, voice_key)
        self._relaunch()

    def _configure_tts_voice(self) -> None:
        """Show a dialog to set the TTS voice name."""
        tts = getattr(self, "_tts_client", None)
        current_voice = getattr(tts, "_voice", "") if tts else ""
        if not current_voice:
            current_voice = self._load_preference("tts_voice") or os.environ.get("SPOKE_TTS_VOICE", "")
        if tts is not None:
            current_model = getattr(tts, "_model_id", "")
        elif getattr(self, "_tts_backend", "local") == "sidecar":
            current_model = (
                self._load_preference("tts_sidecar_model")
                or os.environ.get("SPOKE_TTS_MODEL", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
            )
        else:
            current_model = (
                self._load_preference("tts_model")
                or os.environ.get("SPOKE_TTS_MODEL", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
            )
        is_omnivoice = _is_omnivoice_tts_model(current_model) and getattr(self, "_tts_backend", "local") != "sidecar"
        alert = NSAlert.new()
        alert.setMessageText_("TTS Prompt" if is_omnivoice else "TTS Voice")
        alert.setInformativeText_(
            _omnivoice_prompt_lexicon_text()
            if is_omnivoice
            else "Enter the voice name to use for TTS synthesis."
        )
        field = _PastableTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 24))
        field.setStringValue_(current_voice)
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        alert.layout()
        alert.window().makeFirstResponder_(field)
        response = _run_modal_with_paste(alert)
        if response != 1000:
            return
        value = field.stringValue()
        if not isinstance(value, str):
            return
        value = value.strip()
        if not value:
            return
        self._save_preference("tts_voice", value)
        logger.info("TTS voice saved: %s", value)
        self._relaunch()

    def _prepare_clients(self) -> None:
        seen_clients = []
        ordered_clients = [
            ("transcription", getattr(self, "_transcription_model_id", None), self._client),
            ("preview", getattr(self, "_preview_model_id", None), self._preview_client),
        ]
        for role, model_id, client in ordered_clients:
            if client is None or any(existing is client for existing in seen_clients):
                continue
            seen_clients.append(client)
            if model_id is not None and not self._model_allowed(model_id):
                raise RuntimeError(
                    f"{role.title()} model {model_id} is not supported on this machine."
                )
            prepare = getattr(client, "prepare", None)
            if callable(prepare):
                with self._local_inference_context(client):
                    prepare()
                _record_runtime_phase(
                    "client.prepare.succeeded",
                    role=role,
                    model=model_id,
                    client_type=type(client).__name__,
                )

    def _close_clients(self) -> None:
        seen_clients = []
        for client in list(getattr(self, "_client_cache", {}).values()) + [
            getattr(self, "_client", None),
            getattr(self, "_preview_client", None),
        ]:
            if client is None or any(existing is client for existing in seen_clients):
                continue
            seen_clients.append(client)
            close = getattr(client, "close", None)
            if callable(close):
                close()

    # ── Heartbeat & model TTL ─────────────────────────────────

    def _register_loaded_models(self) -> None:
        """Tell the heartbeat manager which models are currently resident."""
        hb = getattr(self, "_heartbeat", None)
        if hb is None:
            return
        for client in self._iter_unique_clients():
            loaded = getattr(client, "is_loaded", False)
            if loaded:
                model_id = getattr(client, "_model", None) or getattr(client, "_model_id", None)
                if model_id:
                    hb.register_model(model_id)

    def _start_heartbeat_timer(self) -> None:
        """Schedule the heartbeat timer on the main-thread run loop."""
        from Foundation import NSTimer

        if getattr(self, "_heartbeat_timer", None) is not None:
            return
        hb = getattr(self, "_heartbeat", None)
        if hb is None:
            return
        self._heartbeat_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            HEARTBEAT_INTERVAL_S,
            self,
            "heartbeatTick:",
            None,
            True,
        )
        # Write the first heartbeat immediately.
        self._heartbeat.tick()
        logger.info(
            "Heartbeat started (interval=%.0fs, model_ttl=%.0fs)",
            HEARTBEAT_INTERVAL_S,
            self._heartbeat._ttl,
        )

    def heartbeatTick_(self, _timer) -> None:
        """NSTimer callback — write heartbeat, evict expired models."""
        # Ensure newly loaded models (e.g. TTS after async warmup) get registered.
        self._register_loaded_models()
        expired = self._heartbeat.tick()
        if expired:
            logger.info("Heartbeat TTL eviction: %s", expired)

    def _evict_model(self, model_id: str) -> None:
        """Eviction callback from HeartbeatManager — unload a model by ID."""
        for client in self._iter_unique_clients():
            client_model = getattr(client, "_model", None) or getattr(client, "_model_id", None)
            if client_model == model_id:
                unload = getattr(client, "unload", None)
                if callable(unload):
                    with self._local_inference_context(client):
                        unload()
                    self._heartbeat.unregister_model(model_id)
                    self._heartbeat.clear_metal_cache()
                    logger.info("Evicted model %s", model_id)
                    return

    def _touch_model(self, client) -> None:
        """Update last-use timestamp for a client's model."""
        hb = getattr(self, "_heartbeat", None)
        if hb is None:
            return
        model_id = getattr(client, "_model", None) or getattr(client, "_model_id", None)
        if model_id:
            hb.touch(model_id)

    def _iter_unique_clients(self):
        """Yield each unique client object (deduplicated)."""
        seen = []
        candidates = list(getattr(self, "_client_cache", {}).values()) + [
            getattr(self, "_client", None),
            getattr(self, "_preview_client", None),
            getattr(self, "_tts_client", None),
        ]
        for client in candidates:
            if client is None or any(c is client for c in seen):
                continue
            seen.append(client)
            yield client

    def _local_whisper_controls_available(self) -> bool:
        if not getattr(self, "_local_mode", False):
            return False
        model_ids = [
            getattr(
                self,
                "_preview_model_id",
                os.environ.get("SPOKE_PREVIEW_MODEL")
                or os.environ.get("SPOKE_WHISPER_MODEL")
                or _DEFAULT_PREVIEW_MODEL,
            ),
            getattr(
                self,
                "_transcription_model_id",
                os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
                or os.environ.get("SPOKE_WHISPER_MODEL")
                or self._default_transcription_model(),
            ),
        ]
        return any(model_id and not model_id.startswith("Qwen/") for model_id in model_ids)

    def _local_whisper_eager_eval_available(self) -> bool:
        return self._local_whisper_controls_available() and supports_eager_eval()

    def _toggle_local_whisper_setting(self, setting: str) -> None:
        if not self._local_whisper_controls_available():
            return
        if setting == "eager_eval" and not self._local_whisper_eager_eval_available():
            logger.info(
                "Ignoring eager_eval toggle because the installed mlx-whisper build does not support it yet"
            )
            return
        decode_timeout = getattr(
            self,
            "_local_whisper_decode_timeout",
            _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
        )
        eager_eval = getattr(
            self,
            "_local_whisper_eager_eval",
            _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
        )
        if setting == "decode_timeout":
            decode_timeout = (
                None
                if decode_timeout is not None
                else _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
            )
        elif setting == "eager_eval":
            eager_eval = not eager_eval
        else:
            return
        self._apply_local_whisper_settings(decode_timeout, eager_eval)

    def _apply_local_whisper_settings(
        self, decode_timeout: float | None, eager_eval: bool
    ) -> None:
        current_timeout = getattr(
            self,
            "_local_whisper_decode_timeout",
            _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
        )
        current_eager = getattr(
            self,
            "_local_whisper_eager_eval",
            _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
        )
        if decode_timeout == current_timeout and eager_eval == current_eager:
            return
        logger.info(
            "Switching local Whisper settings (relaunching): decode_timeout %s -> %s, eager_eval %s -> %s",
            current_timeout,
            decode_timeout,
            current_eager,
            eager_eval,
        )
        self._local_whisper_decode_timeout = decode_timeout
        self._local_whisper_eager_eval = eager_eval
        self._save_local_whisper_preferences(decode_timeout, eager_eval)
        self._relaunch()

    @staticmethod
    def _coerce_decode_timeout_setting(value) -> float | None:
        if value is None:
            return _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
        if isinstance(value, (int, float)):
            return None if value <= 0 else float(value)
        normalized = str(value).strip().lower()
        if normalized in {"", "default"}:
            return _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
        if normalized in {"off", "none", "false"}:
            return None
        try:
            timeout = float(normalized)
        except ValueError:
            logger.warning(
                "Invalid SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT=%r; using default %.1fs",
                value,
                _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
            )
            return _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
        return None if timeout <= 0 else timeout

    @staticmethod
    def _coerce_eager_eval_setting(value) -> bool:
        if value is None:
            return _DEFAULT_LOCAL_WHISPER_EAGER_EVAL
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _format_decode_timeout_env(value: float | None) -> str:
        if value is None:
            return "off"
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)

    def _relaunch(self) -> None:
        _record_runtime_phase("app.relaunch")
        os.environ["SPOKE_PARENT_LAUNCH_ID"] = _PROCESS_LAUNCH_ID
        os.environ["SPOKE_LAUNCH_ID"] = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._detector.uninstall()
        self._preview_active = False
        self._close_clients()
        os.execv(sys.executable, [sys.executable, "-m", "spoke"])

    def _local_inference_context(self, client):
        lock = getattr(self, "_local_inference_lock", None)
        if lock is None or isinstance(client, TranscriptionClient):
            return nullcontext()
        # Touch the model on each local inference so TTL doesn't expire
        # while the model is actively in use.
        self._touch_model(client)
        return lock

    def _inject_result_text(self, text: str, status_text: str) -> None:
        # Remove the overlay from screen so it doesn't appear in the
        # verification screenshot or mask the focused element.
        if self._overlay is not None:
            self._overlay.order_out()

        # Save clipboard state before inject_text overwrites it, in case we
        # need to enter recovery mode after OCR verification.
        self._pre_paste_clipboard = save_pasteboard()
        # Give the target app a brief moment to refocus after the overlay
        # disappears before we synthesize Cmd+V.
        self._result_pending_inject = (text, status_text)
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, "resultInjectDelayed:", None, False
        )

    def resultInjectDelayed_(self, timer) -> None:
        """Paste normal-path text after a short post-overlay refocus delay."""
        pending = getattr(self, "_result_pending_inject", None)
        self._result_pending_inject = None
        if pending is None:
            return
        text, status_text = pending

        def _on_clipboard_restored():
            if self._menubar is not None:
                # Resume hands-free if it was paused for a spacebar hold
                hf = getattr(self, "_handsfree", None)
                if getattr(self, "_handsfree_paused_for_hold", False) and hf is not None:
                    self._handsfree_paused_for_hold = False
                    hf._start_dictating()
                else:
                    self._menubar.set_status_text("Ready — hold spacebar")

        inject_text(text, on_restored=_on_clipboard_restored)
        if self._menubar is not None:
            self._menubar.set_status_text(status_text)

        # Schedule background OCR verification to confirm the paste landed.
        # If it didn't, we'll enter recovery mode.
        self._verify_paste_text = text
        self._verify_paste_attempt = 0
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.15, self, "verifyPaste:", None, False
        )

    def _enter_recovery_mode(self, text: str) -> None:
        """Paste verification failed — enter the tray automatically.

        Recovery mode is tray entered automatically when paste verification
        fails. The tray subsumes recovery as a special case.
        """
        # Save clipboard state for potential restore
        pre = getattr(self, "_pre_paste_clipboard", _NOT_CAPTURED)
        if pre is not _NOT_CAPTURED:
            self._recovery_saved_clipboard = pre
        else:
            self._recovery_saved_clipboard = save_pasteboard()
        self._pre_paste_clipboard = _NOT_CAPTURED

        # Put transcription on pasteboard for manual paste
        set_pasteboard_only(text)

        # Enter tray with the failed paste text
        self._enter_tray(text)

    def _recovery_retry_insert(self) -> None:
        """Spacebar retry: attempt Insert from recovery, OCR verify after.

        Always attempts the paste (just like the normal flow), then uses
        OCR to verify. If it didn't land, bounces the overlay back.
        If it landed, clears recovery state.
        """
        if self._recovery_text is None:
            return

        # No focused element at all → bounce immediately, don't waste time
        if not has_focused_text_input():
            logger.info("Recovery retry — no focused element, bouncing")
            if self._overlay is not None:
                self._overlay.bounce()
            return

        text = self._recovery_text or ""

        # Dismiss overlay so it doesn't appear in the OCR screenshot
        if self._overlay is not None:
            self._overlay.order_out()

        # Paste
        inject_text(text)

        # OCR verify — reuse the same verification pipeline
        self._verify_paste_text = text
        self._verify_paste_attempt = 0
        # Override the verify result handler to bounce-back on failure
        # instead of entering recovery (we're already in recovery)
        self._recovery_retry_pending = True
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.15, self, "verifyPaste:", None, False
        )

    def _on_recovery_dismiss(self) -> None:
        """Dismiss button: restore clipboard and hide overlay."""
        if self._recovery_text is None:
            return  # already dismissed
        restore_pasteboard(self._recovery_saved_clipboard)
        if self._overlay is not None:
            self._overlay.dismiss_recovery()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")
        self._clear_recovery_state()

    def _on_recovery_insert(self) -> None:
        """Insert button: same as spacebar retry — check first, bounce on failure."""
        self._recovery_retry_insert()

    def doRecoveryInsert_(self, timer) -> None:
        """Delayed paste after recovery Insert — gives target app time to refocus."""
        pending = getattr(self, "_recovery_pending_insert", None)
        if pending is None:
            return
        text, saved = pending
        self._recovery_pending_insert = None

        if not has_focused_text_input():
            # Still no text field — the target app didn't refocus in time.
            # Re-enter recovery mode rather than losing the text.
            logger.warning("Insert failed — target app did not refocus, re-entering recovery")
            self._enter_recovery_mode(text)
            return

        def _on_restored():
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")

        inject_text(text, on_restored=_on_restored)
        if self._menubar is not None:
            self._menubar.set_status_text("Pasted!")
        self._clear_recovery_state()

    def _on_recovery_clipboard_toggle(self) -> None:
        """Clipboard button: toggle between transcription and original clipboard."""
        if self._recovery_clipboard_state == "idle":
            # First click: transcription is already on clipboard from _enter_recovery_mode
            self._recovery_clipboard_state = "transcription_on_clipboard"
            # Show preview of what was clobbered
            old_text = self._get_clipboard_preview_text(self._recovery_saved_clipboard)
            if self._overlay is not None:
                self._overlay.set_clipboard_preview(old_text)
        elif self._recovery_clipboard_state == "transcription_on_clipboard":
            # Restore old clipboard, show transcription preview
            restore_pasteboard(self._recovery_saved_clipboard)
            self._recovery_clipboard_state = "original_on_clipboard"
            if self._overlay is not None:
                self._overlay.set_clipboard_preview(self._recovery_text or "")
        elif self._recovery_clipboard_state == "original_on_clipboard":
            # Put transcription back on clipboard, show old preview
            set_pasteboard_only(self._recovery_text or "")
            self._recovery_clipboard_state = "transcription_on_clipboard"
            old_text = self._get_clipboard_preview_text(self._recovery_saved_clipboard)
            if self._overlay is not None:
                self._overlay.set_clipboard_preview(old_text)

    def _cancel_recovery(self) -> None:
        """Cancel recovery/tray overlay if active. Restores original clipboard."""
        if getattr(self, "_recovery_text", None) is not None:
            restore_pasteboard(getattr(self, "_recovery_saved_clipboard", None))
            if self._overlay is not None:
                # dismiss_recovery handles button-based recovery mode cleanup.
                # For tray mode (show_tray), we just need to hide the overlay.
                if self._overlay._recovery_mode:
                    self._overlay.dismiss_recovery()
                else:
                    self._overlay.hide()
            self._clear_recovery_state()

    def _clear_recovery_state(self) -> None:
        """Reset recovery state fields."""
        self._recovery_saved_clipboard = None
        self._recovery_text = None
        self._recovery_clipboard_state = "idle"
        self._recovery_pending_insert = None

    @staticmethod
    def _get_clipboard_preview_text(saved: list[tuple[str, bytes]] | None) -> str:
        """Extract a text preview from saved clipboard contents."""
        if not saved:
            return "(empty)"
        for ptype, data in saved:
            if "utf8" in ptype.lower() or "string" in ptype.lower() or "text" in ptype.lower():
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        # Non-text content
        return "(non-text)"

    @staticmethod
    def _model_allowed(model_id: str) -> bool:
        """Guard: some models require specific hardware or installed files."""
        if "large-v3-turbo" in model_id and _RAM_GB < _MIN_RAM_GB_FOR_V3_TURBO:
            return False
        if model_id == _PARAKEET_MODEL_ID:
            # Only show if the model files are present or the user has pointed us at them
            env_dir = os.environ.get("SPOKE_PARAKEET_MODEL_DIR", "")
            if env_dir and (Path(env_dir).expanduser() / "AudioEncoder.mlmodelc").exists():
                return True
            hf_snapshots = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--FluidInference--parakeet-ctc-110m-coreml"
                / "snapshots"
            )
            if hf_snapshots.is_dir():
                return any(
                    (s / "AudioEncoder.mlmodelc").exists() for s in hf_snapshots.iterdir()
                )
            return False
        return True

    def _quit(self) -> None:
        self._detector.uninstall()
        self._preview_active = False
        hf = getattr(self, "_handsfree", None)
        if hf is not None and hf.is_active:
            hf.disable()
        if hasattr(self, "_terraform_hud") and self._terraform_hud is not None:
            self._terraform_hud.cleanup()
        self._close_clients()
        NSApp.terminate_(None)

    def _show_accessibility_alert(self) -> None:
        """Show a dialog explaining the Accessibility permission requirement."""
        alert = NSAlert.new()
        alert.setMessageText_("Accessibility Permission Required")
        alert.setInformativeText_(
            "Spoke needs Accessibility access to detect spacebar holds.\n\n"
            "Go to System Settings → Privacy & Security → Accessibility "
            "and enable access for your terminal app (Terminal, iTerm2, etc.).\n\n"
            "Then relaunch Spoke."
        )
        alert.addButtonWithTitle_("OK")
        # Temporarily become a regular app so the alert is visible
        NSApp.setActivationPolicy_(1)  # NSApplicationActivationPolicyRegular
        alert.runModal()

    def _show_model_load_alert(self, error: Exception) -> None:
        alert = NSAlert.new()
        alert.setMessageText_("Model Load Failed")
        alert.setInformativeText_(
            "Spoke could not prepare the selected models.\n\n"
            f"{error}\n\n"
            "Choose a different model from the menu and Spoke will relaunch."
        )
        alert.addButtonWithTitle_("OK")
        NSApp.setActivationPolicy_(1)
        alert.runModal()


def _acquire_instance_lock() -> None:
    """Single-instance guard — kill any stuck old instance and take the lock."""
    import fcntl
    import signal as sig
    import time

    lock_path = _LOCK_PATH
    current_pid = os.getpid()
    parent_pid = os.getppid()
    _record_runtime_phase("instance_lock.start", lock_path=lock_path)
    logger.info(
        "Single-instance guard starting (pid=%d ppid=%d lock=%s)",
        current_pid,
        parent_pid,
        lock_path,
    )
    lock_file = open(lock_path, "a+")
    lock_file.seek(0)
    old_pid: int | None = None
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info(
            "Single-instance guard found an existing lock holder (pid=%d lock=%s)",
            current_pid,
            lock_path,
        )
        try:
            lock_file.seek(0)
            old_pid = int(lock_file.read().strip())
            logger.warning(
                "Single-instance guard sending SIGTERM to prior pid=%d from pid=%d",
                old_pid,
                current_pid,
            )
            os.kill(old_pid, sig.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        for _attempt in range(10):
            time.sleep(0.2)
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if _attempt == 4:
                    try:
                        os.kill(old_pid, sig.SIGKILL)
                        logger.info("Escalated to SIGKILL (pid=%d)", old_pid)
                    except (ProcessLookupError, PermissionError, NameError):
                        pass
                continue
        else:
            logger.warning("Another instance is running — exiting")
            _record_runtime_phase("instance_lock.exit_existing_instance", lock_path=lock_path)
            sys.exit(0)

    # Lock acquired — but a predecessor can release the flock before it has
    # fully exited. Treat unreaped zombies as already dead rather than as live
    # blocked apps, so the warning/SIGKILL path reflects real survivorship.
    if old_pid is not None and old_pid != current_pid and _is_process_alive(old_pid):
        logger.warning(
            "Predecessor pid=%d released lock but is still alive — sending SIGKILL",
            old_pid,
        )
        try:
            os.kill(old_pid, sig.SIGKILL)
            time.sleep(0.1)
        except (ProcessLookupError, PermissionError):
            pass

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(current_pid))
    lock_file.flush()
    logger.info(
        "Single-instance guard acquired lock (pid=%d lock=%s)",
        current_pid,
        lock_path,
    )
    _record_runtime_phase("instance_lock.acquired", lock_path=lock_path)
    # Keep lock_file alive for process lifetime
    _acquire_instance_lock._lock_file = lock_file


def main() -> None:
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_crash_diagnostics()
    _record_runtime_phase("process.start")

    zombie_sweep()
    _acquire_instance_lock()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    delegate = SpokeAppDelegate.alloc().init()
    app.setDelegate_(delegate)

    # Clean shutdown on SIGTERM — uninstall event tap and remove status item
    # before dying so we don't leave a zombie tap or ghost menu bar icon
    def _handle_sigterm(signum, frame):
        lock_pid = "unavailable"
        try:
            with open(_LOCK_PATH, encoding="utf-8") as lock_file:
                lock_pid = lock_file.read().strip() or "empty"
        except OSError:
            pass
        logger.info(
            "Received SIGTERM — cleaning up (pid=%d ppid=%d cwd=%s lock_pid=%s)",
            os.getpid(),
            os.getppid(),
            os.getcwd(),
            lock_pid,
        )
        _record_runtime_phase("signal.sigterm", lock_pid=lock_pid)
        delegate._detector.uninstall()
        if delegate._menubar is not None:
            delegate._menubar.cleanup()
        # Remove heartbeat so next launch doesn't see us as a zombie.
        if hasattr(delegate, "_heartbeat"):
            delegate._heartbeat.remove()
        NSApp.terminate_(None)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    from PyObjCTools import AppHelper

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
