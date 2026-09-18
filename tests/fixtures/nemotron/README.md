# Observed Native Phase Fixture

`v010-offline-phase-costs.stderr` contains all eight admitted phase lines from
a real CPU/full-buffer invocation on September 18, 2026, not invented output.
Unrelated native diagnostics and per-graph detail are excluded by Spoke's
production privacy parser. No speech text or vocabulary is included.

- Native source: NVIDIA NeMo-Speech.cpp v0.1.0,
  `4f9676226f667d14608487df744f375db87127f8`, plus
  `native/nemotron/offline-phase-costs.patch`.
- Patch SHA-256: `a54ddf9207525f41e59771a2bcbc4058ef6ec9c00faed4204eaa86c632a1d168`.
- Loaded ASR dylib SHA-256: `0b98e5cd786b8f715d264096cf3294f52574cf0d626023f7fb9360add5727527`.
- Model SHA-256: `a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae`.
- Input WAV SHA-256: `5abbdd775e3bcabb63a1ef2dd63785610ae21e86e0c1b2a8d1c8e3e4857a8ee2`.
- Flags: `--json transcribe <wav> --model <gguf> --device cpu --language en
  --format json --no-batching --no-warmup`; `NEMO_SPEECH_TIMING=1`.
- Raw replay: `/Users/noahlyons/Library/Application Support/Spoke/incident-evidence/20260918-nemotron-warmup-repair/native-r1.stderr`;
  sibling `native-r1.json`, `native-r1-loader.stderr`, and
  `native-r1-doctor.json` preserve output, loaded library identity, and backend.

The unit test checks parser conformance to this observed version and stable
phase relationships. It does not establish current installed runtime identity,
latency under contention, or transcription quality. Re-run a retained WAV
through the selected runtime when that runtime or this patch changes.
