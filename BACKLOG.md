# Spoke Backlog

## Distribution (ship blockers)

- [ ] **App icon** (.icns)
- [ ] **Code signing + notarization** — Apple Developer account ($99/yr). Required for clean install without "unidentified developer" dialog. Also stabilizes TCC permissions across rebuilds.
- [ ] **First-launch model download with progress UI** — Currently the model downloads silently on first transcription (~400MB), which looks like a hang. Need a dialog with progress bar.
- [ ] **README demo GIF/video** — Demo recorded, being sliced.

## Product (v1.1+)

- [ ] **Menubar source toggles** — Independent selection of preview and transcription backends. Previews: local | sidecar | off. Transcription: local | sidecar.
- [x] **Paste-failure recovery** — Shipped. Recovery overlay with retry/dismiss/clipboard columns, OCR verification, bounce animation on failure.
- [ ] **Toggle mode** — Press-to-start/press-to-stop as alternative to hold-to-record. Configurable hotkey via menubar dropdown. For accessibility and workflows where holding is impractical.
- [ ] **Bundle size optimization** — torch is ~245MB of dead weight (only needed by mlx_whisper's unused torch_whisper.py). Investigate dropping without breaking metallib resolution.

## Phase 5 — Silence-batched hybrid transcription

- [ ] **Silence detection** — RMS threshold + minimum duration on per-chunk amplitude. Infrastructure already in capture.py.
- [ ] **Utterance segmentation** — Split audio at silence boundaries into discrete segments.
- [ ] **Background sidecar batching** — Send completed segments to sidecar during recording, accumulate results.
- [ ] **Local preview of current utterance** — Real-time transcription of in-progress speech.
- [ ] **Result merging** — Stitch sidecar segment results + final segment into complete transcription on release.
- [ ] **Graceful degradation** — If sidecar unavailable, fall back to full-buffer local transcription.

## Known issues

- [ ] **Command overlay text direction** — Both user utterance and assistant response currently extrude upward out of the top of the overlay. Intended behavior: user utterance stays pinned at the top of the overlay, assistant response ascends underneath it. Long responses should scroll upward beneath the fixed utterance header.
- [ ] **Glow corner intensity** — Linear gradients produce slightly lower intensity at rounded corners where they overlap. Proper fix requires radial/path gradient or CGImage-based rendering.
- [ ] **TCC permission fragility** — Ad-hoc signing means every rebuild changes binary identity, invalidating Accessibility/Microphone grants. Proper fix is Developer ID certificate (covered above).
