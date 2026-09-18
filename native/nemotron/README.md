# Nemotron Offline Timing Patch

`offline-phase-costs.patch` adds measurements only to NVIDIA NeMo-Speech.cpp
v0.1.0, commit `4f9676226f667d14608487df744f375db87127f8`. It does not change
decoding, segmentation, vocabulary bias, endpointing, or model weights.

Spoke explicitly passes `--no-warmup` to the one-process-per-dictation CLI.
The CLI otherwise warms a streaming runner before the actual offline request.
Mixed cache/offline timing remains contradictory for this no-warmup invocation;
the parser does not silently reinterpret arbitrary mixed routes as valid.

The existing native `offline-transducer decode` line encloses inference and
decoding. Spoke classifies it as `offline_total`, never exclusive decoder time.
Additional patch lines are:

- `setup`: lazy offline-path setup, before feature extraction, per segment.
- `decoder-step`: sum of decoder `step()` calls across offline segments. It
  excludes encoding, decoder construction/configuration, and `finalize()`.
- `rnnt-work`: the RNNT greedy decoder's accumulated encoder frames, predictor
  calls, joint calls, speculative joint frames, and emitted tokens. Other
  decoder classes do not emit fabricated zero-valued RNNT evidence.

Legacy binaries remain usable. Their cost breakdown is explicitly unobserved;
partial new telemetry remains partial. Whole-request wall time remains the
outer cost, not a sum of overlapping native timers.

For a patched runtime, export the exact source revision and its pinned ggml
submodule `c03b4e2bcece5134827881af90242086daf75be5` and build CPU ASR according to
NVIDIA's `docs/build.md`. When the export is nested inside a Spoke worktree,
run `git apply --check --directory=<export>` and then
`git apply --directory=<export>` from the Spoke root. Running Git from inside
the nested export can silently skip paths; verify the resulting source and
actual emitted telemetry. The pinned release's SentencePiece build script names
source revision `17d7580d6407802f85855d2cc9190634e2c95624`.

Use a separate immutable runtime prefix and `SPOKE_NEMO_SPEECH_BINARY` to select
it. Preserve source, patch, dependency, build configuration and artifact hashes
in the rollout receipt. Verify the actual selected binary with `--json doctor`
and full-buffer audio before selecting it for the operator. Never replace a
runtime in place underneath an active Spoke process.

The local timing-only composition retains the official CLI and all official
runtime libraries except `libnemo_speech_asr.dylib`. Replace that library only
in the new prefix, relocate its build-directory RPATH to `@loader_path`, and
ad-hoc sign it. Verify the remaining artifacts match the official prefix and
the loader resolves ggml from the new prefix, not the build tree. This keeps
the original CPU kernels in the comparison.
