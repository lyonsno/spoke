# Developer And Operator Surfaces

This document holds real `spoke` capabilities that do not belong on the
public README but still need a durable canonical home.

## Bounded Post-Transcription Repair Pass

`spoke` keeps a bounded post-transcription repair pass for recurring
project-specific vocabulary observed in real logs.

This is a developer-facing correction surface, not a public product promise.
The implementation currently lives in [`spoke/dedup.py`](../spoke/dedup.py),
and README omission is intentional unless the repair pass becomes a visible
user-facing control or configuration surface.

## Transcription Vocabulary Prompt

Prompt-capable transcription backends resolve live context for every utterance
from bounded built-in vocabulary, the caller-owned
`SPOKE_TRANSCRIPTION_PROMPT_PATH` file (default
`~/.config/spoke/transcription_prompt.txt`), and optional inline
`SPOKE_TRANSCRIPTION_PROMPT` text. Prompt content is not capped or logged.
Route diagnostics record sources, character count, SHA-256 identity, backend
support, and whether the backend actually applied the requested prompt.

`SPOKE_TRANSCRIPTION_PROMPT_BUILTIN=0` disables the built-in vocabulary while
preserving file and inline sources. MLX Whisper applies supported context via
`initial_prompt`; OpenAI-compatible transcription routes send multipart
`prompt`. An unsupported runtime must omit the prompt and report it as
requested but ineffective.

## Interactive GPU Lease

Spoke can request a GPU Greenroom interactive lease for each real recording:

```sh
export SPOKE_GPU_INTERACTIVE_LEASE=1
export SPOKE_GPU_GREENROOM_BINARY=/path/to/gpu-greenroom
export SPOKE_GPU_GREENROOM_ROUTE_ID=gpu-greenroom@<commit>
```

The request launches asynchronously after audio capture starts, so a blocked
Greenroom lock cannot delay recording. The per-utterance holder remains alive
through final ASR and releases by stdin EOF on success, failure, empty audio,
or shutdown. Spoke reports `lease-effective` only after an effective Greenroom
event from a live holder process; requested, failed, released, or stale holders
remain `scheduler-unverified`.

Spoke writes caller-owned reports under
`~/Library/Application Support/Spoke/gpu-lease-receipts` by default. Override
that path with `SPOKE_GPU_LEASE_RECEIPT_DIR` and the queue with
`SPOKE_GPU_GREENROOM_DIR`. The lease excludes new jobs using Greenroom's shared
`gpu.lock`; it does not preempt a running job or govern nonparticipating Metal,
browser, rendering, or image-generation processes.

## Optical Witness Debug Surfaces

Optical witness frame-strip manifests are developer-facing debug records, not
consumer request payloads. They may carry internal `transition.phase` metadata
and lifecycle snapshots for race correlation, but production requests must keep
`progress` out of the public contract.
