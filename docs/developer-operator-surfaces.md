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

## Optical Witness Debug Surfaces

Optical witness frame-strip manifests are developer-facing debug records, not
consumer request payloads. They may carry internal `transition.phase` metadata
and lifecycle snapshots for race correlation, but production requests must keep
`progress` out of the public contract.
