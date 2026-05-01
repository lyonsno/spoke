#!/usr/bin/env python3
"""Run a small ACP backend probe and print structured JSON."""

from __future__ import annotations

from spoke.acp_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
