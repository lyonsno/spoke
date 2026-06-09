#!/usr/bin/env python3
"""Run the visual witness selected by this smoke target.

Stable launcher shims may call this legacy command-overlay entry point from the
selected worktree even before the launcher itself knows newer target-specific
witness routes. Keep the route decision target-side so smoke env ownership
stays with the worktree under test.
"""

from __future__ import annotations

import os

from spoke.perceptasia_throughglass_witness import main as throughglass_main
from spoke.retina_lasso_witness import main as command_overlay_main


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    if _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE"):
        return throughglass_main()
    return command_overlay_main()


if __name__ == "__main__":
    raise SystemExit(main())
