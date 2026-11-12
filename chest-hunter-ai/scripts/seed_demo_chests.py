#!/usr/bin/env python3
"""Seed the deterministic demo research-arena chest locations.

Equivalent to ``python -m chest_hunter seed-demo``; accepts the same options::

    python scripts/seed_demo_chests.py [--map MAP_ID] [--config PATH] [--db PATH]

The database must already exist (``python -m chest_hunter init-db``).
Seeding is idempotent: existing spawn positions are skipped, never duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chest_hunter.cli import main
except ModuleNotFoundError:  # running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from chest_hunter.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["seed-demo", *sys.argv[1:]]))
