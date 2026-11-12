# Fortnite-bot research workspace

This repository holds the **chest-hunter-ai** research project: a lightweight
autonomous chest-hunting agent developed in a controlled **2D research
simulator**. It does not interact with the live Fortnite client, game
services, accounts or matchmaking.

| Phase | Status | Location |
|---|---|---|
| 1. Data foundation (SQLite chest locations, sessions, observations, CLI, tests) | Done | [`chest-hunter-ai/`](chest-hunter-ai/README.md) |
| 2. Autonomous navigation (2D simulator, A*, route planning, FSM agent, Tkinter GUI, benchmarks) | Done | [`chest-hunter-ai/`](chest-hunter-ai/README.md) |
| 3. Computer vision and visual navigation (simulated camera, OpenCV detection, tracking, localisation, occupancy mapping, frontier exploration, visual verification, oracle-vs-visual benchmark) | Done | [`chest-hunter-ai/`](chest-hunter-ai/README.md#14-phase-3-visual-perception-and-navigation) |

Quick start:

```bash
cd chest-hunter-ai
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python -m chest_hunter init-db
python -m chest_hunter seed-demo
python -m chest_hunter simulate --render        # GUI (needs Tkinter + display)
python -m chest_hunter simulate --headless      # no window

# Phase 3: the vision-based agent
python -m chest_hunter seed-demo --scenario visual_arena
python -m chest_hunter simulate --map visual_arena_01 --agent visual --seed 42 --render
python -m chest_hunter benchmark-vision --map visual_arena_01 --episodes 100 --seed 42

python -m unittest discover -s tests -v
```

Dependencies: Python 3.11+, plus NumPy and `opencv-python-headless` for the
Phase 3 visual agent (CPU only). The Phase 3 "camera" is the simulator's own
rendering; nothing captures the screen or touches any game.

See [`chest-hunter-ai/README.md`](chest-hunter-ai/README.md) for full documentation.
