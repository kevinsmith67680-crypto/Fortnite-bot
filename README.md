# Fortnite-bot research workspace

This repository holds the **chest-hunter-ai** research project: a lightweight
autonomous chest-hunting agent developed in a controlled **2D research
simulator**. It does not interact with the live Fortnite client, game
services, accounts or matchmaking.

| Phase | Status | Location |
|---|---|---|
| 1. Data foundation (SQLite chest locations, sessions, observations, CLI, tests) | Done | [`chest-hunter-ai/`](chest-hunter-ai/README.md) |
| 2. Autonomous navigation (2D simulator, A*, route planning, FSM agent, Tkinter GUI, benchmarks) | Done | [`chest-hunter-ai/`](chest-hunter-ai/README.md) |

Quick start:

```bash
cd chest-hunter-ai
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python -m chest_hunter init-db
python -m chest_hunter seed-demo
python -m chest_hunter simulate --render        # GUI (needs Tkinter + display)
python -m chest_hunter simulate --headless      # no window
python -m unittest discover -s tests -v
```

See [`chest-hunter-ai/README.md`](chest-hunter-ai/README.md) for full documentation.
