# Fortnite-bot research workspace

This repository holds the **chest-hunter-ai** research project: a lightweight,
autonomous chest-hunting agent developed against a controlled **2D research
simulator**. It does not use the live Fortnite client.

| Phase | Status | Location |
|---|---|---|
| 1. Data foundation (SQLite chest locations, sessions, observations, CLI, tests) | Done | [`chest-hunter-ai/`](chest-hunter-ai/README.md) |
| 2. Route planning and navigation in a 2D simulator | Planned | |

Quick start:

```bash
cd chest-hunter-ai
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python -m chest_hunter init-db
python -m chest_hunter seed-demo
python -m unittest discover -s tests -v
```

See [`chest-hunter-ai/README.md`](chest-hunter-ai/README.md) for full documentation.
