# Unreal Engine 5 research project (prepared, not yet built)

`ChestHunterResearch/` is a small UE 5.4 C++ project: a third-person research arena with
chest actors, a spring-arm camera rendered through a scene capture, a Recast navigation
mesh, and a loopback-only TCP bridge that speaks the same protocol as the Python mock
server (`src/chest_hunter/unreal_bridge/`).

**Status:** the sources and configuration are complete as far as could be done without an
Unreal installation. They have **not** been compiled, the automation tests have **not** been
run, and the Python agent has **not** been connected to an Unreal instance. What *is*
checked automatically (`tests/test_unreal_sources.py`): protocol constants, commands, error
codes, sizes and parameters match the Python side, security properties visible in the
source, and the shared deterministic-randomness code compiled with g++ gives bit-identical
results to Python.

Uses only engine content (`/Engine/BasicShapes`); no third-party game assets. The arena
layout comes from `Content/ResearchData/research_arena_3d.json`, produced by
`python -m chest_hunter export-arena-3d`.

Build, run and protocol details: [`docs/phase4_3d_architecture.md`](../docs/phase4_3d_architecture.md).

## Phase 5 additions (also not compiled)

* `python -m chest_hunter generate-maps --split test --count 4 --export-unreal` writes
  procedural layouts (`proc_train_NNN`, `proc_test_NNN`) next to `research_arena_3d.json`.
  The arena builder loads any of them by map id, and `HELLO` lists every layout file found.
* Optional `appearance` fields in a layout are applied at reset: `lighting` scales the
  sun, and `chest_value` / `chest_hue_shift` tint the chest materials.
* Human demonstrations can be recorded against a running Unreal server with
  `record-demo --controller human --backend unreal` (untested). Exact replay of such a
  recording requires the engine to step deterministically, which has not been verified.
