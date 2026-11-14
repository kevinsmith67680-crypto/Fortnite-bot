"""Batch recording of demonstrations from the deterministic Phase 4 controller.

These are **synthetic** demonstrations (``source = synthetic-scripted-expert``): the
"demonstrator" is the scripted visual agent, not a person. They exercise the whole
learning pipeline and give the imitation policy something to imitate, but a model trained
on them says nothing about imitating human play. Human demonstrations are recorded with
the teleoperation window (``learning.teleop``, ``record-demo --controller human``).

Every episode also writes its sessions and chest observations to the experience database,
which the spawn estimator learns from.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chest_hunter.config import Settings
from chest_hunter.database import DatabaseManager
from chest_hunter.learning.demonstration_recorder import DemonstrationRecorder, next_episode_id


@dataclass(frozen=True, slots=True)
class DemoJob:
    episode_id: int
    map_id: str
    seed: int
    episode: int


def plan_jobs(output: str | Path, maps: Sequence[str], *, episodes_per_map: int,
              seed: int) -> list[DemoJob]:
    first = next_episode_id(output)
    jobs = []
    k = 0
    for map_id in maps:
        for e in range(episodes_per_map):
            jobs.append(DemoJob(first + k, map_id, seed, e))
            k += 1
    return jobs


def _seed_maps(db: DatabaseManager, maps: Sequence[str]) -> None:
    from chest_hunter.scenarios import ARENA_3D_ID, seed_arena_3d, seed_procedural_map
    from chest_hunter.simulation3d.procedural import is_procedural

    for map_id in sorted(set(maps)):
        if map_id == ARENA_3D_ID:
            seed_arena_3d(db)
        elif is_procedural(map_id):
            seed_procedural_map(db, map_id)
        else:
            raise ValueError(f"unknown 3D map {map_id!r}")


def _run_job(args: tuple[DemoJob, str, str, Settings, dict[str, Any]]) -> dict[str, Any]:
    job, output, db_path, settings, options = args
    from chest_hunter.agent.runner_3d import Runner3D

    started = time.perf_counter()
    with DatabaseManager(db_path, timeout=300.0) as db:
        recorder = DemonstrationRecorder(output)
        runner = Runner3D.from_settings(db, settings, mode="hybrid", seed=job.seed,
                                        episode=job.episode, backend="reference",
                                        map_id=job.map_id, demo_recorder=recorder,
                                        demo_episode_id=job.episode_id, **options)
        result = runner.run()
    return {"episode_id": job.episode_id, "map_id": job.map_id, "seed": job.seed,
            "episode": job.episode, "outcome": result.outcome,
            "chests_opened": result.chests_opened, "sim_seconds": round(result.sim_seconds, 2),
            "bytes": recorder.bytes_written,
            "wall_seconds": round(time.perf_counter() - started, 1)}


def record_scripted_demos(output: str | Path, db_path: str | Path, settings: Settings, *,
                          maps: Sequence[str], episodes_per_map: int, seed: int,
                          workers: int = 1, max_ticks: int | None = None,
                          width: int | None = None, height: int | None = None,
                          progress: Callable[[dict[str, Any]], None] | None = None
                          ) -> list[dict[str, Any]]:
    """Record ``episodes_per_map`` expert episodes on every map into ``output``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with DatabaseManager(db_path) as db:
        db.initialize()
        _seed_maps(db, maps)
    jobs = plan_jobs(output, maps, episodes_per_map=episodes_per_map, seed=seed)
    options = {k: v for k, v in (("max_ticks", max_ticks), ("width", width), ("height", height))
               if v is not None}
    args = [(job, str(output), str(db_path), settings, options) for job in jobs]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for a in args:
            r = _run_job(a)
            results.append(r)
            if progress:
                progress(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_run_job, args):
                results.append(r)
                if progress:
                    progress(r)
    return results
