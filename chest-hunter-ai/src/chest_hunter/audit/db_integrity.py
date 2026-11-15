"""Phase 8 database and data-integrity audit (real SQLite files, real episodes).

Checks:

* sessions - one per episode, finished with ``ended_at``, none left ``running``;
* referential integrity - ``PRAGMA integrity_check`` and ``PRAGMA foreign_key_check``;
* no duplicated openings (one opening per session and spawn point);
* openings recorded in SQLite vs the simulator's ground truth per session (evaluation
  harness only): a recorded opening that did not happen, or a true opening that is missing;
* statistics vs raw records - the session statistics vs the observation rows;
* interrupted episodes - a session interrupted by an exception ends ``aborted``;
* migrations - a database created at schema version 1 with data is migrated to the current
  version and keeps its data;
* concurrent writers - two processes record episodes into one file at the same time;
* invalid values - out-of-range or malformed values are refused by the repositories and the
  schema constraints;
* active-exploration episodes - whether their openings reach the database at all.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sqlite3
from pathlib import Path
from typing import Any


def _one(db: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = db.query_one(sql, params)
    if row is None:
        raise RuntimeError(f"no row for {sql!r}")
    return row


def _episode(db: Any, settings: Any, *, seed: int, episode: int, max_ticks: int,
             map_id: str = "research_arena_3d", exploration: bool = False,
             interrupt_after: int | None = None) -> Any:
    from chest_hunter.agent.runner_3d import Runner3D
    from chest_hunter.exploration.active import ExplorationConfig

    def hook(runner: Any, result: Any) -> None:
        agent = getattr(runner.controller, "agent", runner.controller)
        if interrupt_after is not None and agent.metrics.decisions >= interrupt_after:
            raise KeyboardInterrupt("audit: simulated operator interrupt")

    runner = Runner3D.from_settings(
        db, settings, mode="hybrid", seed=seed, episode=episode, backend="reference",
        map_id=map_id, collect_truth=True, max_ticks=max_ticks, width=320, height=180,
        exploration=ExplorationConfig() if exploration else None, on_decision=hook)
    return runner, runner.run()


def _worker(path: str, settings_path: str, seed: int, episodes: int, max_ticks: int,
            queue: Any) -> None:
    from chest_hunter.config import load_config
    from chest_hunter.database import DatabaseManager

    db = DatabaseManager(path)
    errors = []
    try:
        for e in range(episodes):
            try:
                _episode(db, load_config(settings_path), seed=seed, episode=e,
                         max_ticks=max_ticks)
            except Exception as exc:                       # reported, not hidden
                errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        db.close()
    queue.put(errors)


def evaluate(settings: Any, workdir: Path, *, episodes: int = 6, max_ticks: int = 2400,
             seed: int = 31) -> dict[str, Any]:
    from chest_hunter.chest_repository import ChestRepository
    from chest_hunter.database import MIGRATIONS, SCHEMA_VERSION, DatabaseManager
    from chest_hunter.localization.world_model_3d import PriorStatus3D
    from chest_hunter.scenarios import seed_arena_3d
    from chest_hunter.session_repository import SessionRepository

    workdir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # -- recorded episodes vs ground truth ---------------------------------------------------
    path = workdir / "episodes.db"
    path.unlink(missing_ok=True)
    db = DatabaseManager(path)
    db.initialize()
    seed_arena_3d(db)
    mismatches = []
    agent_vs_db = []
    for e in range(episodes):
        runner, result = _episode(db, settings, seed=seed, episode=e, max_ticks=max_ticks)
        sid = result.session_id
        recorded = {int(r["chest_location_id"]) for r in db.query_all(
            "SELECT chest_location_id FROM chest_observations WHERE session_id = ? "
            "AND chest_opened = 1", (sid,))}
        truth = set(result.truth.get("location_ids_opened", []))
        agent = getattr(runner.controller, "agent", runner.controller)
        credited = {p.location_id for p in agent.world.priors.values()
                    if p.status is PriorStatus3D.OPENED}
        mismatches.append({"session": sid, "recorded": len(recorded), "truth": len(truth),
                           "recorded_not_true": sorted(recorded - truth),
                           "true_not_recorded": sorted(truth - recorded)})
        agent_vs_db.append(credited == recorded)
    sessions = db.query_all("SELECT id, status, ended_at FROM simulation_sessions")
    check("one session per episode", len(sessions) == episodes, len(sessions))
    check("every session finished with ended_at",
          all(s["status"] != "running" and s["ended_at"] for s in sessions),
          sorted({s["status"] for s in sessions}))
    check("integrity_check ok",
          _one(db, "PRAGMA integrity_check")[0] == "ok")
    check("no foreign-key violations", not db.query_all("PRAGMA foreign_key_check"))
    dup = db.query_all("SELECT session_id, chest_location_id, COUNT(*) AS n "
                       "FROM chest_observations WHERE chest_opened = 1 "
                       "GROUP BY session_id, chest_location_id HAVING n > 1")
    check("no duplicated openings", not dup, [dict(r) for r in dup])
    check("database openings equal the agent's credited openings", all(agent_vs_db))
    check("no recorded opening that did not happen",
          all(not m["recorded_not_true"] for m in mismatches),
          [m for m in mismatches if m["recorded_not_true"]])
    missing = sum(len(m["true_not_recorded"]) for m in mismatches)
    total_true = sum(m["truth"] for m in mismatches)
    check("every true opening recorded", missing == 0,
          {"missing": missing, "true_openings": total_true})
    stats_repo = SessionRepository(db)
    repo_mismatch = []
    for s in sessions:
        st = stats_repo.get_session_statistics(int(s["id"]))
        raw = _one(db, "SELECT COUNT(*) AS n FROM chest_observations WHERE session_id = ? "
                           "AND chest_opened = 1", (s["id"],))["n"]
        if int(st.chests_opened) != int(raw):
            repo_mismatch.append(int(s["id"]))
    check("session statistics equal raw observation rows", not repo_mismatch, repo_mismatch)

    # -- interrupted episode -------------------------------------------------------------------
    with contextlib.suppress(KeyboardInterrupt):
        _episode(db, settings, seed=seed, episode=99, max_ticks=max_ticks, interrupt_after=200)
    last = _one(db, "SELECT status, ended_at FROM simulation_sessions ORDER BY id DESC "
                        "LIMIT 1")
    check("an interrupted episode's session ends aborted", last["status"] == "aborted"
          and last["ended_at"] is not None, dict(last))

    # -- active exploration persistence ----------------------------------------------------------
    runner, result = _episode(db, settings, seed=seed, episode=7, max_ticks=max_ticks,
                              map_id="p7_open_test_004", exploration=True)
    rows = _one(db, "SELECT COUNT(*) AS n FROM chest_observations WHERE session_id = ?",
                        (result.session_id,))["n"]
    check("active-exploration openings reach the database",
          rows >= (result.truth.get("opened_truth") or 0) > 0 or
          (result.truth.get("opened_truth") or 0) == 0,
          {"observation_rows": rows, "true_openings": result.truth.get("opened_truth"),
           "credited": result.metrics.chests_opened})
    db.close()

    # -- invalid values ----------------------------------------------------------------------------
    inv = DatabaseManager(workdir / "invalid.db")
    (workdir / "invalid.db").unlink(missing_ok=True)
    inv = DatabaseManager(workdir / "invalid.db")
    inv.initialize()
    repo = ChestRepository(inv)
    rejected: dict[str, Any] = {}
    for label, kwargs in (("probability 1.5", {"spawn_probability": 1.5}),
                          ("NaN coordinate", {"x": float("nan")}),
                          ("empty map id", {"map_id": " "}),
                          ("negative open time", {"estimated_open_seconds": -1.0})):
        args: dict[str, Any] = {"map_id": "m", "location_name": "L", "x": 1.0,
                                "y": 2.0} | kwargs
        try:
            repo.add_chest(**args)
            rejected[label] = False
        except Exception as exc:                             # expected: refused
            rejected[label] = type(exc).__name__
    try:
        with inv.transaction() as conn:
            conn.execute("INSERT INTO chest_observations (session_id, chest_location_id, "
                         "observed_at, chest_present, chest_opened) VALUES (999, 999, "
                         "'2026-01-01T00:00:00+00:00', 0, 1)")
        rejected["observation without session/location, opened without present"] = False
    except Exception as exc:
        rejected["observation without session/location, opened without present"] = \
            type(exc).__name__
    inv.close()
    check("invalid values are refused", all(rejected.values()), rejected)

    # -- migration from schema version 1 ---------------------------------------------------------
    mpath = workdir / "migrate.db"
    mpath.unlink(missing_ok=True)
    con = sqlite3.connect(mpath)
    version, statements = MIGRATIONS[0]
    for sql in statements:
        con.execute(sql)
    con.execute(f"PRAGMA user_version = {version}")
    con.execute("INSERT INTO chest_locations (map_id, location_name, x, y, z, spawn_probability,"
                " estimated_open_seconds, enabled, created_at, updated_at) VALUES ('m', 'A', "
                "1.0, 2.0, 0.0, 0.5, 2.0, 1, '2026-01-01T00:00:00+00:00', "
                "'2026-01-01T00:00:00+00:00')")
    con.commit()
    con.close()
    mdb = DatabaseManager(mpath)
    mdb.initialize()
    kept = _one(mdb, "SELECT COUNT(*) AS n FROM chest_locations")["n"]
    check("migration from schema 1 keeps data", mdb.schema_version() == SCHEMA_VERSION
          and kept == 1, {"schema": mdb.schema_version(), "rows": kept})
    mdb.close()

    # -- concurrent writers ------------------------------------------------------------------------
    cpath = workdir / "concurrent.db"
    cpath.unlink(missing_ok=True)
    cdb = DatabaseManager(cpath)
    cdb.initialize()
    seed_arena_3d(cdb)
    cdb.close()
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(str(cpath), str(settings.config_path), 40 + k,
                                               2, 900, queue)) for k in range(2)]
    for p in procs:
        p.start()
    errors = [queue.get(timeout=900) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    cdb = DatabaseManager(cpath)
    n = _one(cdb, "SELECT COUNT(*) AS n FROM simulation_sessions")["n"]
    ok = _one(cdb, "PRAGMA integrity_check")[0] == "ok"
    cdb.close()
    check("two concurrent writer processes", ok and n == 4 and not any(errors),
          {"sessions": n, "integrity": ok, "errors": errors})
    return {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks),
            "episodes": episodes, "max_ticks": max_ticks, "per_session": mismatches,
            "environment": "Python reference arena, SQLite files on local disk"}
