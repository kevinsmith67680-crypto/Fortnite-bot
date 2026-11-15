"""Per-episode experience records and their summaries (Phase 6).

Besides presence observations (spawn estimates) and route records (travel times, route
reliability), every 3D episode of the visual agent leaves one ``episode_experience`` row:
exploration coverage, openings, interaction attempts and failures, routes started and
navigation failures, recoveries by class with their outcomes, and the supervisor summary.
Rows carry the experience partition key, so summaries never mix environment versions,
agent versions or simulation configurations unless asked to.
"""

from __future__ import annotations

import json
from typing import Any

from chest_hunter.database import DatabaseManager
from chest_hunter.models import utc_now


def record_episode(db: DatabaseManager, *, session_id: int, map_id: str, partition_key: str,
                   outcome: str, sim_seconds: float, agent: Any,
                   supervisor: dict[str, Any] | None) -> None:
    """Store the agent's own end-of-episode statistics (no ground truth)."""
    priors = list(agent.world.priors.values())
    resolved = sum(1 for p in priors if p.resolved)
    m = agent.metrics
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO episode_experience (session_id, map_id, partition_key,
                outcome, simulated_seconds, spawn_points, spawn_points_resolved, coverage,
                openings, interaction_attempts, failed_interactions, routes_started,
                navigation_failures, recoveries, recovery_json, supervisor_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, map_id, partition_key, outcome, round(max(0.0, sim_seconds), 3),
                  len(priors), resolved, round(resolved / len(priors), 4) if priors else 0.0,
                  m.chests_opened, m.interaction_attempts, m.failed_interactions,
                  m.routes_started, m.navigation_failures, m.recoveries,
                  json.dumps(m.recovery_log.to_dict(), sort_keys=True),
                  json.dumps(supervisor, sort_keys=True, default=str) if supervisor else None,
                  utc_now().isoformat()))


def summarise_experience(db: DatabaseManager, map_id: str, *,
                         partition_key: str | None = None) -> dict[str, Any]:
    """Aggregate experience of one map (one partition, or all when ``partition_key`` is
    None, reported per partition)."""
    rows = db.query_all(
        "SELECT * FROM episode_experience WHERE map_id = ? AND (? IS NULL OR partition_key = ?) "
        "ORDER BY session_id", (map_id, partition_key, partition_key))
    groups: dict[str, list[Any]] = {}
    for r in rows:
        groups.setdefault(str(r["partition_key"]), []).append(r)
    out: dict[str, Any] = {}
    for key, rs in groups.items():
        minutes = sum(float(r["simulated_seconds"]) for r in rs) / 60.0
        attempts = sum(int(r["interaction_attempts"]) for r in rs)
        routes = sum(int(r["routes_started"]) + int(r["navigation_failures"]) for r in rs)
        rec_att = rec_ok = 0
        for r in rs:
            for v in json.loads(r["recovery_json"]).values():
                rec_att += int(v["attempts"])
                rec_ok += int(v["resolved"])
        out[key] = {
            "episodes": len(rs),
            "outcomes": {o: sum(1 for r in rs if r["outcome"] == o)
                         for o in sorted({r["outcome"] for r in rs})},
            "mean_coverage": round(sum(float(r["coverage"]) for r in rs) / len(rs), 4),
            "openings_per_sim_minute": round(sum(int(r["openings"]) for r in rs) / minutes, 4)
            if minutes else None,
            "interaction_failure_rate": round(sum(int(r["failed_interactions"]) for r in rs)
                                              / attempts, 4) if attempts else None,
            "navigation_failure_rate": round(sum(int(r["navigation_failures"]) for r in rs)
                                             / routes, 4) if routes else None,
            "recovery_success_rate": round(rec_ok / rec_att, 4) if rec_att else None,
        }
    return out
