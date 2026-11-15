"""Phase 8 navigation audit: the real agent (planning, path following, recovery, supervisor)
sent to one goal per episode on the reference arena.

The agent runs unchanged except for *what* it is asked to reach: a :class:`GoalExplorer`
hands it a single goal point as its exploration target (then reports exploration done), and
confidence thresholds that ignore every chest stop it from diverting to chests on the way.
Everything else - NavMesh path queries, waypoint following, look-while-travelling, stuck
detection, recovery ladders, the supervisor - is the production code path.

Goals are spawn points of held-out procedural maps and the research arena, classified by
terrain: ``open_terrain`` (straight line nearly free), ``around_obstacles`` (path at least
15 % longer than the straight line), ``indoor_room`` (inside a building/room: doorways),
``elevated`` (stairs, ramps, upper floors, plateaus: goal 0.9 m or more above the start
floor), ``unreachable`` (no path exists; the agent must give up, quickly and boundedly).

Per goal: reached (within the 2.5 m arrival radius), time to reach (simulated s), detour
ratio (true distance travelled / the navigation service's shortest path to the arrival
point), recoveries, stuck events, routes planned, supervisor interventions; for
unreachable goals: detected, time to detection, and the number of path queries spent.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from chest_hunter.exploration.active import ActiveExplorer, Candidate, ExplorationConfig

TERRAINS = ("open_terrain", "around_obstacles", "indoor_room", "elevated", "unreachable")
DEFAULT_MAPS = ("research_arena_3d", "p7_open_test_002", "p7_indoor_test_002",
                "p7_mixed_test_002", "p7_complex_test_002", "p7_complex_test_003")


class GoalExplorer(ActiveExplorer):
    """An explorer with exactly one target: ``goal``."""

    def __init__(self, goal: tuple[float, float, float]) -> None:
        super().__init__(ExplorationConfig())
        self.goal = goal
        self.issued = False
        self.reached_at: float | None = None
        self.unreachable_at: float | None = None
        self.queries = 0
        self.now: Callable[[], float] = lambda: 0.0
        self.travel: Callable[[], float] = lambda: 0.0
        self.travel_at_reach: float | None = None
        self.where: Callable[[], tuple[float, float, float]] = lambda: (0.0, 0.0, 0.0)
        self.position_at_reach: tuple[float, float, float] | None = None

    def select(self, position: tuple[float, float, float], speed: float,
               path_length: Callable[[tuple[float, float, float]], float | None],
               tentative: Sequence[tuple[float, float, float]] = ()) -> Candidate | None:
        if self.issued:
            self.done = True
            return None
        self.queries += 1
        length = path_length(self.goal)
        if length is None:
            self.stats.unreachable += 1
            self.unreachable_at = self.now()
            self.issued = True
            self.done = True
            return None
        self.issued = True
        c = Candidate(self.map.key(*self.goal), self.goal, 99.0, 0.0, 0.0, "goal",
                      travel_s=length / max(0.1, speed))
        self.target = c
        return c

    def reached(self, c: Candidate) -> None:
        if self.reached_at is None:
            self.reached_at = self.now()
            self.travel_at_reach = self.travel()
            self.position_at_reach = self.where()
        super().reached(c)

    def unreachable(self, c: Candidate) -> None:
        if self.unreachable_at is None:
            self.unreachable_at = self.now()
        super().unreachable(c)

    def scan_plan(self, position: tuple[float, float, float], cam_yaw: float, now: float
                  ) -> list[float]:
        return []

    def target_still_useful(self) -> bool:
        return self.target is not None          # the goal stays the goal until reached


@dataclass(frozen=True, slots=True)
class NavTask:
    map_id: str
    goal: tuple[float, float, float]
    terrain: str
    shortest_m: float | None
    seed: int
    settings_path: str
    max_ticks: int = 2700


def classify_goals(map_id: str, rng: random.Random, per_map: int = 8) -> list[NavTask]:
    from chest_hunter.audit.vision_eval import _indoor
    from chest_hunter.simulation3d.environment3d import cached_arena

    arena, _, nav = cached_arena(map_id)
    start = arena.start
    tasks: dict[str, list[NavTask]] = {t: [] for t in TERRAINS}
    for sp in arena.spawns:
        goal = (sp.x, sp.y, sp.z)
        path = nav.find_path(start, goal, goal_tolerance=2.0)
        if not path.found:
            terrain = "unreachable"
            length = None
        else:
            length = float(path.length)
            straight = math.dist(start[:2], goal[:2])
            if sp.z - start[2] >= 0.9:
                terrain = "elevated"
            elif _indoor(arena, sp.x, sp.y):
                terrain = "indoor_room"
            elif length > 1.15 * straight + 1.0:
                terrain = "around_obstacles"
            else:
                terrain = "open_terrain"
        tasks[terrain].append(NavTask(map_id, goal, terrain, length, 0, ""))
    out: list[NavTask] = []
    for t in TERRAINS:
        rng.shuffle(tasks[t])
        out += tasks[t][: max(1, per_map // len(TERRAINS) + (1 if t != "unreachable" else 0))]
    return out


def run_nav_task(task: NavTask) -> dict[str, Any]:
    from chest_hunter.agent.controller_3d import Agent3DConfig
    from chest_hunter.agent.runner_3d import Runner3D
    from chest_hunter.config import load_config
    from chest_hunter.database import DatabaseManager
    from chest_hunter.exploration.active import ConfidenceThresholds
    from chest_hunter.runtime.supervisor import SupervisorConfig

    settings = load_config(task.settings_path)
    explorer = GoalExplorer(task.goal)
    ignore_chests = ConfidenceThresholds(approach_confidence=2.0, approach_hits=10**6,
                                         inspect_confidence=2.0, inspect_hits=10**6)
    db = DatabaseManager(":memory:")
    db.initialize()

    def hook(runner: Any, result: Any) -> None:
        agent = getattr(runner.controller, "agent", runner.controller)
        explorer.now = lambda: float(agent.now)
        env = runner.server.session.env           # evaluation side: true position and distance
        explorer.travel = lambda: float(env.character.distance_travelled)
        explorer.where = lambda: (env.character.x, env.character.y, env.character.z)

    try:
        runner = Runner3D.from_settings(
            db, settings, mode="hybrid", seed=task.seed, map_id=task.map_id,
            backend="reference", max_ticks=task.max_ticks, width=320, height=180,
            config=Agent3DConfig(smart_recovery=True), exploration=lambda: explorer,
            thresholds=ignore_chests, supervisor=SupervisorConfig(), on_decision=hook)
        result = runner.run()
    finally:
        db.close()
    m = result.metrics
    sup = result.supervisor or {}
    travelled = explorer.travel_at_reach
    shortest = task.shortest_m
    if explorer.position_at_reach is not None:
        from chest_hunter.simulation3d.environment3d import cached_arena

        arena, _, nav = cached_arena(task.map_id)
        path = nav.find_path(arena.start, explorer.position_at_reach, goal_tolerance=0.5)
        shortest = float(path.length) if path.found else shortest
    return {
        "map_id": task.map_id, "terrain": task.terrain,
        "goal": [round(v, 2) for v in task.goal],
        "shortest_path_m": round(task.shortest_m, 2) if task.shortest_m else None,
        "reached": explorer.reached_at is not None,
        "time_to_reach_s": round(explorer.reached_at, 2) if explorer.reached_at else None,
        "travelled_to_reach_m": round(travelled, 2) if travelled is not None else None,
        "shortest_to_arrival_m": round(shortest, 2) if shortest else None,
        # travelled / the navigation service's own shortest path to the arrival point (that
        # path is grid A* + string pulling, a few per cent longer than the true geodesic, so
        # values slightly below 1 are possible)
        "detour_ratio": round(travelled / shortest, 4) if travelled and shortest else None,
        "unreachable_detected": explorer.unreachable_at is not None,
        "time_to_unreachable_s": round(explorer.unreachable_at, 2)
        if explorer.unreachable_at is not None else None,
        "goal_path_queries": explorer.queries,
        "recoveries": m.recoveries, "stuck_events": m.recovery_reasons.get("stuck", 0),
        "recovery_reasons": dict(m.recovery_reasons),
        "routes_planned": m.routes_started, "navigation_failures": m.navigation_failures,
        "supervisor_interventions": int(sup.get("interventions_total", 0)),
        "outcome": result.outcome, "sim_seconds": round(result.sim_seconds, 2),
    }


def evaluate(settings: Any, *, maps: Sequence[str] = DEFAULT_MAPS, per_map: int = 8,
             seed: int = 11, workers: int = 1) -> dict[str, Any]:
    rng = random.Random(seed)
    tasks = [NavTask(t.map_id, t.goal, t.terrain, t.shortest_m, seed,
                     str(settings.config_path)) for m in maps
             for t in classify_goals(m, rng, per_map)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_nav_task, tasks))
    else:
        rows = [run_nav_task(t) for t in tasks]
    per: dict[str, Any] = {}
    for terrain in TERRAINS:
        rs = [r for r in rows if r["terrain"] == terrain]
        if not rs:
            continue
        if terrain == "unreachable":
            per[terrain] = {
                "tasks": len(rs),
                "detected_rate": round(sum(r["unreachable_detected"] for r in rs) / len(rs), 4),
                "time_to_detection_s": _stats([r["time_to_unreachable_s"] for r in rs]),
                "goal_path_queries_max": max(r["goal_path_queries"] for r in rs),
                "false_reached": sum(r["reached"] for r in rs)}
            continue
        ok = [r for r in rs if r["reached"]]
        per[terrain] = {
            "tasks": len(rs), "success_rate": round(len(ok) / len(rs), 4),
            "time_to_reach_s": _stats([r["time_to_reach_s"] for r in ok]),
            "detour_ratio": _stats([r["detour_ratio"] for r in ok]),
            "recoveries_per_task": _stats([r["recoveries"] for r in rs]),
            "stuck_events": sum(r["stuck_events"] for r in rs),
            "routes_per_task": _stats([r["routes_planned"] for r in rs]),
            "supervisor_interventions": sum(r["supervisor_interventions"] for r in rs),
            "failures": [{k: r[k] for k in ("map_id", "goal", "outcome", "recovery_reasons")}
                         for r in rs if not r["reached"]]}
    reachable = [r for r in rows if r["terrain"] != "unreachable"]
    return {"tasks": len(rows), "per_terrain": per,
            "overall_success_rate": round(sum(r["reached"] for r in reachable)
                                          / len(reachable), 4) if reachable else None,
            "maps": list(maps), "seed": seed,
            "environment": "Python reference arena (NOT Unreal)", "rows": rows}


def _stats(values: Sequence[float | None]) -> dict[str, float | None]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 3),
            "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}
