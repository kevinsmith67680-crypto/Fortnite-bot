"""Paired benchmark of route-planning strategies.

For every episode index ``k`` each strategy runs on the *same* environment:
the same map layout and obstacles (``map_seed = seed``, or ``seed + k`` with
``vary_obstacles``), the same chest spawns (``episode = k``), the same start
position, speed and tick budget. Only the strategy differs, so per-episode
differences are attributable to the strategy (a paired design).

Throughput (chests opened per minute) uses **simulated** time from the fixed
timestep, so it does not depend on host speed. ``cpu_seconds`` is real
process CPU time per episode (including SQLite writes), reported separately.

By default the benchmark runs on an in-memory copy of the map's chest
locations, so the research database is not filled with benchmark sessions;
``persist_db`` writes the sessions to a real database instead.
"""

from __future__ import annotations

import csv
import itertools
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chest_hunter.agent.runner import SimulationRunner, load_world, make_oracle
from chest_hunter.analytics.metrics import EpisodeMetrics, StrategySummary, summarise
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.config import Settings
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ChestHunterError, ValidationError
from chest_hunter.models import STRATEGY_NAMES, utc_now
from chest_hunter.navigation.astar import DistanceOracle
from chest_hunter.simulation.environment import World

logger = logging.getLogger(__name__)

CSV_FIELDS: tuple[str, ...] = (
    "episode", "strategy", "seed", "outcome", "ticks", "simulated_seconds",
    "chests_per_minute", "chests_opened", "chests_found", "chests_spawned",
    "locations_visited", "candidate_locations", "failed_interactions", "recoveries",
    "unreachable_targets", "abandoned_targets", "travel_distance", "rewards",
    "target_selections", "astar_searches", "astar_cache_hits", "observations_recorded",
    "db_flushes", "cpu_seconds", "wall_seconds", "failure_reason",
)


@dataclass(frozen=True)
class PairedComparison:
    baseline: str
    challenger: str
    episodes: int
    mean_difference: float
    median_difference: float
    challenger_wins: int
    baseline_wins: int
    ties: int


@dataclass
class BenchmarkReport:
    map_id: str
    seed: int
    episodes: int
    strategies: tuple[str, ...]
    max_ticks: int
    vary_obstacles: bool
    rows: list[EpisodeMetrics] = field(default_factory=list)
    summaries: dict[str, StrategySummary] = field(default_factory=dict)
    comparisons: list[PairedComparison] = field(default_factory=list)
    setup_seconds: float = 0.0
    total_wall_seconds: float = 0.0
    csv_path: Path | None = None

    def format_table(self) -> str:
        headers = ("Strategy", "Chests/Min (mean)", "Chests/Min (median)", "Std",
                   "Travel Distance (mean)", "Opened", "Failed", "Recoveries",
                   "Unreachable", "CPU s/episode", "Ticks/CPU s")
        rows = []
        for name in self.strategies:
            s = self.summaries[name]
            rows.append((
                _display_name(name), f"{s.mean_chests_per_minute:.3f}",
                f"{s.median_chests_per_minute:.3f}", f"{s.stdev_chests_per_minute:.3f}",
                f"{s.mean_travel_distance:.1f}", str(s.total_opened),
                str(s.total_failed_interactions), str(s.total_recoveries),
                str(s.total_unreachable + s.total_abandoned),
                f"{s.mean_cpu_seconds:.3f}", f"{s.mean_ticks_per_cpu_second:,.0f}",
            ))
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        lines = ["  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                           for i, (h, w) in enumerate(zip(headers, widths, strict=True)))]
        lines.append("  ".join("-" * w for w in widths))
        for row in rows:
            lines.append("  ".join(v.ljust(w) if i == 0 else v.rjust(w)
                                   for i, (v, w) in enumerate(zip(row, widths, strict=True))))
        return "\n".join(lines)

    def format_comparisons(self) -> str:
        lines = []
        for c in self.comparisons:
            lines.append(
                f"{_display_name(c.challenger)} vs {_display_name(c.baseline)}: "
                f"mean difference {c.mean_difference:+.3f} chests/min, "
                f"median {c.median_difference:+.3f}; better in {c.challenger_wins}, "
                f"worse in {c.baseline_wins}, tied in {c.ties} of {c.episodes} paired episodes"
            )
        return "\n".join(lines)


def _display_name(strategy: str) -> str:
    return {"nearest": "Nearest", "weighted": "ProbabilityWeighted",
            "lookahead": "Lookahead"}.get(strategy, strategy)


def clone_map_chests(source: DatabaseManager, target: DatabaseManager, map_id: str) -> int:
    """Copy a map's chest locations, *preserving IDs* (spawn rolls are keyed by ID)."""
    locations = ChestRepository(source).get_chests_by_map(map_id)
    with target.transaction() as conn:
        for loc in locations:
            conn.execute(
                "INSERT INTO chest_locations (id, map_id, location_name, x, y, z, "
                "spawn_probability, estimated_open_seconds, enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (loc.id, loc.map_id, loc.location_name, loc.x, loc.y, loc.z,
                 loc.spawn_probability, loc.estimated_open_seconds, int(loc.enabled),
                 loc.created_at, loc.updated_at),
            )
    return len(locations)


def run_benchmark(
    source_db: DatabaseManager,
    settings: Settings,
    *,
    map_id: str,
    strategies: Sequence[str] = ("nearest", "weighted"),
    episodes: int = 100,
    seed: int = 42,
    max_ticks: int | None = None,
    vary_obstacles: bool = False,
    csv_path: Path | None = None,
    persist_db: DatabaseManager | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BenchmarkReport:
    """Run ``episodes`` paired episodes for every strategy and summarise them."""
    strategies = tuple(dict.fromkeys(s.lower() for s in strategies))
    unknown = [s for s in strategies if s not in STRATEGY_NAMES]
    if unknown or not strategies:
        raise ValidationError(
            f"strategies must be chosen from {', '.join(STRATEGY_NAMES)}; got {list(strategies)}.")
    if episodes < 1:
        raise ValidationError("episodes must be at least 1.")
    budget = settings.simulation.max_ticks if max_ticks is None else max_ticks
    if budget < 1:
        raise ValidationError("max_ticks must be at least 1.")

    started = time.perf_counter()
    if persist_db is not None:
        db = persist_db
        owns_db = False
    else:
        db = DatabaseManager(":memory:")
        db.initialize()
        owns_db = True
        if not clone_map_chests(source_db, db, map_id):
            db.close()
            raise ChestHunterError(f"Map '{map_id}' has no chest locations to benchmark.")

    report = BenchmarkReport(map_id=map_id, seed=seed, episodes=episodes,
                             strategies=strategies, max_ticks=budget,
                             vary_obstacles=vary_obstacles)
    clock_base = utc_now()
    try:
        world_cache: dict[int, tuple[World, DistanceOracle]] = {}

        def world_for(map_seed: int) -> tuple[World, DistanceOracle]:
            if map_seed not in world_cache:
                if vary_obstacles:
                    world_cache.clear()  # keep memory flat when every episode differs
                setup = time.perf_counter()
                world = load_world(db, settings, map_id=map_id, map_seed=map_seed)
                oracle = make_oracle(settings)
                oracle.warm(world.grid, sorted({c.cell for c in world.chests}))
                report.setup_seconds += time.perf_counter() - setup
                world_cache[map_seed] = (world, oracle)
            return world_cache[map_seed]

        total = episodes * len(strategies)
        done = 0
        for episode in range(episodes):
            world, oracle = world_for(seed + episode if vary_obstacles else seed)
            for strategy in strategies:
                runner = SimulationRunner(db, settings, map_id=map_id, strategy=strategy,
                                          seed=seed, episode=episode, max_ticks=budget,
                                          world=world, oracle=oracle, clock_base=clock_base)
                report.rows.append(runner.run().metrics)
                done += 1
                if progress is not None:
                    progress(done, total)
    finally:
        if owns_db:
            db.close()

    by_strategy = {s: [m for m in report.rows if m.strategy == s] for s in strategies}
    report.summaries = {s: summarise(s, rows) for s, rows in by_strategy.items()}
    for baseline, challenger in itertools.combinations(strategies, 2):
        report.comparisons.append(_compare(baseline, by_strategy[baseline],
                                           challenger, by_strategy[challenger]))
    report.total_wall_seconds = time.perf_counter() - started
    if csv_path is not None:
        write_csv(report, csv_path)
        report.csv_path = csv_path
    return report


def _compare(baseline: str, base_rows: list[EpisodeMetrics], challenger: str,
             challenger_rows: list[EpisodeMetrics]) -> PairedComparison:
    diffs = []
    wins = losses = ties = 0
    for b, c in zip(base_rows, challenger_rows, strict=True):
        assert b.episode == c.episode
        delta = (c.chests_per_minute or 0.0) - (b.chests_per_minute or 0.0)
        diffs.append(delta)
        if abs(delta) < 1e-9:
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
    return PairedComparison(baseline, challenger, len(diffs), statistics.fmean(diffs),
                            statistics.median(diffs), wins, losses, ties)


def write_csv(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for metrics in report.rows:
            writer.writerow(metrics.as_row())
