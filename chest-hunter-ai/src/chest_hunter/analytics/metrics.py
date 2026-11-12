"""Per-episode performance metrics.

Primary metric: **chests opened per minute of simulated time**

    E = N_opened / T_minutes,  T = ticks / tick_rate / 60

Simulated time comes from the fixed timestep, so the metric is identical
however fast the host computer runs the simulation. Real execution cost is
reported separately as ``cpu_seconds`` (process CPU time) and
``wall_seconds``.

Distances are in grid cells.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EpisodeMetrics:
    strategy: str
    seed: int
    episode: int
    ticks: int = 0
    simulated_seconds: float = 0.0
    travel_distance: float = 0.0
    candidate_locations: int = 0
    chests_spawned: int = 0
    locations_visited: int = 0
    chests_found: int = 0
    chests_opened: int = 0
    failed_interactions: int = 0
    recoveries: int = 0
    unreachable_targets: int = 0
    abandoned_targets: int = 0
    rewards: int = 0
    target_selections: int = 0
    astar_searches: int = 0
    astar_cache_hits: int = 0
    observations_recorded: int = 0
    db_flushes: int = 0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    outcome: str | None = None
    failure_reason: str | None = None

    @property
    def chests_per_minute(self) -> float | None:
        """Chests opened per simulated minute, or ``None`` before any time passed."""
        if self.simulated_seconds <= 0.0:
            return None
        return self.chests_opened / (self.simulated_seconds / 60.0)

    @property
    def ticks_per_cpu_second(self) -> float | None:
        if self.cpu_seconds <= 0.0:
            return None
        return self.ticks / self.cpu_seconds

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        rate = self.chests_per_minute
        row["chests_per_minute"] = None if rate is None else round(rate, 6)
        return row


@dataclass(frozen=True)
class StrategySummary:
    strategy: str
    episodes: int
    mean_chests_per_minute: float
    median_chests_per_minute: float
    stdev_chests_per_minute: float
    mean_travel_distance: float
    total_travel_distance: float
    total_opened: int
    total_failed_interactions: int
    total_recoveries: int
    total_unreachable: int
    total_abandoned: int
    mean_simulated_seconds: float
    mean_cpu_seconds: float
    median_cpu_seconds: float
    mean_ticks_per_cpu_second: float


def summarise(strategy: str, episodes: Sequence[EpisodeMetrics]) -> StrategySummary:
    """Aggregate one strategy's episodes (rates of 0-duration episodes count as 0)."""
    if not episodes:
        raise ValueError("No episodes to summarise.")
    rates = [m.chests_per_minute or 0.0 for m in episodes]
    cpu = [m.cpu_seconds for m in episodes]
    tps = [m.ticks_per_cpu_second or 0.0 for m in episodes]
    return StrategySummary(
        strategy=strategy,
        episodes=len(episodes),
        mean_chests_per_minute=statistics.fmean(rates),
        median_chests_per_minute=statistics.median(rates),
        stdev_chests_per_minute=statistics.stdev(rates) if len(rates) > 1 else 0.0,
        mean_travel_distance=statistics.fmean(m.travel_distance for m in episodes),
        total_travel_distance=sum(m.travel_distance for m in episodes),
        total_opened=sum(m.chests_opened for m in episodes),
        total_failed_interactions=sum(m.failed_interactions for m in episodes),
        total_recoveries=sum(m.recoveries for m in episodes),
        total_unreachable=sum(m.unreachable_targets for m in episodes),
        total_abandoned=sum(m.abandoned_targets for m in episodes),
        mean_simulated_seconds=statistics.fmean(m.simulated_seconds for m in episodes),
        mean_cpu_seconds=statistics.fmean(cpu),
        median_cpu_seconds=statistics.median(cpu),
        mean_ticks_per_cpu_second=statistics.fmean(tps),
    )
