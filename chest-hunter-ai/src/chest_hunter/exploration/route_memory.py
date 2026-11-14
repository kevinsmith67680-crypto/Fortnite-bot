"""Lightweight memory of navigation attempts (no frames; a few numbers per route).

Each attempt stores the start and destination region (a coarse grid cell), the planned
length and time, the time it actually took, whether the agent got there, how many
recoveries it needed and, for chest targets, whether the chest was opened. From these:

* **travel-time factor** for a region pair: median of ``actual / estimated`` over recent
  successful attempts (falls back to the map-wide median, then to 1.0);
* **navigation reliability** of a destination: ``Beta(4.5 + successes, 0.5 + failures)``
  mean,
  with older sessions down-weighted so a single failure fades (temporary failures) while
  failures repeated over several sessions persist (persistent obstacles).

What it does not mix up:

* **map geometry changes**: every record carries the layout version; other versions are
  ignored, and :meth:`invalidate` marks records stale explicitly;
* **random chest-spawn changes**: an attempt that reached an empty spot is a navigation
  *success*; chest presence belongs to the spawn estimator, not here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Final

from chest_hunter.database import DatabaseManager
from chest_hunter.models import utc_now

REGION_SIZE: Final = 15.0
#: Weight of a record ``k`` sessions old: DECAY ** k.
DECAY: Final = 0.7
#: Prior pseudo-counts for navigation reliability (mean 0.9 without data).
PRIOR_SUCCESS: Final = 4.5
PRIOR_FAILURE: Final = 0.5


def region_of(x: float, y: float, z: float = 0.0) -> str:
    """Coarse region label of a position (15 m cells; upper floors are separate)."""
    level = int(round(z / 2.5))
    return f"r{int(x // REGION_SIZE)}_{int(y // REGION_SIZE)}" + (f"_L{level}" if level else "")


@dataclass(slots=True)
class RouteAttempt:
    start_region: str
    dest_region: str
    location_id: int | None
    estimated_length: float
    estimated_seconds: float
    started: float
    finished: float | None = None
    success: bool | None = None
    failure_kind: str | None = None
    recoveries: int = 0
    chest_opened: bool | None = None

    @property
    def actual_seconds(self) -> float:
        return 0.0 if self.finished is None else max(0.0, self.finished - self.started)


@dataclass(slots=True)
class _Stats:
    ratios: list[float] = field(default_factory=list)
    success_w: float = 0.0
    failure_w: float = 0.0


class RouteMemory:
    """Route statistics for one map and layout version (loaded once per episode)."""

    def __init__(self, db: DatabaseManager | None, map_id: str, layout_version: str | None,
                 *, max_sessions: int = 30) -> None:
        self.db = db
        self.map_id = map_id
        self.layout_version = layout_version or ""
        self.pending: list[RouteAttempt] = []
        self._pairs: dict[tuple[str, str], _Stats] = {}
        self._dest: dict[str, _Stats] = {}
        self._locations: dict[int, _Stats] = {}
        self._all_ratios: list[float] = []
        if db is not None:
            self._load(max_sessions)

    def _load(self, max_sessions: int) -> None:
        assert self.db is not None
        rows = self.db.query_all(
            """
            SELECT start_region, dest_region, chest_location_id, estimated_seconds,
                   actual_seconds, success, session_id
            FROM route_memory
            WHERE map_id = ? AND layout_version = ? AND invalidated = 0
            ORDER BY id
            """, (self.map_id, self.layout_version))
        sessions = sorted({r["session_id"] for r in rows if r["session_id"] is not None})
        recent = sessions[-max_sessions:]
        age = {s: len(recent) - 1 - i for i, s in enumerate(recent)}
        for r in rows:
            sid = r["session_id"]
            if sid is not None and sid not in age:
                continue
            self._add(str(r["start_region"]), str(r["dest_region"]), r["chest_location_id"],
                      float(r["estimated_seconds"]), float(r["actual_seconds"]),
                      bool(r["success"]), DECAY ** age.get(sid, 0))

    def _add(self, start: str, dest: str, location_id: int | None, est: float, actual: float,
             success: bool, weight: float) -> None:
        targets = [self._pairs.setdefault((start, dest), _Stats()),
                   self._dest.setdefault(dest, _Stats())]
        if location_id is not None:
            targets.append(self._locations.setdefault(int(location_id), _Stats()))
        for st in targets:
            if success:
                st.success_w += weight
            else:
                st.failure_w += weight
        if success and est > 1.0:
            ratio = min(5.0, max(0.5, actual / est))
            self._pairs[(start, dest)].ratios.append(ratio)
            self._all_ratios.append(ratio)

    # -- queries ----------------------------------------------------------------------------

    def travel_factor(self, start: str, dest: str) -> float:
        st = self._pairs.get((start, dest))
        if st is not None and len(st.ratios) >= 2:
            return float(statistics.median(st.ratios))
        if len(self._all_ratios) >= 3:
            return float(statistics.median(self._all_ratios))
        return 1.0

    def reliability(self, dest: str, location_id: int | None = None) -> float:
        """Probability of reaching the destination: posterior mean of
        ``Beta(PRIOR_SUCCESS + successes, PRIOR_FAILURE + failures)`` with decayed counts
        (0.9 when nothing is known yet)."""
        st = self._locations.get(location_id) if location_id is not None else None
        if st is None or st.success_w + st.failure_w < 0.5:
            st = self._dest.get(dest)
        s, f = (st.success_w, st.failure_w) if st is not None else (0.0, 0.0)
        return (PRIOR_SUCCESS + s) / (PRIOR_SUCCESS + PRIOR_FAILURE + s + f)

    def persistent_failure(self, dest: str) -> bool:
        st = self._dest.get(dest)
        return st is not None and st.failure_w >= 1.5 and st.failure_w > 2 * st.success_w

    # -- recording --------------------------------------------------------------------------

    def begin(self, start: tuple[float, float, float], dest: tuple[float, float, float],
              location_id: int | None, estimated_length: float, speed: float,
              now: float) -> RouteAttempt:
        attempt = RouteAttempt(region_of(*start), region_of(*dest), location_id,
                               estimated_length, estimated_length / max(0.1, speed), now)
        self.pending.append(attempt)
        return attempt

    def finish(self, attempt: RouteAttempt, now: float, *, success: bool,
               failure_kind: str | None = None) -> None:
        if attempt.finished is not None:
            return
        attempt.finished = now
        attempt.success = success
        attempt.failure_kind = failure_kind
        self._add(attempt.start_region, attempt.dest_region, attempt.location_id,
                  attempt.estimated_seconds, attempt.actual_seconds, success, 1.0)

    def flush(self, session_id: int | None) -> int:
        """Write finished attempts of this episode to the database."""
        done = [a for a in self.pending if a.finished is not None]
        if self.db is None or not done:
            self.pending = []
            return 0
        now = utc_now().isoformat()
        with self.db.transaction() as conn:
            for a in done:
                conn.execute(
                    """
                    INSERT INTO route_memory (map_id, layout_version, session_id, start_region,
                        dest_region, chest_location_id, estimated_length, estimated_seconds,
                        actual_seconds, success, failure_kind, recoveries, chest_opened,
                        created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (self.map_id, self.layout_version, session_id, a.start_region,
                          a.dest_region, a.location_id, round(a.estimated_length, 3),
                          round(a.estimated_seconds, 3), round(a.actual_seconds, 3),
                          int(bool(a.success)), a.failure_kind, a.recoveries,
                          None if a.chest_opened is None else int(a.chest_opened), now))
        self.pending = []
        return len(done)

    def invalidate(self, *, layout_version: str | None = None,
                   before_session: int | None = None) -> int:
        """Mark records stale (e.g. after a level edit or when obstacles were moved)."""
        if self.db is None:
            return 0
        sql = "UPDATE route_memory SET invalidated = 1 WHERE map_id = ?"
        args: list[object] = [self.map_id]
        if layout_version is not None:
            sql += " AND layout_version = ?"
            args.append(layout_version)
        if before_session is not None:
            sql += " AND session_id < ?"
            args.append(before_session)
        with self.db.transaction() as conn:
            return int(conn.execute(sql, args).rowcount)

    def summary(self) -> dict[str, float]:
        succ = sum(s.success_w for s in self._dest.values())
        fail = sum(s.failure_w for s in self._dest.values())
        return {"region_pairs": float(len(self._pairs)),
                "weighted_successes": round(succ, 2), "weighted_failures": round(fail, 2),
                "median_travel_factor": round(statistics.median(self._all_ratios), 3)
                if self._all_ratios else math.nan}
