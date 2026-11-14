"""Bayesian chest-spawn estimates from the recorded observation history.

For every chest location ``i`` the probability that a chest is present at the start of an
episode is modelled as ``p_i ~ Beta(alpha_i, beta_i)``:

* prior: ``Beta(s * p0, s * (1 - p0))`` with ``p0`` the configured spawn probability and
  ``s`` the prior strength in pseudo-observations (``prior="configured"``), or
  ``Beta(1, 1)`` (``prior="uniform"``);
* every *valid* observation adds one to ``alpha`` (chest present) or ``beta`` (absent);
* the estimate is the posterior mean ``alpha / (alpha + beta)`` with a 95 % credible
  interval.

What counts as an observation:

* one per (session, location): within an episode presence does not change, so repeated
  records of the same spot are duplicates (a "present" anywhere in the session wins);
* only sessions that finished normally (``completed`` by default; aborted sessions may
  have stopped mid-way through recording);
* only sessions run on the same layout version (``layout_hash`` in the run parameters);
  sessions without a version (manual Phase 1 records, older runs) are included unless
  ``include_unversioned=False``;
* locations that were never observed keep their prior: unvisited is not "absent";
* rewards (``reward_received``) are a different quantity and are never used here.

Estimates are written to ``spawn_estimates``; ``chest_locations.spawn_probability`` (the
configured value) is never modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

from chest_hunter.analytics.stats import beta_interval
from chest_hunter.database import DatabaseManager
from chest_hunter.models import utc_now

PRIORS: Final = ("configured", "uniform")


@dataclass(frozen=True, slots=True)
class SpawnEstimate:
    location_id: int
    location_name: str
    configured_probability: float
    prior_alpha: float
    prior_beta: float
    present: int
    absent: int
    sessions: int

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.present

    @property
    def beta(self) -> float:
        return self.prior_beta + self.absent

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    @property
    def observed(self) -> bool:
        return self.present + self.absent > 0

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        return beta_interval(self.alpha, self.beta, level)


def information_gain(alpha: float, beta: float) -> tuple[float, float]:
    """(episode entropy in bits, expected posterior-variance reduction of one observation).

    The first measures how uncertain the presence of a chest is *in this episode*; the
    second how much one more observation teaches about the long-run probability."""
    p = alpha / (alpha + beta)
    h = 0.0 if p <= 0.0 or p >= 1.0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    n = alpha + beta
    var = alpha * beta / (n * n * (n + 1.0))
    after = p * ((alpha + 1) * beta / ((n + 1) ** 2 * (n + 2))) + \
        (1 - p) * (alpha * (beta + 1) / ((n + 1) ** 2 * (n + 2)))
    return h, max(0.0, var - after)


class SpawnEstimator:
    def __init__(self, db: DatabaseManager, *, prior: str = "configured",
                 prior_strength: float = 4.0, statuses: tuple[str, ...] = ("completed",),
                 include_unversioned: bool = True) -> None:
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {', '.join(PRIORS)}")
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        self.db = db
        self.prior = prior
        self.prior_strength = prior_strength
        self.statuses = statuses
        self.include_unversioned = include_unversioned

    def _prior(self, p0: float) -> tuple[float, float]:
        if self.prior == "uniform":
            return 1.0, 1.0
        p0 = min(0.99, max(0.01, p0))
        return self.prior_strength * p0, self.prior_strength * (1.0 - p0)

    def observation_counts(self, map_id: str, layout_version: str | None
                           ) -> dict[int, tuple[int, int, int]]:
        """location id -> (present, absent, sessions), one sample per (session, location)."""
        marks = ",".join("?" for _ in self.statuses)
        rows = self.db.query_all(
            f"""
            SELECT o.chest_location_id AS loc, o.session_id AS sess,
                   MAX(o.chest_present) AS present,
                   json_extract(r.parameters_json, '$.layout_hash') AS layout
            FROM chest_observations o
            JOIN simulation_sessions s ON s.id = o.session_id
            JOIN chest_locations c ON c.id = o.chest_location_id
            LEFT JOIN simulation_runs r ON r.session_id = s.id
            WHERE c.map_id = ? AND s.status IN ({marks})
            GROUP BY o.chest_location_id, o.session_id
            """, (map_id, *self.statuses))
        counts: dict[int, list[int]] = {}
        for row in rows:
            layout = row["layout"]
            if layout is None:
                if not self.include_unversioned:
                    continue
            elif layout_version is not None and layout != layout_version:
                continue
            c = counts.setdefault(int(row["loc"]), [0, 0, 0])
            c[0 if row["present"] else 1] += 1
            c[2] += 1
        return {k: (v[0], v[1], v[2]) for k, v in counts.items()}

    def estimate(self, map_id: str, layout_version: str | None = None
                 ) -> dict[int, SpawnEstimate]:
        locations = self.db.query_all(
            "SELECT id, location_name, spawn_probability FROM chest_locations "
            "WHERE map_id = ? AND enabled = 1 ORDER BY id", (map_id,))
        counts = self.observation_counts(map_id, layout_version)
        out = {}
        for row in locations:
            lid = int(row["id"])
            a0, b0 = self._prior(float(row["spawn_probability"]))
            present, absent, sessions = counts.get(lid, (0, 0, 0))
            out[lid] = SpawnEstimate(lid, str(row["location_name"]),
                                     float(row["spawn_probability"]), a0, b0, present, absent,
                                     sessions)
        return out

    def store(self, estimates: dict[int, SpawnEstimate], layout_version: str | None) -> int:
        """Upsert the estimates into ``spawn_estimates`` (configured values untouched)."""
        version = layout_version or ""
        now = utc_now().isoformat()
        with self.db.transaction() as conn:
            for e in estimates.values():
                lo, hi = e.credible_interval()
                conn.execute(
                    """
                    INSERT INTO spawn_estimates (chest_location_id, layout_version, prior_alpha,
                        prior_beta, present_count, absent_count, posterior_mean, credible_low,
                        credible_high, sessions_used, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (chest_location_id, layout_version) DO UPDATE SET
                        prior_alpha = excluded.prior_alpha, prior_beta = excluded.prior_beta,
                        present_count = excluded.present_count,
                        absent_count = excluded.absent_count,
                        posterior_mean = excluded.posterior_mean,
                        credible_low = excluded.credible_low,
                        credible_high = excluded.credible_high,
                        sessions_used = excluded.sessions_used, updated_at = excluded.updated_at
                    """, (e.location_id, version, e.prior_alpha, e.prior_beta, e.present,
                          e.absent, e.mean, lo, hi, e.sessions, now))
        return len(estimates)

    def report(self, map_id: str, layout_version: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for e in self.estimate(map_id, layout_version).values():
            lo, hi = e.credible_interval()
            rows.append({"location_id": e.location_id, "location": e.location_name,
                         "configured": round(e.configured_probability, 3),
                         "present": e.present, "absent": e.absent,
                         "posterior_mean": round(e.mean, 4), "credible_95": [round(lo, 4),
                                                                              round(hi, 4)]})
        return rows
