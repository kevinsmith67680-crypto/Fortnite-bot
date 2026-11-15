"""Active visual exploration without prior chest coordinates (Phase 7).

The agent builds its own picture of an unfamiliar environment from what it is allowed to
use: its camera, its own odometry and the research environment's navigation service (path
queries and walkability raycasts). Three levels, each at its own rate:

* **Global exploration** (event-driven): :class:`ActiveExplorer` keeps a sparse
  :class:`CoverageMap` (2 m cells per floor level) of walkable space found by raycast fans
  and of what the camera has actually looked at, groups unseen walkable cells into frontier
  candidates and scores them::

      S_i = (lambda_g G_i + lambda_c C_i) / (T_i + eps) - lambda_r R_i

  ``G`` expected information gain (unseen walkable cells the target lets the camera cover,
  plus unknown space beyond open rays), ``C`` estimated chest-discovery value (learned
  discoveries per newly seen cell for the region type, plus unconfirmed detections nearby),
  ``T`` travel time (NavMesh path length for the best few, straight distance x detour for
  the rest, plus the scan), ``R`` risk (failed approaches and unreachable marks nearby).
  Candidates are recomputed only after enough new cells were seen or when a target is
  needed, at most ``max_candidates`` are scored, and a computation that runs past its
  budget returns the best candidate so far.
* **Local navigation** (every decision): the agent's path follower on NavMesh paths.
* **Visual inspection** (vision profile rate): :class:`CameraSectors` plans where to look.

Nothing here reads simulator state: walkability comes from the permitted raycast service,
everything else from the agent's own observations.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

STRATEGIES: Final = ("nearest_frontier", "max_information", "chest_prior", "cluster",
                     "risk_adjusted")
REGION_TYPES: Final = ("enclosed", "open", "elevated")

#: (lambda_g, lambda_c, lambda_r, cluster weight) per exploration strategy. The value
#: ``lambda_g G + lambda_c C`` is in "newly seen cells" units: ``chest_prior`` counts one
#: expected chest as 40 cells, ``cluster`` adds up to 8 cells per nearby discovery,
#: ``risk_adjusted`` subtracts 0.3 score units per failure nearby (a typical score is
#: 0.2-1). ``nearest_frontier`` ignores the value: the score is 1 / travel time.
STRATEGY_WEIGHTS: Final[dict[str, tuple[float, float, float, float]]] = {
    "nearest_frontier": (0.0, 0.0, 0.01, 0.0),
    "max_information": (1.0, 0.0, 0.01, 0.0),
    "chest_prior": (0.5, 40.0, 0.01, 0.0),
    "cluster": (0.5, 10.0, 0.01, 8.0),
    "risk_adjusted": (1.0, 10.0, 0.3, 0.0),
}


@dataclass(frozen=True, slots=True)
class ExplorationConfig:
    strategy: str = "max_information"
    cell_size_m: float = 2.0
    view_range_m: float = 11.0
    ray_count: int = 16
    ray_reach_m: float = 15.0              # the bridge's NAV_RAYCAST limit
    refan_distance_m: float = 2.5
    max_cells: int = 30_000
    max_candidates: int = 12
    path_queries: int = 4
    block_cells: int = 3                   # candidates group unseen cells in 3x3 blocks
    min_gain_cells: float = 3.0
    replan_new_cells: int = 12
    epsilon_s: float = 1.0
    detour_factor: float = 1.3
    scan_seconds: float = 4.0
    planning_budget_ms: float = 20.0
    sectors: int = 12
    sector_min_gain: float = 1.5
    sector_cooldown_s: float = 20.0
    arrive_radius_m: float = 2.5

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"exploration strategy must be one of {', '.join(STRATEGIES)}")
        if self.cell_size_m <= 0 or self.view_range_m <= 0 or self.ray_reach_m <= 0:
            raise ValueError("cell size, view range and ray reach must be positive")
        if self.ray_count < 4 or self.sectors < 4 or self.max_candidates < 1 \
                or self.max_cells < 100 or self.path_queries < 1:
            raise ValueError("ray_count, sectors >= 4; max_candidates, path_queries >= 1; "
                             "max_cells >= 100")


@dataclass(slots=True)
class Cell:
    walkable: bool = False
    blocked: bool = False
    seen: float = -math.inf          # simulated time the camera last covered it
    views: int = 0
    open_edge: bool = False          # a ray reached its full length here: space continues


Key = tuple[int, int, int]


class CoverageMap:
    """Sparse grid of what is walkable (raycasts) and what the camera has seen."""

    def __init__(self, cell_size: float = 2.0, max_cells: int = 30_000) -> None:
        self.cell = cell_size
        self.max_cells = max_cells
        self.cells: dict[Key, Cell] = {}
        self.seen_count = 0
        self.walkable_count = 0
        self.dropped = 0

    @staticmethod
    def level(z: float) -> int:
        return int(round(z / 0.25))

    def key(self, x: float, y: float, z: float) -> Key:
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)), self.level(z))

    def centre(self, key: Key) -> tuple[float, float, float]:
        return ((key[0] + 0.5) * self.cell, (key[1] + 0.5) * self.cell, key[2] * 0.25)

    def _get(self, key: Key) -> Cell | None:
        c = self.cells.get(key)
        if c is None:
            if len(self.cells) >= self.max_cells:
                self.dropped += 1
                return None
            c = self.cells[key] = Cell()
        return c

    def integrate_fan(self, origin: tuple[float, float, float], bearings: Sequence[float],
                      distances: Sequence[float], reach: float) -> int:
        """Mark cells along walkability rays (stairs are followed by the service, so a ray
        keeps the start level here: the level of cells beyond a height change is
        approximate). Returns the number of newly walkable cells."""
        new = 0
        step = self.cell * 0.5
        for bearing, dist in zip(bearings, distances, strict=True):
            rad = math.radians(bearing)
            dx, dy = math.cos(rad), math.sin(rad)
            t = 0.0
            while t <= dist:
                c = self._get(self.key(origin[0] + dx * t, origin[1] + dy * t, origin[2]))
                if c is not None and not c.walkable:
                    c.walkable = True
                    self.walkable_count += 1
                    new += 1
                t += step
            end = self.key(origin[0] + dx * dist, origin[1] + dy * dist, origin[2])
            if dist >= reach - 1e-6:
                c = self._get(end)
                if c is not None:
                    c.open_edge = True
            else:
                beyond = self._get(self.key(origin[0] + dx * (dist + step),
                                            origin[1] + dy * (dist + step), origin[2]))
                if beyond is not None and not beyond.walkable:
                    beyond.blocked = True
        return new

    def integrate_view(self, cam: tuple[float, float, float], yaw_deg: float, hfov_deg: float,
                       view_range: float, now: float, free_range: Callable[[float], float],
                       level_z: float) -> int:
        """Cells inside the camera's horizontal view wedge, closer than both the view range and
        the walkable distance in that direction, count as seen. Returns newly seen cells."""
        new = 0
        r = int(math.ceil(view_range / self.cell))
        ci, cj, lvl = self.key(cam[0], cam[1], level_z)
        half = hfov_deg / 2.0
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                x, y = (i + 0.5) * self.cell, (j + 0.5) * self.cell
                dx, dy = x - cam[0], y - cam[1]
                d = math.hypot(dx, dy)
                if d > view_range:
                    continue
                bearing = math.degrees(math.atan2(dy, dx)) % 360.0
                if d > self.cell and abs((bearing - yaw_deg + 180.0) % 360.0 - 180.0) > half:
                    continue
                if d > free_range(bearing) + self.cell:
                    continue                                   # behind a wall or obstacle
                c = self.cells.get((i, j, lvl))
                if c is None:
                    c = self._get((i, j, lvl))
                    if c is None:
                        continue
                if c.views == 0:
                    new += 1
                    self.seen_count += 1
                c.views += 1
                c.seen = now
        return new

    def coverage(self) -> float | None:
        walk = [c for c in self.cells.values() if c.walkable]
        if not walk:
            return None
        return sum(1 for c in walk if c.views > 0) / len(walk)

    def unseen_walkable(self) -> list[Key]:
        return [k for k, c in self.cells.items() if c.walkable and c.views == 0]

    def clear_seen(self) -> None:
        """Forget what was seen (after a localisation jump the old marks are misaligned)."""
        for c in self.cells.values():
            c.views = 0
            c.seen = -math.inf
        self.seen_count = 0


@dataclass(slots=True)
class Candidate:
    key: Key
    point: tuple[float, float, float]
    gain: float
    chest_value: float
    risk: float
    region: str
    travel_s: float = 0.0
    score: float = 0.0


@dataclass(slots=True)
class ExplorationStats:
    selections: int = 0
    candidate_sets: int = 0
    candidates_scored: int = 0
    path_queries: int = 0
    unreachable: int = 0
    reached: int = 0
    scans_planned: int = 0
    budget_overruns: int = 0
    replans_after_observation: int = 0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    gains_realised: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        return {"selections": self.selections, "candidate_sets": self.candidate_sets,
                "candidates_scored": self.candidates_scored, "path_queries": self.path_queries,
                "unreachable_targets": self.unreachable, "targets_reached": self.reached,
                "scans_planned": self.scans_planned, "budget_overruns": self.budget_overruns,
                "replans_after_observation": self.replans_after_observation,
                "latency_mean_ms": round(sum(lat) / len(lat), 4) if lat else None,
                "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 4) if lat else None}


class CameraSectors:
    """Where to look: angular sectors with last-observed time, visibility, detections and an
    information-gain heuristic. Scans sweep in one direction (no oscillation), skip recently
    covered sectors and stop when no sector promises new information."""

    def __init__(self, sectors: int = 12, *, min_gain: float = 1.5, cooldown_s: float = 20.0
                 ) -> None:
        self.n = sectors
        self.width = 360.0 / sectors
        self.min_gain = min_gain
        self.cooldown = cooldown_s
        self.last_observed = [-math.inf] * sectors
        self.visibility = [1.0] * sectors
        self.detections = [0] * sectors
        self.confidence = [0.0] * sectors
        self.needs_inspection = [False] * sectors

    def index(self, yaw_deg: float) -> int:
        return int((yaw_deg % 360.0) // self.width) % self.n

    def centre(self, k: int) -> float:
        return (k + 0.5) * self.width

    def observe(self, yaw_deg: float, hfov_deg: float, now: float) -> None:
        half = hfov_deg / 2.0
        for k in range(self.n):
            if abs((self.centre(k) - yaw_deg + 180.0) % 360.0 - 180.0) <= half:
                self.last_observed[k] = now

    def note_detection(self, bearing_deg: float, confidence: float, confirmed: bool) -> None:
        k = self.index(bearing_deg)
        self.detections[k] += 1
        self.confidence[k] = max(self.confidence[k], confidence)
        self.needs_inspection[k] = not confirmed

    def gains(self, unseen_by_sector: Sequence[float], now: float) -> list[float]:
        out = []
        for k in range(self.n):
            g = unseen_by_sector[k] * self.visibility[k]
            if self.needs_inspection[k]:
                g += 3.0
            if now - self.last_observed[k] < self.cooldown and not self.needs_inspection[k]:
                g *= 0.25                      # recently covered: little left to learn
            out.append(g)
        return out

    def plan(self, current_yaw: float, gains: Sequence[float], *, limit: int = 5
             ) -> list[float]:
        """Yaws to look at, in one sweep direction from the current yaw, best direction first;
        empty when nothing promises information (the scan stops)."""
        useful = [k for k in range(self.n) if gains[k] >= self.min_gain]
        if not useful:
            return []
        start = self.index(current_yaw)

        def sweep(direction: int) -> list[int]:
            order = [(start + direction * s) % self.n for s in range(self.n)]
            return [k for k in order if k in useful][:limit]

        cw, ccw = sweep(1), sweep(-1)

        def value(seq: list[int]) -> float:     # gain discounted by the rotation needed
            total, rot = 0.0, 0
            for k in seq:
                rot = min((k - start) % self.n, (start - k) % self.n)
                total += gains[k] / (1.0 + 0.15 * rot)
            return total

        best = cw if value(cw) >= value(ccw) else ccw
        return [self.centre(k) for k in best]


class ActiveExplorer:
    """Global exploration for one episode (episode memory; built fresh for every episode)."""

    def __init__(self, config: ExplorationConfig | None = None, *,
                 discovery_rates: dict[str, float] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.cfg = config or ExplorationConfig()
        self.map = CoverageMap(self.cfg.cell_size_m, self.cfg.max_cells)
        self.sectors = CameraSectors(self.cfg.sectors, min_gain=self.cfg.sector_min_gain,
                                     cooldown_s=self.cfg.sector_cooldown_s)
        #: chests found per newly seen cell, by region type (persistent research memory;
        #: defaults until learned).
        self.discovery_rates = dict(discovery_rates or {"enclosed": 0.02, "open": 0.01,
                                                        "elevated": 0.03})
        self.stats = ExplorationStats()
        self.clock = clock
        self._fan_origin: tuple[float, float, float] | None = None
        self._fan: tuple[list[float], list[float]] = ([], [])
        self._candidates: list[Candidate] | None = None
        self._seen_at_candidates = 0
        self.blacklist: set[Key] = set()
        self.searched: set[Key] = set()
        self.failures: deque[tuple[float, float, float]] = deque(maxlen=200)
        self.discovered: deque[tuple[float, float, float]] = deque(maxlen=200)
        self.target: Candidate | None = None
        self.done = False
        self.cells_at_target_start = 0
        self._ray_enclosure: dict[Key, float] = {}
        #: learning signals for the persistent memory (exploration.self_improving)
        self.cells_by_region: dict[str, float] = {r: 0.0 for r in REGION_TYPES}
        self.discoveries_by_region: dict[str, float] = {r: 0.0 for r in REGION_TYPES}
        self.discoveries_noted = 0
        self.travel_seconds = 0.0

    # -- observations ---------------------------------------------------------------------------

    def needs_fan(self, position: tuple[float, float, float]) -> bool:
        return self._fan_origin is None or math.dist(position[:2], self._fan_origin[:2]) \
            >= self.cfg.refan_distance_m or abs(position[2] - self._fan_origin[2]) > 0.5

    def bearings(self) -> list[float]:
        return [k * 360.0 / self.cfg.ray_count for k in range(self.cfg.ray_count)]

    def add_fan(self, origin: tuple[float, float, float], distances: Sequence[float]) -> int:
        bearings = self.bearings()
        new = self.map.integrate_fan(origin, bearings, distances, self.cfg.ray_reach_m)
        self._fan_origin = origin
        self._fan = (bearings, list(distances))
        key = self.map.key(*origin)
        self._ray_enclosure[key] = sum(min(d, 12.0) for d in distances) / (12.0 * len(distances))
        for k in range(self.sectors.n):                     # sector visibility from the fan
            centre = self.sectors.centre(k)
            nearest = min(range(len(bearings)),
                          key=lambda i: abs((bearings[i] - centre + 180.0) % 360.0 - 180.0))
            self.sectors.visibility[k] = min(1.0, distances[nearest] / self.cfg.view_range_m)
        return new

    def free_range(self, bearing: float) -> float:
        bearings, dists = self._fan
        if not bearings:
            return self.cfg.view_range_m
        nearest = min(range(len(bearings)),
                      key=lambda i: abs((bearings[i] - bearing + 180.0) % 360.0 - 180.0))
        return dists[nearest]

    def observe(self, camera: tuple[float, float, float], agent_z: float, yaw_deg: float,
                hfov_deg: float, now: float) -> int:
        new = self.map.integrate_view(camera, yaw_deg, hfov_deg, self.cfg.view_range_m, now,
                                      self.free_range, agent_z)
        self.sectors.observe(yaw_deg, hfov_deg, now)
        if new:
            region = self.region_type(self.map.key(camera[0], camera[1], agent_z))
            self.cells_by_region[region] += new
        return new

    def note_detection(self, position: tuple[float, float, float], agent: tuple[float, float,
                                                                                float],
                       confidence: float, confirmed: bool) -> None:
        bearing = math.degrees(math.atan2(position[1] - agent[1], position[0] - agent[0]))
        self.sectors.note_detection(bearing % 360.0, confidence, confirmed)

    def note_discovery(self, position: tuple[float, float, float]) -> None:
        self.discovered.append(position)
        self.discoveries_noted += 1
        self.discoveries_by_region[self.region_type(self.map.key(*position))] += 1.0

    def note_failure(self, position: tuple[float, float, float]) -> None:
        self.failures.append(position)

    # -- candidates -------------------------------------------------------------------------------

    def region_type(self, key: Key) -> str:
        if key[2] * 0.25 > 0.6:
            return "elevated"
        near = [v for k, v in self._ray_enclosure.items()
                if abs(k[0] - key[0]) <= 3 and abs(k[1] - key[1]) <= 3 and k[2] == key[2]]
        if near and sum(near) / len(near) < 0.45:
            return "enclosed"
        return "open"

    def _build_candidates(self, position: tuple[float, float, float],
                          tentative: Sequence[tuple[float, float, float]]) -> list[Candidate]:
        cfg = self.cfg
        b = cfg.block_cells
        blocks: dict[tuple[int, int, int], list[Key]] = {}
        for k in self.map.unseen_walkable():
            if k in self.blacklist:
                continue
            blocks.setdefault((k[0] // b, k[1] // b, k[2]), []).append(k)
        for k, c in self.map.cells.items():          # open edges: space continues beyond
            if c.open_edge and c.walkable and k not in self.blacklist:
                blocks.setdefault((k[0] // b, k[1] // b, k[2]), []).append(k)
        out = []
        for bk, keys in blocks.items():
            if bk in self.searched:
                continue
            gain = float(len(keys)) + 4.0 * sum(1 for k in keys if self.map.cells[k].open_edge)
            if gain < cfg.min_gain_cells:
                continue
            # the walkable cell of the block nearest to the block's centroid
            cx = sum(k[0] for k in keys) / len(keys)
            cy = sum(k[1] for k in keys) / len(keys)
            rep = min(keys, key=lambda k: (k[0] - cx) ** 2 + (k[1] - cy) ** 2)
            point = self.map.centre(rep)
            region = self.region_type(rep)
            chest = self.discovery_rates.get(region, 0.01) * len(keys)
            chest += sum(1.0 for t in tentative if math.dist(t[:2], point[:2]) <= 6.0)
            risk = float(sum(1 for f in self.failures if math.dist(f[:2], point[:2]) <= 4.0))
            out.append(Candidate(rep, point, gain, chest, risk, region))
        out.sort(key=lambda c: (-(c.gain + 10 * c.chest_value) / (1.0 + math.dist(
            position[:2], c.point[:2])), c.key))
        return out[: max(cfg.max_candidates * 3, cfg.max_candidates)]

    def score(self, c: Candidate, travel_s: float) -> float:
        lg, lc, lr, cluster = STRATEGY_WEIGHTS[self.cfg.strategy]
        value = lg * c.gain + lc * c.chest_value
        if self.cfg.strategy == "nearest_frontier":
            value = 1.0
        if cluster:
            value += cluster * sum(math.exp(-math.dist(d[:2], c.point[:2]) / 10.0)
                                   for d in self.discovered)
        return value / (travel_s + self.cfg.scan_seconds + self.cfg.epsilon_s) - lr * c.risk

    def select(self, position: tuple[float, float, float], speed: float,
               path_length: Callable[[tuple[float, float, float]], float | None],
               tentative: Sequence[tuple[float, float, float]] = ()) -> Candidate | None:
        """The next exploration target, or ``None`` when no reachable unexplored area is left."""
        started = self.clock()
        cfg = self.cfg
        if self._candidates is None or self.map.seen_count - self._seen_at_candidates \
                >= cfg.replan_new_cells:
            self._candidates = self._build_candidates(position, tentative)
            self._seen_at_candidates = self.map.seen_count
            self.stats.candidate_sets += 1
        pool = self._candidates
        prelim = []
        for c in pool[:cfg.max_candidates * 3]:
            straight = math.dist(position, c.point) * cfg.detour_factor / max(0.1, speed)
            prelim.append((self.score(c, straight), c))
        prelim.sort(key=lambda e: (-e[0], e[1].key))
        prelim = prelim[:cfg.max_candidates]
        self.stats.candidates_scored += len(prelim)
        best: Candidate | None = None
        queries = 0
        for _, c in prelim:
            if queries >= cfg.path_queries:
                break
            if 1000.0 * (self.clock() - started) > cfg.planning_budget_ms and best is not None:
                self.stats.budget_overruns += 1
                break
            queries += 1
            self.stats.path_queries += 1
            length = path_length(c.point)
            if length is None:
                self.blacklist.add(c.key)
                self.stats.unreachable += 1
                continue
            c.travel_s = length / max(0.1, speed)
            c.score = self.score(c, c.travel_s)
            if best is None or (c.score, -c.key[0], -c.key[1]) > (best.score, -best.key[0],
                                                                    -best.key[1]):
                best = c
        self.stats.latencies_ms.append(1000.0 * (self.clock() - started))
        self.stats.selections += 1
        if best is None:
            if not prelim:
                self.done = True
            self._candidates = None
            return None
        self.target = best
        self.cells_at_target_start = self.map.seen_count
        return best

    def reached(self, c: Candidate) -> None:
        b = self.cfg.block_cells
        self.searched.add((c.key[0] // b, c.key[1] // b, c.key[2]))
        self.stats.reached += 1
        self.stats.gains_realised.append(float(self.map.seen_count - self.cells_at_target_start))
        if len(self.stats.gains_realised) > 2000:
            del self.stats.gains_realised[:1000]
        self._candidates = None
        self.target = None

    def unreachable(self, c: Candidate) -> None:
        self.blacklist.add(c.key)
        self.stats.unreachable += 1
        self._candidates = None
        self.target = None

    def target_still_useful(self) -> bool:
        """False once the target's area has been seen on the way (replan)."""
        t = self.target
        if t is None:
            return False
        b = self.cfg.block_cells
        bk = (t.key[0] // b, t.key[1] // b, t.key[2])
        left = sum(1 for k, c in self.map.cells.items()
                   if (k[0] // b, k[1] // b, k[2]) == bk and c.walkable
                   and (c.views == 0 or c.open_edge))
        if left < self.cfg.min_gain_cells:
            self.stats.replans_after_observation += 1
            return False
        return True

    # -- camera ---------------------------------------------------------------------------------

    def unseen_by_sector(self, position: tuple[float, float, float]) -> list[float]:
        counts = [0.0] * self.sectors.n
        r = self.cfg.view_range_m
        lvl = self.map.level(position[2])
        for k, c in self.map.cells.items():
            if k[2] != lvl or not c.walkable or c.views > 0:
                continue
            x, y, _ = self.map.centre(k)
            d = math.hypot(x - position[0], y - position[1])
            if d > r or d < 0.5:
                continue
            bearing = math.degrees(math.atan2(y - position[1], x - position[0]))
            counts[self.sectors.index(bearing)] += 1.0
        return counts

    def scan_plan(self, position: tuple[float, float, float], cam_yaw: float, now: float
                  ) -> list[float]:
        gains = self.sectors.gains(self.unseen_by_sector(position), now)
        plan = self.sectors.plan(cam_yaw, gains)
        if plan:
            self.stats.scans_planned += 1
        return plan

    def summary(self) -> dict[str, Any]:
        cov = self.map.coverage()
        return {"strategy": self.cfg.strategy, "cells": len(self.map.cells),
                "walkable_cells": self.map.walkable_count, "seen_cells": self.map.seen_count,
                "coverage_of_known_walkable": round(cov, 4) if cov is not None else None,
                "dropped_cells": self.map.dropped, "done": self.done,
                **self.stats.to_dict()}


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Uncertainty-aware decisions on chest tracks (validated in the Phase 7 benchmark)."""

    approach_confidence: float = 0.55
    approach_hits: int = 2
    approach_sigma_m: float = 1.2
    inspect_confidence: float = 0.35
    inspect_hits: int = 2

    def classify(self, track: Any) -> str:
        """``approach`` | ``inspect`` | ``ignore`` for a chest track."""
        if track.category != "chest":
            return "ignore"
        if track.confidence >= self.approach_confidence and track.hits >= self.approach_hits \
                and track.sigma <= self.approach_sigma_m:
            return "approach"
        if track.confidence >= self.inspect_confidence or track.hits >= self.inspect_hits:
            return "inspect"
        return "ignore"
