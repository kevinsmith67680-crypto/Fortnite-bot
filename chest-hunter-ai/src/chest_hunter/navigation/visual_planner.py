"""Planning on the *estimated* world: chest targets, frontier exploration, routes.

* Routes use the Phase 2 A* on the estimated occupancy grid. Known walls
  block; unknown cells are traversable with an extra cost
  (``unknown_cell_cost``), so the agent prefers explored space. Unknown cells
  are revealed long before the agent reaches them, and the route is replanned
  when a wall is discovered on it.
* **Frontiers** are FREE cells 4-adjacent to UNKNOWN cells, grouped with
  connected components. Score: ``S = G / (T + eps)``, where ``G`` is the
  number of unknown cells inside a camera-sized window around the frontier
  plus ``prior_weight`` times the summed probability of unresolved spawn
  priors in that window. ``T`` is an inexpensive travel-time estimate
  (straight-line distance x 1.3 / speed, plus acceleration overhead). Only
  the best few frontiers are checked with A*. Unreachable ones are
  blacklisted for a while.
* **Inspection goals** are unresolved spawn priors whose surroundings are
  already known. The agent drives towards them until the location has been
  seen clearly.
* **Chest targets** are confirmed, unopened chest tracks, scored like the
  Phase 2 weighted strategy with visual confidence in place of the spawn
  probability: ``S = confidence / (path_time + open_time)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.localization.world_state import SpawnPrior, SpawnStatus, WorldModel
from chest_hunter.navigation.astar import PathFinder, PathResult
from chest_hunter.simulation.collision import Cell
from chest_hunter.vision.obstacle_detector import FREE, UNKNOWN
from chest_hunter.vision.visual_tracker import ObjectTrack

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
EPSILON = 0.5            # seconds; prevents division by zero
DETOUR_FACTOR = 1.3
MAX_ASTAR_CHECKS = 6
BLACKLIST_SECONDS = 30.0


@dataclass(frozen=True)
class Goal:
    kind: str                     # "chest", "frontier" or "inspect"
    cell: Cell
    position: tuple[float, float]
    score: float
    track_id: int | None = None
    location_id: int | None = None
    gain: float = 0.0


class VisualPlanner:
    def __init__(self, world: WorldModel, *, speed: float, acceleration: float,
                 view_width: float, view_height: float, open_seconds: float,
                 prior_weight: float = 300.0, use_priors: bool = True,
                 unknown_cost: float = 0.5, allow_diagonal: bool = True,
                 prevent_corner_cutting: bool = True) -> None:
        self.world = world
        self.speed = speed
        self.overhead = speed / acceleration
        self.half_w = max(1, int(view_width / 2))
        self.half_h = max(1, int(view_height / 2))
        self.open_seconds = open_seconds
        self.prior_weight = prior_weight if use_priors else 0.0
        self.pathfinder = PathFinder(allow_diagonal=allow_diagonal,
                                     prevent_corner_cutting=prevent_corner_cutting,
                                     penalty_cells=world.unknown_cells, penalty=unknown_cost)
        self._blacklist: dict[Cell, float] = {}
        self.frontier_evaluations = 0

    # -- routes ----------------------------------------------------------------------

    def plan(self, start: Cell, goal: Cell) -> PathResult:
        return self.pathfinder.find_path(self.world.planning_grid, start, goal)

    def start_cell(self, agent: tuple[float, float]) -> Cell:
        """The cell to plan from. If the estimated position falls in a blocked cell
        (the body is never inside a wall, so the estimate is off by a fraction of a
        cell), use the nearest walkable cell within two cells instead. Planning
        from a wall would make every goal look unreachable."""
        cell = _cell(agent)
        grid = self.world.planning_grid
        if grid.is_walkable(*cell):
            return cell
        options = [(cell[0] + dx, cell[1] + dy) for dx in range(-2, 3) for dy in range(-2, 3)]
        free = [c for c in options if grid.is_walkable(*c)]
        if not free:
            return cell
        return min(free, key=lambda c: (math.dist((c[0] + 0.5, c[1] + 0.5), agent), c))

    def blacklist(self, cell: Cell, now: float) -> None:
        self._blacklist[cell] = now + BLACKLIST_SECONDS

    def _blacklisted(self, cell: Cell, now: float) -> bool:
        until = self._blacklist.get(cell)
        if until is None:
            return False
        if until < now:
            del self._blacklist[cell]
            return False
        return True

    def _time_estimate(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b) * DETOUR_FACTOR / self.speed + self.overhead

    # -- chest targets -------------------------------------------------------------------

    def choose_chest(self, agent: tuple[float, float],
                     exclude: set[int]) -> tuple[Goal, PathResult] | None:
        """Best reachable chest track; marks unreachable spawn priors as such."""
        options = []
        for track in self.world.chest_targets():
            if track.track_id in exclude:
                continue
            t = self._time_estimate(agent, track.estimated_position) + self.open_seconds
            options.append((track.confidence / (t + EPSILON), track))
        options.sort(key=lambda o: (-o[0], o[1].track_id))
        start = self.start_cell(agent)
        for _score, track in options[:MAX_ASTAR_CHECKS]:
            goal_cell = _cell(track.estimated_position)
            result = self.plan(start, goal_cell)
            if result.found:
                path_time = result.cost / self.speed + self.overhead + self.open_seconds
                return (Goal("chest", goal_cell, track.estimated_position,
                             track.confidence / (path_time + EPSILON), track.track_id,
                             track.location_id), result)
            self.mark_unreachable(track)
        return None

    def mark_unreachable(self, track: ObjectTrack) -> None:
        if track.location_id is not None:
            prior = self.world.priors.get(track.location_id)
            if prior is not None and not prior.resolved:
                prior.status = SpawnStatus.UNREACHABLE

    # -- exploration -------------------------------------------------------------------

    def frontier_goals(self, agent: tuple[float, float]) -> list[Goal]:
        occ = self.world.occupancy.cells
        unknown = (occ == UNKNOWN).astype(np.uint8)
        free = (occ == FREE).astype(np.uint8)
        frontier = cv2.bitwise_and(free, cv2.dilate(unknown, _CROSS))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier, connectivity=8)
        if count <= 1:
            return []
        integral = cv2.integral(unknown)
        prior_map = self._prior_map(occ.shape)
        prior_integral = cv2.integral(prior_map) if prior_map is not None else None
        goals = []
        h, w = occ.shape
        for label in range(1, count):
            size = int(stats[label, cv2.CC_STAT_AREA])
            if size < 2:
                continue
            ys, xs = np.nonzero(labels == label)
            cx, cy = centroids[label]
            i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            cell = (int(xs[i]), int(ys[i]))
            x0, x1 = max(0, cell[0] - self.half_w), min(w, cell[0] + self.half_w + 1)
            y0, y1 = max(0, cell[1] - self.half_h), min(h, cell[1] + self.half_h + 1)
            gain = float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
            prior_gain = 0.0
            if prior_integral is not None:
                prior_gain = float(prior_integral[y1, x1] - prior_integral[y0, x1]
                                   - prior_integral[y1, x0] + prior_integral[y0, x0])
            position = (cell[0] + 0.5, cell[1] + 0.5)
            total_gain = gain + self.prior_weight * prior_gain
            score = total_gain / (self._time_estimate(agent, position) + EPSILON)
            goals.append(Goal("frontier", cell, position, score, gain=total_gain))
        self.frontier_evaluations += 1
        return goals

    def _prior_map(self, shape: tuple[int, ...]) -> np.ndarray | None:
        if self.prior_weight <= 0:
            return None
        prior_map = np.zeros(shape, np.float64)
        for prior in self.world.unresolved_priors():
            if prior.status is SpawnStatus.UNKNOWN:
                x, y = int(prior.x), int(prior.y)
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    prior_map[y, x] += prior.probability
        return prior_map

    def inspection_goals(self, agent: tuple[float, float]) -> list[Goal]:
        """Unresolved priors in known space that still need a clear look."""
        goals = []
        for prior in self.world.unresolved_priors():
            if prior.status is not SpawnStatus.UNKNOWN:
                continue
            cell = (int(prior.x), int(prior.y))
            if self.world.occupancy.state(*cell) == UNKNOWN:
                continue   # frontier exploration will reveal it
            gain = self.prior_weight * prior.probability if self.prior_weight > 0 else 1.0
            score = gain / (self._time_estimate(agent, (prior.x, prior.y)) + EPSILON)
            goals.append(Goal("inspect", cell, (prior.x, prior.y), score,
                              location_id=prior.location_id, gain=gain))
        return goals

    def choose_exploration(self, agent: tuple[float, float],
                           now: float) -> tuple[Goal, PathResult] | None:
        candidates = self.frontier_goals(agent) + self.inspection_goals(agent)
        candidates = [g for g in candidates if not self._blacklisted(g.cell, now)]
        candidates.sort(key=lambda g: (-g.score, g.cell))
        start = self.start_cell(agent)
        for goal in candidates[:MAX_ASTAR_CHECKS]:
            result = self.plan(start, goal.cell)
            if result.found:
                return goal, result
            self.blacklist(goal.cell, now)
            if goal.kind == "inspect" and goal.location_id is not None:
                prior = self.world.priors.get(goal.location_id)
                if prior is not None:
                    prior.status = SpawnStatus.UNREACHABLE
        return None

    def prior_resolved(self, location_id: int | None) -> bool:
        if location_id is None:
            return False
        prior: SpawnPrior | None = self.world.priors.get(location_id)
        return prior is None or prior.status is not SpawnStatus.UNKNOWN


def _cell(position: tuple[float, float]) -> Cell:
    return (math.floor(position[0]), math.floor(position[1]))
