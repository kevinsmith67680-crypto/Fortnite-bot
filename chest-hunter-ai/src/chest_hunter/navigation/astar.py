"""Grid A* pathfinding, cached path lookups and shortest-distance fields.

Movement model (shared by A* and the distance fields so costs agree):

* 4-directional moves cost 1 and use the Manhattan heuristic.
* With ``allow_diagonal`` diagonal moves cost sqrt(2) and the octile
  heuristic is used. Both heuristics are admissible and consistent.
* With ``prevent_corner_cutting`` a diagonal move is only allowed when both
  adjacent orthogonal cells are walkable, so paths never squeeze between two
  blocked cells or clip a wall corner.

Ties are broken deterministically by ``(f, h, insertion order)`` and the
neighbour order is fixed, so identical inputs always yield identical paths.
"""

from __future__ import annotations

import heapq
import math
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Final

from chest_hunter.simulation.collision import Cell, GridMap

SQRT2: Final = math.sqrt(2.0)
INF: Final = math.inf

_CARDINAL: Final = ((1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0))
_DIAGONAL: Final = ((1, 1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2), (1, -1, SQRT2))


class FailureReason:
    START_OUT_OF_BOUNDS = "start_out_of_bounds"
    GOAL_OUT_OF_BOUNDS = "goal_out_of_bounds"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True)
class PathResult:
    """Outcome of a path search. ``path`` includes both start and goal cells."""

    found: bool
    path: tuple[Cell, ...]
    cost: float
    expanded: int
    reason: str | None = None

    @classmethod
    def failure(cls, reason: str, expanded: int = 0) -> PathResult:
        return cls(False, (), INF, expanded, reason)


def neighbour_moves(allow_diagonal: bool) -> tuple[tuple[int, int, float], ...]:
    return _CARDINAL + _DIAGONAL if allow_diagonal else _CARDINAL


def heuristic(dx: int, dy: int, allow_diagonal: bool) -> float:
    dx, dy = abs(dx), abs(dy)
    if allow_diagonal:
        return dx + dy + (SQRT2 - 2.0) * min(dx, dy)
    return float(dx + dy)


def astar(
    grid: GridMap,
    start: Cell,
    goal: Cell,
    *,
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
    penalty_cells: bytearray | None = None,
    penalty: float = 0.0,
) -> PathResult:
    """Find a shortest path from ``start`` to ``goal`` (cell coordinates).

    ``penalty_cells`` (row-major, non-zero = penalised) adds ``penalty`` to the
    cost of entering such a cell, e.g. to prefer explored space over unknown
    space. Penalties are non-negative, so the heuristic stays admissible.
    """
    if not grid.in_bounds(*start):
        return PathResult.failure(FailureReason.START_OUT_OF_BOUNDS)
    if not grid.in_bounds(*goal):
        return PathResult.failure(FailureReason.GOAL_OUT_OF_BOUNDS)
    if grid.is_blocked(*start):
        return PathResult.failure(FailureReason.START_BLOCKED)
    if grid.is_blocked(*goal):
        return PathResult.failure(FailureReason.GOAL_BLOCKED)
    if start == goal:
        return PathResult(True, (start,), 0.0, 0)

    width, height = grid.width, grid.height
    cells = grid.cells
    size = width * height
    moves = neighbour_moves(allow_diagonal)
    gx, gy = goal
    goal_index = gy * width + gx
    start_index = start[1] * width + start[0]

    g_score = [INF] * size
    parent = [-1] * size
    closed = bytearray(size)
    g_score[start_index] = 0.0
    h0 = heuristic(start[0] - gx, start[1] - gy, allow_diagonal)
    open_heap: list[tuple[float, float, int, int]] = [(h0, h0, 0, start_index)]
    sequence = 0
    expanded = 0
    diag_bonus = SQRT2 - 2.0
    penalised = penalty_cells if penalty_cells is not None and penalty > 0 else None

    while open_heap:
        _, _, _, node = heapq.heappop(open_heap)
        if closed[node]:
            continue
        if node == goal_index:
            path: list[Cell] = []
            while node != -1:
                path.append((node % width, node // width))
                node = parent[node]
            path.reverse()
            return PathResult(True, tuple(path), g_score[goal_index], expanded)
        closed[node] = 1
        expanded += 1
        x, y = node % width, node // width
        g_node = g_score[node]
        for dx, dy, cost in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbour = ny * width + nx
            if cells[neighbour] or closed[neighbour]:
                continue
            if dx and dy and prevent_corner_cutting and (
                cells[y * width + nx] or cells[ny * width + x]
            ):
                continue
            tentative = g_node + cost
            if penalised is not None and penalised[neighbour]:
                tentative += penalty
            if tentative < g_score[neighbour] - 1e-12:
                g_score[neighbour] = tentative
                parent[neighbour] = node
                hx, hy = abs(nx - gx), abs(ny - gy)
                h = (hx + hy + diag_bonus * (hx if hx < hy else hy)) if allow_diagonal \
                    else float(hx + hy)
                sequence += 1
                heapq.heappush(open_heap, (tentative + h, h, sequence, neighbour))
    return PathResult.failure(FailureReason.UNREACHABLE, expanded)


def distance_field(
    grid: GridMap,
    source: Cell,
    *,
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
) -> array[float]:
    """Shortest path cost from ``source`` to every cell (Dijkstra; ``inf`` if unreachable).

    Moves are symmetric, so ``field[b]`` is also the cost from ``b`` to
    ``source``: one field per chest answers "how far is this chest from
    anywhere" in O(1).
    """
    width, height = grid.width, grid.height
    size = width * height
    dist = array("d", [INF]) * size
    if not grid.in_bounds(*source) or grid.is_blocked(*source):
        return dist
    cells = grid.cells
    moves = neighbour_moves(allow_diagonal)
    start_index = source[1] * width + source[0]
    dist[start_index] = 0.0
    done = bytearray(size)
    heap: list[tuple[float, int]] = [(0.0, start_index)]
    while heap:
        d, node = heapq.heappop(heap)
        if done[node]:
            continue
        done[node] = 1
        x, y = node % width, node // width
        for dx, dy, cost in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbour = ny * width + nx
            if cells[neighbour] or done[neighbour]:
                continue
            if dx and dy and prevent_corner_cutting and (
                cells[y * width + nx] or cells[ny * width + x]
            ):
                continue
            nd = d + cost
            if nd < dist[neighbour] - 1e-12:
                dist[neighbour] = nd
                heapq.heappush(heap, (nd, neighbour))
    return dist


class _GridBoundCache:
    """Base for caches that must be dropped when the grid object or version changes."""

    def __init__(self) -> None:
        self._grid: GridMap | None = None
        self._version = -1
        self.invalidations = 0

    def _sync(self, grid: GridMap) -> bool:
        """Bind to ``grid``; return True if existing entries were dropped."""
        if grid is self._grid and grid.version == self._version:
            return False
        stale = self._grid is not None
        self._grid = grid
        self._version = grid.version
        self._clear()
        if stale:
            self.invalidations += 1
        return stale

    def _clear(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class PathFinder(_GridBoundCache):
    """A* with an LRU cache of results for a static grid.

    The cache is invalidated automatically when the grid object is replaced
    or its ``version`` changes (obstacles added or removed).
    """

    def __init__(self, *, allow_diagonal: bool = True, prevent_corner_cutting: bool = True,
                 cache_paths: bool = True, cache_size: int = 2048,
                 penalty_cells: bytearray | None = None, penalty: float = 0.0) -> None:
        """``penalty_cells`` must change only together with the grid's ``version``."""
        super().__init__()
        self.allow_diagonal = allow_diagonal
        self.prevent_corner_cutting = prevent_corner_cutting
        self.penalty_cells = penalty_cells
        self.penalty = penalty
        self.cache_paths = cache_paths
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[Cell, Cell], PathResult] = OrderedDict()
        self.searches = 0
        self.cache_hits = 0

    def _clear(self) -> None:
        self._cache.clear()

    def find_path(self, grid: GridMap, start: Cell, goal: Cell) -> PathResult:
        self._sync(grid)
        key = (start, goal)
        if self.cache_paths:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                self._cache.move_to_end(key)
                return cached
        result = astar(grid, start, goal, allow_diagonal=self.allow_diagonal,
                       prevent_corner_cutting=self.prevent_corner_cutting,
                       penalty_cells=self.penalty_cells, penalty=self.penalty)
        self.searches += 1
        if self.cache_paths:
            self._cache[key] = result
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def invalidate(self) -> None:
        self._cache.clear()
        self.invalidations += 1


class DistanceOracle(_GridBoundCache):
    """Caches one :func:`distance_field` per target cell for a static grid."""

    def __init__(self, *, allow_diagonal: bool = True, prevent_corner_cutting: bool = True,
                 max_fields: int = 512) -> None:
        super().__init__()
        self.allow_diagonal = allow_diagonal
        self.prevent_corner_cutting = prevent_corner_cutting
        self.max_fields = max_fields
        self._fields: OrderedDict[Cell, array[float]] = OrderedDict()
        self.fields_computed = 0

    def _clear(self) -> None:
        self._fields.clear()

    def field(self, grid: GridMap, target: Cell) -> array[float]:
        self._sync(grid)
        cached = self._fields.get(target)
        if cached is not None:
            self._fields.move_to_end(target)
            return cached
        computed = distance_field(grid, target, allow_diagonal=self.allow_diagonal,
                                  prevent_corner_cutting=self.prevent_corner_cutting)
        self.fields_computed += 1
        self._fields[target] = computed
        if len(self._fields) > self.max_fields:
            self._fields.popitem(last=False)
        return computed

    def distance(self, grid: GridMap, source: Cell, target: Cell) -> float:
        """Shortest path cost between two cells (``inf`` if unreachable or out of bounds)."""
        if not grid.in_bounds(*source):
            return INF
        return self.field(grid, target)[source[1] * grid.width + source[0]]

    def warm(self, grid: GridMap, targets: list[Cell]) -> None:
        """Precompute fields for ``targets`` (e.g. every chest cell of a map)."""
        for target in targets:
            self.field(grid, target)
