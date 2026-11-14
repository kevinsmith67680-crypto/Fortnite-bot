"""Multi-layer navigation grid: the reference environment's stand-in for a NavMesh.

Unreal's Navigation System voxelises the level into a NavMesh and answers
path queries (``FindPathToLocationSynchronously``) and point projections
(``ProjectPointToNavigation``). This module offers the same service for the
Python reference arena:

* Every 0.5 m column can hold several walkable surfaces (ground, upper floors,
  stair steps, platforms). A surface is kept only if a capsule of the agent's
  radius and height fits above it.
* Neighbouring surfaces are connected when the height difference is small
  enough to walk (stairs are traversed like a ramp, as the NavMesh does), and
  diagonal moves may not cut corners.
* Queries run A* on that graph and return smoothed 3D waypoints.

It is part of the *environment* (a permitted low-level navigation service):
it knows the true geometry but nothing about chests.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from chest_hunter.simulation3d.geometry import Box, World3D

NEIGHBOURS: Final = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
Node = tuple[int, int, int]           # (i, j, layer)


@dataclass(frozen=True, slots=True)
class NavPath:
    status: str               # ok, unreachable, start_off_navmesh, goal_off_navmesh
    waypoints: tuple[tuple[float, float, float], ...] = ()
    length: float = 0.0
    expanded: int = 0

    @property
    def found(self) -> bool:
        return self.status == "ok"


class NavGrid3D:
    def __init__(self, world: World3D, *, resolution: float = 0.5, agent_radius: float = 0.35,
                 agent_height: float = 1.8, step_height: float = 0.55,
                 max_climb: float = 0.55, max_layers: int = 4) -> None:
        self.world = world
        self.res = resolution
        self.radius = agent_radius
        self.height = agent_height
        self.step = step_height
        self.max_climb = max_climb
        bx0, by0, bx1, by1 = world.bounds
        self.origin = (bx0, by0)
        self.nx = int(math.ceil((bx1 - bx0) / resolution))
        self.ny = int(math.ceil((by1 - by0) / resolution))
        self.layers = max_layers
        self.surfaces = self._build_surfaces()
        # Python lists of per-column surface tuples: fast scalar access in A*.
        self._cols: list[list[tuple[float, ...]]] = [
            [tuple(float(h) for h in self.surfaces[i, j] if not math.isnan(h))
             for j in range(self.ny)] for i in range(self.nx)]
        self._cache: dict[tuple[Node, Node, int], NavPath] = {}
        self._blocked: frozenset[tuple[int, int]] = frozenset()
        self.queries = 0
        self._component = self._label_components()

    # -- construction -----------------------------------------------------------------

    def _centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.origin[0] + (np.arange(self.nx) + 0.5) * self.res
        ys = self.origin[1] + (np.arange(self.ny) + 0.5) * self.res
        return xs, ys

    def _index_range(self, lo: float, hi: float, origin: float, n: int) -> tuple[int, int]:
        """Columns whose centre lies within [lo, hi]."""
        a = int(math.ceil((lo - origin) / self.res - 0.5))
        b = int(math.floor((hi - origin) / self.res - 0.5))
        return max(0, a), min(n - 1, b)

    def _build_surfaces(self) -> np.ndarray:
        s = np.full((self.nx, self.ny, self.layers), np.nan)
        s[:, :, 0] = self.world.ground_z
        boxes = [b for b in self.world.boxes if b.kind != "ground"]
        # 1. Walkable tops become surfaces (a top within 0.1 m of an existing one replaces it).
        for b in sorted(boxes, key=lambda b: b.z1):
            if not b.walkable:
                continue
            i0, i1 = self._index_range(b.x0, b.x1, self.origin[0], self.nx)
            j0, j1 = self._index_range(b.y0, b.y1, self.origin[1], self.ny)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    col = s[i, j]
                    with np.errstate(invalid="ignore"):
                        col[(col >= b.z0 - 1e-6) & (col < b.z1 - 0.1)] = np.nan  # buried
                    near = np.nonzero(np.abs(col - b.z1) < 0.1)[0]
                    if near.size:
                        col[near[0]] = max(col[near[0]], b.z1)
                        continue
                    free = np.nonzero(np.isnan(col))[0]
                    if free.size:
                        col[free[0]] = b.z1
        # 2. A non-walkable cover just above a surface (water) removes it.
        for b in boxes:
            if b.walkable or b.blocks:
                continue
            i0, i1 = self._index_range(b.x0, b.x1, self.origin[0], self.nx)
            j0, j1 = self._index_range(b.y0, b.y1, self.origin[1], self.ny)
            if i1 < i0 or j1 < j0:
                continue
            sub = s[i0:i1 + 1, j0:j1 + 1]
            sub[(b.z1 >= sub - 1e-6) & (b.z1 <= sub + self.step)] = np.nan
        # 3. Clearance: blocking boxes (inflated by the agent radius) remove surfaces whose
        #    capsule would intersect them above the step height.
        for b in boxes:
            if not b.blocks:
                continue
            i0, i1 = self._index_range(b.x0 - self.radius, b.x1 + self.radius,
                                       self.origin[0], self.nx)
            j0, j1 = self._index_range(b.y0 - self.radius, b.y1 + self.radius,
                                       self.origin[1], self.ny)
            if i1 < i0 or j1 < j0:
                continue
            sub = s[i0:i1 + 1, j0:j1 + 1]
            with np.errstate(invalid="ignore"):
                hit = (b.z1 > sub + self.step) & (b.z0 < sub + self.height)
            sub[hit] = np.nan
        # Out-of-bounds margin: the perimeter is never walkable.
        return s

    # -- helpers -------------------------------------------------------------------------

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor((x - self.origin[0]) / self.res)),
                int(math.floor((y - self.origin[1]) / self.res)))

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return (self.origin[0] + (i + 0.5) * self.res, self.origin[1] + (j + 0.5) * self.res)

    def node_position(self, node: Node) -> tuple[float, float, float]:
        x, y = self.centre(node[0], node[1])
        return (x, y, self._cols[node[0]][node[1]][node[2]])

    def in_grid(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def surfaces_at(self, i: int, j: int) -> tuple[float, ...]:
        return self._cols[i][j] if self.in_grid(i, j) else ()

    def walkable_count(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.surfaces)))

    def project(self, point: tuple[float, float, float], *, horizontal: float = 1.0,
                vertical: float = 1.5) -> Node | None:
        """Nearest graph node to ``point`` within the given extents (ProjectPointToNavigation)."""
        x, y, z = point
        ci, cj = self.cell_of(x, y)
        reach = int(math.ceil(horizontal / self.res))
        best: tuple[float, Node] | None = None
        for i in range(ci - reach, ci + reach + 1):
            for j in range(cj - reach, cj + reach + 1):
                if not self.in_grid(i, j):
                    continue
                cx, cy = self.centre(i, j)
                dh = math.hypot(cx - x, cy - y)
                if dh > horizontal + self.res * 0.71:
                    continue
                if self._blocked and (i, j) in self._blocked:
                    continue
                for layer, h in enumerate(self._cols[i][j]):
                    dz = abs(h - z)
                    if dz > vertical:
                        continue
                    d = dh * dh + (2.0 * dz) ** 2       # prefer the right floor
                    if best is None or d < best[0]:
                        best = (d, (i, j, layer))
        return None if best is None else best[1]

    def _neighbours(self, node: Node) -> list[tuple[Node, float]]:
        i, j, layer = node
        h = self._cols[i][j][layer]
        out = []
        for di, dj in NEIGHBOURS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                continue
            target = self._compatible(ni, nj, h)
            if target is None:
                continue
            if di and dj and (self._compatible(i + di, j, h) is None
                              or self._compatible(i, j + dj, h) is None):
                continue                                  # no corner cutting
            nh = self._cols[ni][nj][target]
            horizontal = self.res * (1.41421356 if di and dj else 1.0)
            out.append(((ni, nj, target), math.hypot(horizontal, nh - h)))
        return out

    def _compatible(self, i: int, j: int, h: float) -> int | None:
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return None
        if self._blocked and (i, j) in self._blocked:
            return None
        best = None
        best_dz = self.max_climb + 1e-9
        for layer, nh in enumerate(self._cols[i][j]):
            dz = abs(nh - h)
            if dz <= best_dz:
                best, best_dz = layer, dz
        return best

    # -- queries --------------------------------------------------------------------------

    def dynamic_cells(self, boxes: list[Box]) -> frozenset[tuple[int, int]]:
        """Columns blocked by dynamic obstacles (chests), inflated by the agent radius,
        like actors that carve Unreal's dynamic NavMesh."""
        out: set[tuple[int, int]] = set()
        for b in boxes:
            rad = self.radius
            i0, i1 = self._index_range(b.x0 - rad, b.x1 + rad, self.origin[0], self.nx)
            j0, j1 = self._index_range(b.y0 - rad, b.y1 + rad, self.origin[1], self.ny)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    out.add((i, j))
        return frozenset(out)

    def find_path(self, start: tuple[float, float, float], goal: tuple[float, float, float], *,
                  goal_tolerance: float = 0.5, max_expansions: int = 200_000,
                  blocked: frozenset[tuple[int, int]] = frozenset()) -> NavPath:
        """A* path between two world points (each projected onto the graph first)."""
        self.queries += 1
        self._blocked = blocked
        s = self.project(start, horizontal=1.0, vertical=1.2)
        if s is None:
            return NavPath("start_off_navmesh")
        g = self.project(goal, horizontal=max(0.5, goal_tolerance), vertical=1.5)
        if g is None:
            return NavPath("goal_off_navmesh")
        if self._component[s] != self._component[g]:
            return NavPath("unreachable")          # different islands: no search needed
        key = (s, g, hash(blocked))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._astar(s, g, max_expansions)
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _astar(self, start: Node, goal: Node, max_expansions: int) -> NavPath:
        gx, gy, gz = self.node_position(goal)
        def heuristic(n: Node) -> float:
            x, y, z = self.node_position(n)
            return math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)
        open_heap: list[tuple[float, int, Node]] = [(heuristic(start), 0, start)]
        best: dict[Node, float] = {start: 0.0}
        parent: dict[Node, Node] = {}
        counter = 0
        expanded = 0
        closed: set[Node] = set()
        while open_heap:
            _, _, node = heapq.heappop(open_heap)
            if node in closed:
                continue
            if node == goal:
                return self._finish(node, parent, best[node], expanded)
            closed.add(node)
            expanded += 1
            if expanded > max_expansions:
                break
            base = best[node]
            for nb, cost in self._neighbours(node):
                if nb in closed:
                    continue
                g = base + cost
                if g < best.get(nb, math.inf):
                    best[nb] = g
                    parent[nb] = node
                    counter += 1
                    heapq.heappush(open_heap, (g + heuristic(nb), counter, nb))
        return NavPath("unreachable", expanded=expanded)

    def _finish(self, node: Node, parent: dict[Node, Node], length: float,
                expanded: int) -> NavPath:
        nodes = [node]
        while nodes[-1] in parent:
            nodes.append(parent[nodes[-1]])
        nodes.reverse()
        points = [self.node_position(n) for n in nodes]
        return NavPath("ok", tuple(self._smooth(points)), length, expanded)

    def _smooth(self, points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        """String-pulling: drop waypoints whose neighbours see each other on the grid."""
        if len(points) <= 2:
            return points
        out = [points[0]]
        anchor = 0
        k = 2
        while k < len(points):
            if not self._segment_clear(points[anchor], points[k]):
                out.append(points[k - 1])
                anchor = k - 1
            k += 1
        out.append(points[-1])
        return out

    def _segment_clear(self, a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(length / (self.res * 0.5)))
        prev_h = a[2]
        for k in range(1, steps + 1):
            t = k / steps
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            i, j = self.cell_of(x, y)
            if not self.in_grid(i, j):
                return False
            layer = self._compatible(i, j, prev_h)
            if layer is None:
                return False
            h = self._cols[i][j][layer]
            expected = a[2] + (b[2] - a[2]) * t
            if abs(h - expected) > 0.6:
                return False
            # Keep a margin from walls: the 4-neighbourhood must be walkable too.
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if self._compatible(i + di, j + dj, h) is None:
                    return False
            prev_h = h
        return True

    def raycast(self, origin: tuple[float, float, float], bearing_deg: float,
                max_distance: float, blocked: frozenset[tuple[int, int]] = frozenset()) -> float:
        """Distance a walker can go along a horizontal bearing before leaving the walkable
        surface it starts on (NavigationRaycast). Height changes are followed (stairs)."""
        self._blocked = blocked
        s = self.project(origin, horizontal=0.6, vertical=1.2)
        if s is None:
            return 0.0
        h = self._cols[s[0]][s[1]][s[2]]
        rad = math.radians(bearing_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        step = self.res * 0.5
        travelled = 0.0
        x, y = origin[0], origin[1]
        while travelled < max_distance:
            nx_, ny_ = x + dx * step, y + dy * step
            i, j = self.cell_of(nx_, ny_)
            layer = self._compatible(i, j, h)
            if layer is None:
                return travelled
            h = self._cols[i][j][layer]
            x, y = nx_, ny_
            travelled += step
        return max_distance

    def _label_components(self) -> dict[Node, int]:
        self._blocked = frozenset()
        labels: dict[Node, int] = {}
        label = 0
        for i in range(self.nx):
            for j in range(self.ny):
                for layer in range(len(self._cols[i][j])):
                    root = (i, j, layer)
                    if root in labels:
                        continue
                    labels[root] = label
                    stack = [root]
                    while stack:
                        node = stack.pop()
                        for nb, _ in self._neighbours(node):
                            if nb not in labels:
                                labels[nb] = label
                                stack.append(nb)
                    label += 1
        return labels

    def same_component(self, a: Node, b: Node) -> bool:
        return self._component.get(a) == self._component.get(b)

    def reachable_from(self, start: tuple[float, float, float]) -> set[Node]:
        """Every node connected to ``start`` (flood fill; used for validation)."""
        s = self.project(start)
        if s is None:
            return set()
        seen = {s}
        stack = [s]
        while stack:
            node = stack.pop()
            for nb, _ in self._neighbours(node):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen
